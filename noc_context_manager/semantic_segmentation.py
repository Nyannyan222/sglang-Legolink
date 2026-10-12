"""Source-span-only semantic segmentation providers.

ARCHITECTURE_STAGE: 3 -- Semantic Segmentation
CANONICAL_ENTRYPOINT: SemanticSegmentationPipeline, imported from this module.
    The class is *defined* in ``segmentation_validation.py`` because it needs the
    validator and ``segmentation_validation`` imports this module; a module-level
    import here would be circular. The module ``__getattr__`` at the bottom
    resolves the name lazily, so
    ``from noc_context_manager.semantic_segmentation import
    SemanticSegmentationPipeline, SemanticSegmentationValidator`` is the one
    import a caller of Stage 3 needs. It is the same class object, not a copy.
DELEGATED_HELPERS: segmentation_validation.py (validator + pipeline definition),
    atomic_units.py, segmentation_prompts.py, current_llm_segmenter.py,
    current_llm_guardrails.py,
    terminal_bench_integration/frozen_segmentation.py
"""

from __future__ import annotations

import hashlib
import copy
import json
import re
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

from .session import CONTEXT_FIELDS, ExtractedSource, FieldExtraction, SourceSpan
from .tokenizer import TextTokenizer


from .atomic_units import (
    AtomicUnitBuilder, CLASSIFICATION_PROMPT, CLASSIFICATION_PROMPT_VERSION,
    GROUPING_PROMPT, GROUPING_SCHEMA, SPLIT_POINT_VALIDATION_VERSION,
    SPLIT_POINT_VALIDATION_VERSIONS, UnitGroupingError, classification_schema,
    validate_segment_labels, validate_split_points,
)
from .tokenizer import RegexTokenizer, SGLangHTTPTokenizer
from .segmentation_prompts import (
    SegmentationPromptVariant, resolve_prompt_variant, segmentation_prompt,
)

SEGMENTATION_SYSTEM_PROMPT = GROUPING_PROMPT
SEGMENTATION_RESPONSE_SCHEMA = GROUPING_SCHEMA


@dataclass(frozen=True)
class SegmentProposal:
    title: str
    context_field: str
    source_spans: list[SourceSpan]
    question_or_objective: str
    semantic_independence_reason: str
    is_mutable: bool
    reuse_candidate: bool
    confidence: float
    llm_semantic_role: str | None = None
    llm_semantic_topic: str | None = None
    segmentation_decision_source: str | None = None
    classification_decision_source: str | None = None
    unit_start_id: str | None = None
    unit_end_id: str | None = None
    classification_model: str | None = None
    classification_prompt_version: str | None = None
    classification_repair_count: int = 0
    classification_fallback: bool = False
    classification_status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["source_spans"] = [span.to_dict() for span in self.source_spans]
        return row


class SemanticSegmenter(ABC):
    model_name = "unknown"

    @abstractmethod
    def segment_field(
        self, context_field: str, sources: list[ExtractedSource]
    ) -> list[SegmentProposal]: ...

    def segment_incremental_update(
        self, extraction: FieldExtraction, changed_source_ids: set[str] | None = None
    ) -> list[SegmentProposal]:
        changed = changed_source_ids or set(extraction.source_index())
        result: list[SegmentProposal] = []
        for context_field, values in extraction.fields.items():
            selected = [source for source in values if source.source_id in changed]
            if selected:
                result.extend(self.segment_field(context_field, selected))
        return result

    def repair_incremental_update(
        self,
        extraction: FieldExtraction,
        changed_source_ids: set[str] | None,
        validation_errors: list[str],
    ) -> list[SegmentProposal]:
        return self.segment_incremental_update(extraction, changed_source_ids)

    def segment_field_with_feedback(
        self,
        context_field: str,
        sources: list[ExtractedSource],
        validation_errors: list[str] | None = None,
    ) -> list[SegmentProposal]:
        return self.segment_field(context_field, sources)

    def validate_response(self, value: Any) -> list[SegmentProposal]:
        if not isinstance(value, dict) or not isinstance(value.get("sub_contexts"), list):
            raise ValueError("segmentation response requires sub_contexts array")
        result: list[SegmentProposal] = []
        required = {
            "title",
            "context_field",
            "source_spans",
            "question_or_objective",
            "semantic_independence_reason",
            "is_mutable",
            "reuse_candidate",
            "confidence",
        }
        for index, row in enumerate(value["sub_contexts"]):
            if not isinstance(row, dict) or not required.issubset(row):
                raise ValueError(f"sub_contexts[{index}] violates schema")
            if not isinstance(row["source_spans"], list) or not row["source_spans"]:
                raise ValueError(f"sub_contexts[{index}] requires source spans")
            confidence = float(row["confidence"])
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("confidence must be in [0,1]")
            result.append(
                SegmentProposal(
                    title=str(row["title"]),
                    context_field=str(row["context_field"]),
                    source_spans=[SourceSpan.from_dict(span) for span in row["source_spans"]],
                    question_or_objective=str(row["question_or_objective"]),
                    semantic_independence_reason=str(row["semantic_independence_reason"]),
                    is_mutable=bool(row["is_mutable"]),
                    reuse_candidate=bool(row["reuse_candidate"]),
                    confidence=confidence,
                )
            )
        return result


class DeterministicFieldSegmenter(SemanticSegmenter):
    """Safe baseline/fallback: one source-backed block per extracted event/file."""

    model_name = "deterministic-field-v1"

    _MUTABLE_FIELDS = {"user_prompt", "tool_history_output", "conversation_history"}

    def segment_field(
        self, context_field: str, sources: list[ExtractedSource]
    ) -> list[SegmentProposal]:
        return [
            SegmentProposal(
                title=str(source.metadata.get("title") or source.source_path or source.source_id),
                context_field=context_field,
                source_spans=[source.span()],
                question_or_objective=str(
                    source.metadata.get("objective") or source.metadata.get("title") or context_field
                ),
                semantic_independence_reason="deterministic field/source boundary",
                is_mutable=context_field in self._MUTABLE_FIELDS,
                reuse_candidate=context_field not in self._MUTABLE_FIELDS,
                confidence=1.0,
            )
            for source in sources
            if source.raw_text
        ]

    def segment_source_with_token_budget(
        self,
        context_field: str,
        source: ExtractedSource,
        *,
        tokenizer: TextTokenizer,
        max_tokens: int,
    ) -> list[SegmentProposal]:
        """Split one oversized source into exact, contiguous source spans."""

        if max_tokens <= 0:
            raise ValueError("segmentation token budget must be positive")
        text = source.raw_text
        if not text:
            return []
        if tokenizer.count(text) <= max_tokens:
            return self.segment_field(context_field, [source])

        ranges: list[tuple[int, int]] = []
        start = 0
        while start < len(text):
            low = start + 1
            high = len(text)
            best = start
            while low <= high:
                middle = (low + high) // 2
                if tokenizer.count(text[start:middle]) <= max_tokens:
                    best = middle
                    low = middle + 1
                else:
                    high = middle - 1
            if best <= start:
                best = start + 1
            if best < len(text):
                candidates = (
                    text.rfind("\n\n", start + 1, best + 1),
                    text.rfind("\n", start + 1, best + 1),
                    text.rfind(" ", start + 1, best + 1),
                )
                boundary = max(candidates)
                if boundary > start:
                    best = min(
                        best,
                        boundary
                        + (2 if text.startswith("\n\n", boundary) else 1),
                    )
            ranges.append((start, best))
            start = best

        mutable = context_field in self._MUTABLE_FIELDS
        title = str(
            source.metadata.get("title") or source.source_path or source.source_id
        )
        objective = str(
            source.metadata.get("objective")
            or source.metadata.get("title")
            or context_field
        )
        return [
            SegmentProposal(
                title=f"{title} [{index + 1}/{len(ranges)}]",
                context_field=context_field,
                source_spans=[
                    SourceSpan(
                        source_id=source.source_id,
                        start=start,
                        end=end,
                        source_path=source.source_path,
                    )
                ],
                question_or_objective=objective,
                semantic_independence_reason=(
                    "deterministic source window selected after preliminary "
                    "segmentation-budget analysis"
                ),
                is_mutable=mutable,
                reuse_candidate=not mutable,
                confidence=1.0,
            )
            for index, (start, end) in enumerate(ranges)
        ]


class FakeSemanticSegmenter(SemanticSegmenter):
    """Deterministic test provider with injectable proposals or callback."""

    model_name = "fake-semantic-segmenter"

    def __init__(
        self,
        responses: dict[str, list[SegmentProposal] | dict[str, Any]] | None = None,
        callback: Callable[[str, list[ExtractedSource]], list[SegmentProposal]] | None = None,
    ) -> None:
        self.responses = responses or {}
        self.callback = callback
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def segment_field(
        self, context_field: str, sources: list[ExtractedSource]
    ) -> list[SegmentProposal]:
        self.calls.append((context_field, tuple(source.source_id for source in sources)))
        if self.callback:
            return self.callback(context_field, sources)
        response = self.responses.get(context_field)
        if response is None:
            return DeterministicFieldSegmenter().segment_field(context_field, sources)
        if isinstance(response, dict):
            return self.validate_response(response)
        return list(response)


def repair_json_text(text: str) -> str:
    """Conservative repair for code fences, prose wrappers, and trailing commas."""

    value = text.strip()
    value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value)
    start = value.find("{")
    end = value.rfind("}")
    if start >= 0 and end > start:
        value = value[start : end + 1]
    return re.sub(r",\s*([}\]])", r"\1", value)


class OpenAICompatibleSemanticSegmenter(SemanticSegmenter):
    def __init__(
        self,
        *,
        api_base: str,
        model: str,
        api_key: str = "",
        timeout: float = 60.0,
        retries: int = 2,
        max_source_chars: int = 120_000,
        cache_dir: str | Path | None = None,
        transport: Callable[[dict[str, Any]], Any] | None = None,
        semantic_classification_enabled: bool = True,
        prompt_variant: SegmentationPromptVariant | str | None = None,
        split_point_validation_version: str = SPLIT_POINT_VALIDATION_VERSION,
    ) -> None:
        self.endpoint = api_base.rstrip("/") + "/chat/completions"
        self.model_name = model
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries
        self.max_source_chars = max_source_chars
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.transport = transport
        # Stage 2 is a labelling pass over segments Stage 1 has already fixed. It is metadata: role and
        # topic reach no cache key, no eligibility verdict, no selector, no planner and no prompt. A
        # formal serving experiment whose treatment is segmentation granularity therefore turns it off,
        # and pays neither its latency nor its failure modes. Stage 1 is unaffected: same prompt, same
        # schema, same split points, same segments.
        self.semantic_classification_enabled = bool(semantic_classification_enabled)
        # Stage-1 instruction only. `None` is P0, the production prompt byte for byte, so the request, the
        # response-cache hash and the segments are unchanged for every caller that names no variant. The
        # schema, atomic units, validation and materialization do not read this.
        self.segmentation_prompt_variant = resolve_prompt_variant(prompt_variant)
        self.segmentation_system_prompt = segmentation_prompt(self.segmentation_prompt_variant)
        # How split points are validated, recorded on every Stage-1 diagnostic. The current version
        # collapses a back-to-back repeat of one boundary ID; the legacy version exists so a recording
        # made under it can be replayed under it.
        if split_point_validation_version not in SPLIT_POINT_VALIDATION_VERSIONS:
            raise ValueError(f"unknown split-point validation version {split_point_validation_version!r}")
        self.split_point_validation_version = split_point_validation_version
        self.duplicate_split_ids_normalized = 0
        self.llm_calls = 0
        self.cache_hits = 0
        self.segmentation_llm_calls = 0
        self.classification_llm_calls = 0
        self.segmentation_cache_hits = 0
        self.classification_cache_hits = 0
        self.segmentation_repairs = 0
        self.classification_repairs = 0
        self.classification_failures = 0
        self.internal_fallbacks = 0
        self.response_diagnostics: list[dict[str, Any]] = []
        self.atomic_tokenizer = RegexTokenizer() if transport else SGLangHTTPTokenizer(api_base=api_base, model_id=model)
        self._unit_cache = {}
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _request(self, payload: dict[str, Any]) -> Any:
        self.llm_calls += 1
        if self.transport:
            return self.transport(payload)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _source_units(self, source):
        key = (self.atomic_tokenizer.fingerprint, source.source_id, source.context_field, source.raw_text)
        if key not in self._unit_cache:
            self._unit_cache[key] = AtomicUnitBuilder(self.atomic_tokenizer).build(source)
        return self._unit_cache[key]

    def _materialize_response(self, response, context_field, sources, diagnostic):
        """Materialize the frozen split-point-only Stage-1 response."""
        source, = sources
        units = self._source_units(source)
        diagnostic["raw_response"] = copy.deepcopy(response)
        diagnostic["atomic_units"] = [u.to_dict() for u in units]
        diagnostic["raw_proposal_count"] = (
            len(response.get("split_points", [])) + 1 if isinstance(response, dict) else 0
        )
        try:
            groups = validate_split_points(
                response, units, diagnostics=diagnostic, version=self.split_point_validation_version)
        except UnitGroupingError as exc:
            diagnostic["unit_error_code"] = exc.code
            diagnostic["error"] = str(exc)
            raise
        mutable = source.context_field in DeterministicFieldSegmenter._MUTABLE_FIELDS
        proposals = []
        for first, last, reason in groups:
            proposals.append(SegmentProposal(
                title=f"llm-segment-{first.unit_id}-{last.unit_id}",
                context_field=source.context_field,
                source_spans=[SourceSpan(source.source_id, first.start_char, last.end_char, source_path=source.source_path)],
                question_or_objective=str(source.metadata.get("objective") or source.metadata.get("title") or source.context_field),
                semantic_independence_reason=reason,
                is_mutable=mutable, reuse_candidate=not mutable, confidence=1.0,
                segmentation_decision_source=diagnostic["segmentation_decision_source"],
                unit_start_id=first.unit_id, unit_end_id=last.unit_id,
            ))
        diagnostic["context_field_source"] = "field_first_extractor"
        diagnostic["validated_context_fields"] = [p.context_field for p in proposals]
        # The split points the geometry was built from; `raw_response` keeps what the model sent.
        diagnostic["selected_split_points"] = copy.deepcopy(diagnostic["validated_split_points"])
        self.duplicate_split_ids_normalized += diagnostic["duplicate_split_ids_normalized"]
        diagnostic["selected_unit_ranges"] = [[first.unit_id, last.unit_id] for first, last, _ in groups]
        diagnostic["host_spans"] = [p.to_dict()["source_spans"] for p in proposals]
        diagnostic["mutability_policy"] = [{"context_field": source.context_field,
            "model_is_mutable": None, "model_reuse_candidate": None,
            "effective_is_mutable": mutable, "effective_reuse_candidate": not mutable,
            "authority": "host_field_policy"} for _ in proposals]
        return proposals

    def _grouping_schema(self, units):
        schema = copy.deepcopy(SEGMENTATION_RESPONSE_SCHEMA)
        schema['properties']['split_points']['items']['properties']['after_unit_id']['enum'] = [u.unit_id for u in units[:-1]]
        return schema

    def _grouping_request_options(self, units):
        return {}

    def _trivial_grouping_response(self):
        return {"split_points": []}

    def _parse_grouping_response(self, response):
        return json.loads(repair_json_text(response))

    def _load_cached_grouping_response(self, response):
        return json.loads(response)

    _grouping_feedback_name = "split_points"

    @staticmethod
    def _response_value(response):
        if isinstance(response, dict) and "choices" in response:
            response = response["choices"][0]["message"]["content"]
        if isinstance(response, str):
            response = json.loads(repair_json_text(response))
        return response

    @staticmethod
    def _segment_text(source, proposal):
        return "".join(source.raw_text[span.start:span.end] for span in proposal.source_spans)

    def classify_materialized_segments(
        self, context_field, source, proposals, *, use_cache=True
    ):
        """Stage 2: classify fixed host segments without changing their geometry."""
        if not proposals:
            return list(proposals)
        if not self.semantic_classification_enabled:
            # No request, no cache read, no label. `disabled` is recorded on every segment and in the
            # diagnostics, because a run that did not classify must not read as one that classified
            # successfully -- and must not read as one that failed either.
            self.response_diagnostics.append({
                "stage": "classification", "classification_status": "disabled",
                "classification_decision_source": "disabled", "llm_call_count": 0,
                "cache_hit": False, "segment_count": len(proposals),
            })
            return [replace(
                proposal,
                llm_semantic_role=None,
                llm_semantic_topic=None,
                classification_decision_source="disabled",
                classification_model=None,
                classification_prompt_version=None,
                classification_repair_count=0,
                classification_fallback=False,
                classification_status="disabled",
            ) for proposal in proposals]
        segment_ids = [f"S{index:04d}" for index in range(1, len(proposals) + 1)]
        segments = [
            {"segment_id": segment_id, "context_field": context_field,
             "text": self._segment_text(source, proposal)}
            for segment_id, proposal in zip(segment_ids, proposals)
        ]
        schema = classification_schema(segment_ids)
        rendered = json.dumps({"segments": segments}, ensure_ascii=False)
        cache_hash = hashlib.sha256(
            f"{CLASSIFICATION_PROMPT}\0{json.dumps(schema, sort_keys=True)}\0"
            f"{self.model_name}\0{rendered}".encode("utf-8")
        ).hexdigest()
        cache_path = self.cache_dir / f"classification-{cache_hash}.json" if self.cache_dir else None
        diagnostic = {
            "stage": "classification", "classification_status": "pending",
            "classification_decision_source": "llm", "classification_model": self.model_name,
            "classification_prompt_version": CLASSIFICATION_PROMPT_VERSION,
            "expected_segment_ids": segment_ids, "segments": copy.deepcopy(segments),
            "classification_repair_count": 0, "classification_fallback": False,
        }
        self.response_diagnostics.append(diagnostic)
        if use_cache and cache_path and cache_path.exists():
            try:
                response = json.loads(cache_path.read_text(encoding="utf-8"))
                labels = validate_segment_labels(response, segment_ids)
                self.cache_hits += 1
                self.classification_cache_hits += 1
                diagnostic.update({"cache_hit": True, "classification_status": "success",
                                   "raw_response": copy.deepcopy(response),
                                   "received_segment_ids": list(labels)})
                return self._attach_classification(proposals, segment_ids, labels, 0)
            except (ValueError, json.JSONDecodeError):
                pass
        diagnostic["cache_hit"] = False
        previous_error = None
        previous_received = []
        for classification_attempt in range(2):
            repair = ""
            if classification_attempt:
                self.classification_repairs += 1
                diagnostic["classification_repair_count"] = 1
                repair = (
                    "\n\nRepair the previous classification response.\n"
                    f"Expected exactly: {', '.join(segment_ids)}\n"
                    f"Your previous response returned: {', '.join(previous_received) if previous_received else '(none)'}\n"
                    f"Exact schema error: {previous_error}\n"
                    "Return labels for exactly the expected IDs. Do not add, remove, merge, or split segments."
                )
            payload = {
                "model": self.model_name, "temperature": 0,
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": "semantic_subcontext_classification", "strict": True, "schema": schema}},
                "messages": [
                    {"role": "system", "content": CLASSIFICATION_PROMPT},
                    {"role": "user", "content": rendered + repair},
                ],
            }
            diagnostic.setdefault("requests", []).append(copy.deepcopy(payload))
            try:
                self.classification_llm_calls += 1
                raw = self._request(payload)
                diagnostic.setdefault("raw_http_responses", []).append(copy.deepcopy(raw))
                response = self._response_value(raw)
                diagnostic.setdefault("raw_responses", []).append(copy.deepcopy(response))
                labels = validate_segment_labels(response, segment_ids)
                diagnostic.update({"classification_status": "success",
                                   "received_segment_ids": list(labels)})
                if use_cache and cache_path:
                    cache_path.write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
                return self._attach_classification(
                    proposals, segment_ids, labels, classification_attempt
                )
            except Exception as exc:
                previous_error = f"{getattr(exc, 'code', type(exc).__name__)}: {exc}"
                previous_received = list(getattr(exc, "received_segment_ids", ()))
                diagnostic.setdefault("errors", []).append(previous_error)
                diagnostic.setdefault("received_segment_id_attempts", []).append(previous_received)
        self.classification_failures += 1
        diagnostic["classification_status"] = "failed"
        diagnostic["error"] = previous_error
        return [replace(
            proposal, classification_status="failed",
            classification_decision_source="llm", classification_model=self.model_name,
            classification_prompt_version=CLASSIFICATION_PROMPT_VERSION,
            classification_repair_count=1, classification_fallback=False,
        ) for proposal in proposals]

    def _attach_classification(self, proposals, segment_ids, labels, repair_count):
        return [replace(
            proposal,
            llm_semantic_role=labels[segment_id]["semantic_role"],
            llm_semantic_topic=labels[segment_id]["semantic_topic"].strip(),
            classification_decision_source="llm", classification_model=self.model_name,
            classification_prompt_version=CLASSIFICATION_PROMPT_VERSION,
            classification_repair_count=repair_count, classification_fallback=False,
            classification_status="success",
        ) for proposal, segment_id in zip(proposals, segment_ids)]

    def _segment_field(
        self,
        context_field: str,
        sources: list[ExtractedSource],
        validation_errors: list[str] | None = None,
    ) -> list[SegmentProposal]:
        if context_field not in CONTEXT_FIELDS or any(s.context_field != context_field for s in sources):
            raise ValueError("field_crossing: invalid field-first segmentation input")
        # One source per request: unit IDs are stable locally and cannot alias another source.
        if len(sources) != 1:
            return [p for source in sources for p in self._segment_field(context_field,[source],validation_errors)]
        source, = sources
        if not source.raw_text:
            return []
        units = self._source_units(source)
        # N units have N-1 internal boundaries. With one unit Stage 1 is host-trivial.
        if len(units) == 1:
            diagnostic = {"cache_hit": False,
                          "validation_feedback": validation_errors or [],
                          "segmentation_decision_source": "host_trivial_single_unit",
                          "llm_call_count": 0, "stage": "segmentation"}
            self.response_diagnostics.append(diagnostic)
            proposals = self._materialize_response(self._trivial_grouping_response(), context_field, sources, diagnostic)
            return self.classify_materialized_segments(context_field, source, proposals)
        rendered = json.dumps({"units":[{"unit_id":u.unit_id,"text":u.raw_text} for u in units]},ensure_ascii=False)
        schema = self._grouping_schema(units)
        if len(rendered) > self.max_source_chars:
            self.internal_fallbacks += 1
            return DeterministicFieldSegmenter().segment_field(context_field, sources)
        cache_hash = hashlib.sha256(
            f"{self.segmentation_system_prompt}\0{json.dumps(schema, sort_keys=True)}\0{self.model_name}\0{context_field}\0{rendered}\0{json.dumps(validation_errors or [], sort_keys=True)}".encode("utf-8")
        ).hexdigest()
        cache_path = self.cache_dir / f"{cache_hash}.json" if self.cache_dir else None
        if cache_path and cache_path.exists():
            self.cache_hits += 1
            self.segmentation_cache_hits += 1
            diagnostic = {"cache_hit": True, "validation_feedback": validation_errors or [],
                          "segmentation_decision_source": "llm", "llm_call_count": 0,
                          "stage": "segmentation"}
            self.response_diagnostics.append(diagnostic)
            try:
                proposals = self._materialize_response(self._load_cached_grouping_response(cache_path.read_text(encoding="utf-8")), context_field, sources, diagnostic)
                return self.classify_materialized_segments(context_field, source, proposals)
            except ValueError as exc:
                diagnostic["error"] = str(exc)
                raise
        payload = {
            "model": self.model_name,
            "temperature": 0,
            **self._grouping_request_options(units),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "semantic_subcontexts",
                    "strict": True,
                    "schema": schema,
                },
            },
            "messages": [
                {"role": "system", "content": self.segmentation_system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"context_field={context_field}\n\n{rendered}"
                        + (
                            f"\n\nRepair these validation errors in {self._grouping_feedback_name}:\n- "
                            + "\n- ".join(validation_errors)
                            if validation_errors
                            else ""
                        )
                    ),
                },
            ],
        }
        error: Exception | None = None
        for attempt in range(self.retries + 1):
            diagnostic = {"cache_hit": False, "request": copy.deepcopy(payload),
                          "validation_feedback": validation_errors or [], "transport_attempt": attempt,
                          "segmentation_decision_source": "llm", "llm_call_count": 1,
                          "stage": "segmentation"}
            self.response_diagnostics.append(diagnostic)
            try:
                self.segmentation_llm_calls += 1
                response = self._request(payload)
                diagnostic["raw_http_response"] = copy.deepcopy(response)
                if isinstance(response, dict) and "choices" in response:
                    response = response["choices"][0]["message"]["content"]
                if isinstance(response, str):
                    try:
                        response = self._parse_grouping_response(response)
                    except json.JSONDecodeError as exc:
                        diagnostic["unit_error_code"] = "unit_schema"
                        diagnostic["error"] = str(exc)
                        raise UnitGroupingError("unit_schema", f"Return valid JSON with only a {self._grouping_feedback_name} array.") from exc
                proposals = self._materialize_response(response, context_field, sources, diagnostic)
                if cache_path:
                    cache_path.write_text(
                        json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                return self.classify_materialized_segments(context_field, source, proposals)
            except UnitGroupingError:
                raise
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
                error = exc
                diagnostic["error"] = str(exc)
                if attempt < self.retries:
                    time.sleep(min(2**attempt, 4))
        raise RuntimeError(f"semantic segmentation failed: {error}") from error

    def segment_field(
        self, context_field: str, sources: list[ExtractedSource]
    ) -> list[SegmentProposal]:
        return self._segment_field(context_field, sources)

    def segment_field_with_feedback(
        self,
        context_field: str,
        sources: list[ExtractedSource],
        validation_errors: list[str] | None = None,
    ) -> list[SegmentProposal]:
        return self._segment_field(context_field, sources, validation_errors)

    def repair_incremental_update(
        self,
        extraction: FieldExtraction,
        changed_source_ids: set[str] | None,
        validation_errors: list[str],
    ) -> list[SegmentProposal]:
        changed = changed_source_ids or set(extraction.source_index())
        result: list[SegmentProposal] = []
        for context_field, values in extraction.fields.items():
            selected = [source for source in values if source.source_id in changed]
            if selected:
                result.extend(
                    self._segment_field(context_field, selected, validation_errors)
                )
        return result


#: Names whose definition lives in ``segmentation_validation`` but which are part
#: of the Stage 3 public surface. Resolved on first access (PEP 562) to avoid the
#: import cycle; each resolves to the very object ``segmentation_validation``
#: defines.
_STAGE3_LAZY_EXPORTS = (
    "SemanticSegmentationPipeline",
    "SemanticSegmentationValidator",
    "ValidatedSegmentation",
    "SegmentationValidationReport",
    "ValidationStatus",
)


def __getattr__(name: str) -> Any:
    if name == "CurrentLLMSegmenter":
        from .current_llm_segmenter import CurrentLLMSegmenter
        return CurrentLLMSegmenter
    if name in _STAGE3_LAZY_EXPORTS:
        from . import segmentation_validation

        return getattr(segmentation_validation, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
