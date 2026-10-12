"""Exact-byte segmentation artifacts for controlled serving replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from noc_context_manager.semantic_segmentation import SegmentProposal, SemanticSegmenter
from noc_context_manager.session import ExtractedSource, SourceSpan


def source_sha256(raw_text: str) -> str:
    return hashlib.sha256(raw_text.encode("utf-8")).hexdigest()


class ExactArtifactSemanticSegmenter(SemanticSegmenter):
    """Replay frozen spans only when every source byte matches the artifact."""

    model_name = "exact-frozen-segmentation-artifact"

    def __init__(self, artifact: dict[str, Any] | str | Path) -> None:
        if isinstance(artifact, (str, Path)):
            artifact = json.loads(Path(artifact).read_text(encoding="utf-8"))
        self.artifact = dict(artifact)
        self.artifact_sha256 = hashlib.sha256(
            json.dumps(
                self.artifact, sort_keys=True, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        rows = self.artifact.get("sources")
        if not isinstance(rows, list):
            raise ValueError("segmentation artifact requires a sources array")
        self._sources = {}
        for row in rows:
            key = (str(row["context_field"]), str(row["source_id"]))
            if key in self._sources:
                raise ValueError(f"duplicate segmentation artifact source: {key}")
            self._sources[key] = row

    def segment_field(
        self, context_field: str, sources: list[ExtractedSource]
    ) -> list[SegmentProposal]:
        proposals: list[SegmentProposal] = []
        for source in sources:
            key = (context_field, source.source_id)
            row = self._sources.get(key)
            if row is None:
                raise ValueError(f"segmentation artifact miss: {key}")
            actual = source_sha256(source.raw_text)
            if actual != row.get("source_sha256"):
                raise ValueError(
                    f"segmentation artifact hash mismatch for {key}: "
                    f"expected {row.get('source_sha256')}, got {actual}"
                )
            for block in row.get("blocks", []):
                metadata = dict(block.get("metadata") or {})
                spans = [SourceSpan.from_dict(value) for value in block["source_spans"]]
                if any(span.source_id != source.source_id for span in spans):
                    raise ValueError(f"segmentation artifact crossed source boundary: {key}")
                proposals.append(
                    SegmentProposal(
                        title=str(block["title"]),
                        context_field=str(block["context_field"]),
                        source_spans=spans,
                        question_or_objective=str(block["question_or_objective"]),
                        semantic_independence_reason=str(
                            block["semantic_independence_reason"]
                        ),
                        is_mutable=bool(block["is_mutable"]),
                        reuse_candidate=bool(block["reuse_candidate"]),
                        confidence=float(block["segmentation_confidence"]),
                        llm_semantic_role=metadata.get("llm_semantic_role"),
                        llm_semantic_topic=metadata.get("llm_semantic_topic"),
                        segmentation_decision_source=metadata.get("segmentation_decision_source"),
                        classification_decision_source=metadata.get("classification_decision_source"),
                        unit_start_id=metadata.get("classification_unit_start_id"),
                        unit_end_id=metadata.get("classification_unit_end_id"),
                        classification_model=metadata.get("classification_model"),
                        classification_prompt_version=metadata.get("classification_prompt_version"),
                        classification_repair_count=int(metadata.get("classification_repair_count") or 0),
                        classification_fallback=bool(metadata.get("classification_fallback", False)),
                        classification_status=metadata.get("classification_status"),
                    )
                )
        return proposals
