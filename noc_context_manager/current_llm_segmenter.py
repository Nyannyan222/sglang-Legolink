"""Current-LLM v3 wire codec behind the canonical Stage 3 entrypoint.

No Jev functional classifier, namespace, identity, selection or KV changes.
"""
from __future__ import annotations
import copy
import json
import time
from typing import Any
from .atomic_units import UnitGroupingError, LEGACY_SPLIT_POINT_VALIDATION_VERSION
from .semantic_segmentation import OpenAICompatibleSemanticSegmenter
from .current_llm_guardrails import (
    cuts_schema, cuts_max_output_tokens, decode_cuts_response, cuts_prompt, GUARDRAIL_VERSION,
)

class CurrentLLMSegmenter(OpenAICompatibleSemanticSegmenter):
    """Existing boundary algorithm with fixed-position Boolean output encoding."""

    backend_name = "current_llm"
    formal_output_encoding = "cuts_v3"
    guardrail_version = GUARDRAIL_VERSION
    _grouping_feedback_name = "cuts"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault('split_point_validation_version', LEGACY_SPLIT_POINT_VALIDATION_VERSION)
        super().__init__(**kwargs)
        self.segmentation_system_prompt = cuts_prompt(self.segmentation_system_prompt)
        self.request_measurements: list[dict[str, Any]] = []

    def _grouping_schema(self, units):
        return cuts_schema(units)

    def _grouping_request_options(self, units):
        return {'max_tokens': cuts_max_output_tokens(units)}

    def _trivial_grouping_response(self):
        return {'cuts': []}

    def _parse_grouping_response(self, response):
        # Formal output is strict JSON. Never strip fences, fix JSON, coerce, or
        # hide duplicate object keys through json.loads' last-value-wins policy.
        def unique_object(pairs):
            value = {}
            for key, item in pairs:
                if key in value:
                    raise UnitGroupingError('cuts_schema', 'Duplicate JSON object key.')
                value[key] = item
            return value
        def invalid_constant(value):
            raise UnitGroupingError('cuts_type', f'Invalid JSON constant {value}.')
        return json.loads(response, object_pairs_hook=unique_object, parse_constant=invalid_constant)

    def _load_cached_grouping_response(self, response):
        return self._parse_grouping_response(response)

    def _materialize_response(self, response, context_field, sources, diagnostic):
        source, = sources
        units = self._source_units(source)
        diagnostic['raw_response'] = copy.deepcopy(response)
        diagnostic['atomic_units'] = [u.to_dict() for u in units]
        diagnostic['output_encoding'] = self.formal_output_encoding
        try:
            host_response = decode_cuts_response(response, units)
        except UnitGroupingError as exc:
            diagnostic.update(unit_error_code=exc.code, error=str(exc))
            raise
        diagnostic['host_boundary_response'] = copy.deepcopy(host_response)
        try:
            return super()._materialize_response(host_response, context_field, sources, diagnostic)
        finally:
            diagnostic['raw_response'] = copy.deepcopy(response)

    def _request(self, payload: dict[str, Any]) -> Any:
        stage = ("classification" if "classification" in
                 payload.get("response_format", {}).get("json_schema", {}).get("name", "")
                 else "segmentation")
        row = {"stage": stage, "latency_s": None, "response_model": None, "status": "failed"}
        if stage == 'segmentation':
            row['guardrail_version'] = GUARDRAIL_VERSION
            row['max_output_tokens'] = payload.get('max_tokens')
        start = time.perf_counter()
        try:
            response = super()._request(payload)
            if isinstance(response, dict):
                row["response_model"] = response.get("model")
                row['completion_tokens'] = response.get('usage',{}).get('completion_tokens')
                if response.get('choices'):
                    row['finish_reason'] = response['choices'][0].get('finish_reason')
                    if stage == 'segmentation' and row['finish_reason'] == 'length':
                        row['truncated_http_response'] = copy.deepcopy(response)
                        raise UnitGroupingError('output_token_bound', 'Provider reached the output/context token limit; not a clean segmentation.')
            row["status"] = "returned"
            return response
        finally:
            row["latency_s"] = time.perf_counter() - start
            self.request_measurements.append(row)
