"""Opt-in Agent experiment admission; general segmentation fallback is unchanged."""
import json
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from noc_context_manager.current_llm_guardrails import GUARDRAIL_VERSION


def endpoint_identity(base):
    value = urlsplit(base)
    host = value.hostname or ""
    if value.port:
        host += f":{value.port}"
    return urlunsplit((value.scheme, host, value.path.rstrip("/"), "", ""))


def classification_disabled(row):
    """Was Stage 2 deliberately off for this run?

    Read from the segmenter's own recorded configuration, not inferred from an absent label: a run that
    never classified and a run whose classification failed must not look alike.
    """

    return row.get("semantic_classification_enabled") is False


def require_segmentation_evidence(row):
    """Stage 1 only: LLM split points were produced (or replayed) as the experiment intended.

    Stage 2 is not consulted here. It labels segments Stage 1 has already fixed, it reaches no cache key,
    selector, planner or prompt, and a formal segmentation-granularity experiment may run with it off.
    """

    decision_source = row.get("segmentation_decision_source")
    llm_evidence = (
        row.get("segmentation_llm_calls", 0) + row.get("segmentation_cache_hits", 0) > 0
    )
    decision_evidence_valid = (
        (decision_source == "llm" and llm_evidence)
        or (decision_source == "mixed_llm_and_host_trivial_single_unit" and llm_evidence)
        or (
            # A source with one atomic unit has no internal boundary to choose, so Stage 1 is host-trivial
            # and makes no call. With Stage 2 on, that call was the remaining LLM evidence; with Stage 2
            # off there is none to have, and the deterministic no-split decision is the correct one.
            decision_source == "host_trivial_single_unit"
            and row.get("segmentation_llm_calls", 0) == 0
            and (
                classification_disabled(row)
                or row.get("classification_decision_source") == "llm"
            )
        )
    )
    actual = row.get("actual_segmenter")
    segmenter_valid = actual == "OpenAICompatibleSemanticSegmenter"
    if actual == "CurrentLLMSegmenter":
        # v3 is the canonical Stage 3 Current-LLM implementation. Preserve
        # legacy evidence admission while requiring its explicit wire contract
        # and clean boundary decisions for new v3 experiment evidence.
        segmenter_valid = (
            row.get("segmentation_output_encoding") == "cuts_v3"
            and row.get("segmentation_guardrail_version") == GUARDRAIL_VERSION
            and not row.get("validation_repair_count")
            and not row.get("validation_failures")
            and not any(item.get("status") != "valid"
                        for item in row.get("unit_grouping_validation_attempts", []))
            and not any(item.get("error") or item.get("unit_error_code")
                        for item in row.get("llm_responses", []) if item.get("stage") == "segmentation")
            and not any(item.get("status") == "failed" or item.get("finish_reason") == "length"
                        for item in row.get("segmentation_request_measurements", []))
        )
    if (row.get("segmentation_mode") != "llm" or row.get("fallback_used") is not False
            or not row.get("segmentation_model") or not row.get("segmentation_api_base")
            or not segmenter_valid
            or not decision_evidence_valid
            or row.get("integrity_errors") or not row.get("accepted_subcontext_count")):
        raise ValueError(f"LLM segmentation experiment rejected: {row}")


def require_classification_evidence(row):
    """Stage 2, only where the protocol asked for it.

    Disabled is a valid state and is required to say so explicitly: `classification_status == "disabled"`.
    An absent or failed status is still a rejection, so a silently skipped classification cannot pass as a
    deliberately disabled one.
    """

    if classification_disabled(row):
        if row.get("classification_status") != "disabled":
            raise ValueError(
                f"classification disabled but not recorded as disabled: {row.get('classification_status')!r}"
            )
        return
    if row.get("classification_status") != "success":
        raise ValueError(f"LLM classification experiment rejected: {row}")


def require_llm_evidence(row):
    """Both stages, each judged on its own evidence."""

    require_segmentation_evidence(row)
    require_classification_evidence(row)


def install_experiment_gate(pipeline, mode, report_dir):
    """Observe the existing pipeline, persist evidence, then reject invalid experiments."""
    original_run = pipeline.run
    segmenter = pipeline.segmenter

    def run(extraction, *, changed_source_ids=None, **kwargs):
        path = Path(report_dir) / f"llm_evidence_r{extraction.session_revision}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"segmentation_mode": mode, "fallback_used": None,
                                   "error": "segmentation did not complete"}), encoding="utf-8")
        names = ("llm_calls", "segmentation_llm_calls", "classification_llm_calls",
                 "segmentation_cache_hits", "classification_cache_hits", "internal_fallbacks")
        before = [getattr(segmenter, name, 0) for name in names]
        diagnostic_start = len(getattr(segmenter, "response_diagnostics", []))
        measurement_start = len(getattr(segmenter, "request_measurements", []))
        attempts = []
        validation_attempts = []
        validate = pipeline.validator.validate

        def observed(*args, **kw):
            result = validate(*args, **kw)
            attempts.append(result.report)
            validation_attempts.append({"proposals": [p.to_dict() for p in args[0]],
                "validation": result.report.to_dict(),
                "accepted_blocks": [b.to_dict() for b in result.sub_contexts]})
            return result

        pipeline.validator.validate = observed
        try:
            result = original_run(extraction, changed_source_ids=changed_source_ids, **kwargs)
        finally:
            pipeline.validator.validate = validate
        after = [getattr(segmenter, name, 0) for name in names]
        calls, segmentation_calls, classification_calls, segmentation_hits, classification_hits, internal = [a - b for a, b in zip(after, before)]
        sources = extraction.source_index()
        changed = changed_source_ids or set(sources)
        spans = {key: [] for key in changed if sources[key].raw_text}
        errors = []
        coverage = {}
        for block in result.sub_contexts:
            if block.raw_text != "\n\n".join(sources[s.source_id].raw_text[s.start:s.end]
                                             for s in block.source_spans):
                errors.append("source reconstruction mismatch")
            for span in block.source_spans:
                spans.setdefault(span.source_id, []).append((span.start, span.end))
        for key, ranges in spans.items():
            cursor = 0
            for start, end in sorted(ranges):
                if start != cursor:
                    errors.append(f"omission/overlap in {key} at {cursor}")
                cursor = end
            if cursor != len(sources[key].raw_text):
                errors.append(f"incomplete source {key}")
            points = sorted({0, len(sources[key].raw_text), *(p for r in ranges for p in r)})
            covered = overlap = 0
            for start, end in zip(points, points[1:]):
                count = sum(a <= start and end <= b for a, b in ranges)
                covered += (end - start) * bool(count)
                overlap += (end - start) * (count > 1)
            length = len(sources[key].raw_text)
            coverage[key] = {"source_chars": length, "covered_chars": covered,
                "uncovered_chars": length - covered,
                "duplicated_chars": sum(b - a for a, b in ranges) - covered,
                "overlap_chars": overlap, "coverage_ratio": covered / length if length else 1.0}
        oversized = set(result.preliminary_token_count.oversized_source_ids) & changed
        fallback = result.report.fallback_used or bool(internal) or bool(oversized)
        response_diagnostics = getattr(segmenter, "response_diagnostics", [])[diagnostic_start:]
        decision_sources = {
            item.get("segmentation_decision_source") for item in response_diagnostics
            if item.get("segmentation_decision_source")
        }
        if decision_sources == {"host_trivial_single_unit"}:
            decision_source = "host_trivial_single_unit"
        elif decision_sources == {"llm"}:
            decision_source = "llm"
        else:
            decision_source = "mixed_llm_and_host_trivial_single_unit"
        classification_sources = {
            item.get("classification_decision_source") for item in response_diagnostics
            if item.get("classification_decision_source")
        }
        if classification_sources == {"llm"}:
            classification_source = "llm"
        elif classification_sources == {"disabled"}:
            classification_source = "disabled"
        else:
            classification_source = "mixed_or_missing"
        classification_statuses = {
            item.get("classification_status") for item in response_diagnostics
            if item.get("stage") == "classification"
        }
        if classification_statuses == {"success"}:
            classification_status = "success"
        elif classification_statuses == {"disabled"}:
            classification_status = "disabled"
        else:
            classification_status = "failed"
        row = {
            "segmentation_mode": mode,
            # The segmenter's own configuration, so a reader never has to infer why labels are absent.
            "semantic_classification_enabled": getattr(
                segmenter, "semantic_classification_enabled", None
            ),
            "actual_segmenter": "DeterministicFieldSegmenter_or_mixed" if fallback else type(segmenter).__name__,
            "segmentation_output_encoding": getattr(segmenter, "formal_output_encoding", None),
            "segmentation_guardrail_version": getattr(segmenter, "guardrail_version", None),
            "segmentation_request_measurements": [
                item for item in getattr(segmenter, "request_measurements", [])[measurement_start:]
                if item.get("stage") == "segmentation"
            ],
            "segmentation_model": segmenter.model_name,
            "segmentation_api_base": endpoint_identity(getattr(segmenter, "endpoint", "").removesuffix("/chat/completions")),
            "fallback_used": fallback, "llm_calls": calls, "llm_call_count": calls,
            "segmentation_llm_calls": segmentation_calls,
            "classification_llm_calls": classification_calls,
            "segmentation_cache_hits": segmentation_hits,
            "classification_cache_hits": classification_hits,
            "segmentation_decision_source": decision_source,
            "classification_decision_source": classification_source,
            "classification_status": classification_status,
            "total_segment_proposals": sum(r.proposal_count for r in attempts),
            "accepted_subcontext_count": len(result.sub_contexts),
            "rejected_proposal_count": sum(r.proposal_count - r.valid_count for r in attempts),
            "validation_repair_count": result.report.repair_count,
            # Back-to-back repeats of one boundary ID collapsed before validation decided anything.
            # Diagnostic only: it neither creates nor hides a fallback, and the decision source stays
            # whatever produced the accepted split points.
            "duplicate_split_ids_normalized": sum(
                int(item.get("duplicate_split_ids_normalized") or 0) for item in response_diagnostics
                if item.get("stage") == "segmentation"),
            "split_point_validation_versions": sorted({
                item["split_point_validation_version"] for item in response_diagnostics
                if item.get("split_point_validation_version")}),
            "validation_failures": [i.code for r in attempts for i in r.issues],
            "unit_grouping_validation_attempts": getattr(pipeline, "format_validation_attempts", []),
            "integrity_errors": errors,
            "coverage": coverage,
            "context_field_source": "field_first_extractor",
            "llm_responses": response_diagnostics,
            "validation_attempts": validation_attempts,
            "repair_retry_occurred": bool(result.report.repair_count),
            "repaired_proposal_count": attempts[1].proposal_count if result.report.repair_count and len(attempts) > 1 else 0,
            "final_accepted_blocks": [b.to_dict() for b in result.sub_contexts],
        }
        path.write_text(json.dumps(row, indent=2), encoding="utf-8")
        pipeline.segmentation_evidence = row
        pipeline.segmentation_result = result
        require_llm_evidence(row)
        return result

    pipeline.run = run


def validate_artifacts(root):
    rows = [json.loads(p.read_text(encoding="utf-8"))
            for p in Path(root).rglob("llm_evidence_r*.json")]
    if not rows:
        raise ValueError("LLM segmentation experiment rejected: no evidence")
    for row in rows:
        require_llm_evidence(row)
    result = dict(rows[0])
    for key in ("llm_calls", "segmentation_llm_calls", "classification_llm_calls",
                "segmentation_cache_hits", "classification_cache_hits", "total_segment_proposals",
                "accepted_subcontext_count", "rejected_proposal_count", "validation_repair_count"):
        result[key] = sum(row[key] for row in rows)
    # Absent on evidence written before split-point validation v2.
    result["duplicate_split_ids_normalized"] = sum(
        int(row.get("duplicate_split_ids_normalized") or 0) for row in rows)
    result["validation_failures"] = [v for row in rows for v in row["validation_failures"]]
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    summary = validate_artifacts(args.root)
    (args.root / "segmentation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
