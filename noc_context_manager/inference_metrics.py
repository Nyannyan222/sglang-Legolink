"""Metric definitions shared by LegoLink runtime and agent benchmarks."""

from __future__ import annotations

import hashlib
from difflib import SequenceMatcher
from typing import Any, Optional


def output_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def output_agreement(reference: str, candidate: str) -> dict[str, Any]:
    return {
        "exact": reference == candidate,
        "reference_sha256": output_sha256(reference),
        "candidate_sha256": output_sha256(candidate),
        "hash_match": output_sha256(reference) == output_sha256(candidate),
        "similarity": SequenceMatcher(None, reference, candidate).ratio(),
    }


def compute_stream_metrics(
    *,
    request_started_ns: int,
    token_ready_ns: list[int],
    request_completed_ns: int,
    prompt_tokens: Optional[int],
    output_tokens: Optional[int],
    reused_tokens: Optional[int] = None,
    recomputed_tokens: Optional[int] = None,
    prefill_ns: Optional[int] = None,
    semantic_lookup_ns: Optional[int] = None,
    link_plan_build_ns: Optional[int] = None,
    link_ns: Optional[int] = None,
    first_token_sampling_ns: Optional[int] = None,
) -> dict[str, Any]:
    if request_completed_ns < request_started_ns:
        raise ValueError("request completion precedes request start")
    if any(value < request_started_ns for value in token_ready_ns):
        raise ValueError("token timestamp precedes request start")
    if token_ready_ns != sorted(token_ready_ns):
        raise ValueError("token timestamps must be monotonic")

    first_ns = token_ready_ns[0] if token_ready_ns else None
    ttft_ns = first_ns - request_started_ns if first_ns is not None else None
    decode_ns = request_completed_ns - first_ns if first_ns is not None else None
    gaps = [
        current - previous
        for previous, current in zip(token_ready_ns, token_ready_ns[1:])
    ]
    itl_ns = sum(gaps) / len(gaps) if gaps else None
    ordered_gaps = sorted(gaps)

    def percentile(values: list[int], quantile: float) -> Optional[float]:
        if not values:
            return None
        if len(values) == 1:
            return float(values[0])
        position = (len(values) - 1) * quantile
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        fraction = position - lower
        return values[lower] + (values[upper] - values[lower]) * fraction
    linked = None
    if reused_tokens is not None and recomputed_tokens is not None:
        linked = int(reused_tokens) + int(recomputed_tokens)
    cache_ratio = (
        float(reused_tokens) / linked
        if linked and reused_tokens is not None
        else None
    )
    request_duration_s = (request_completed_ns - request_started_ns) / 1_000_000_000
    output_tps = (
        float(output_tokens) / request_duration_s
        if output_tokens is not None and request_duration_s > 0
        else None
    )

    def ms(value: Optional[float]) -> Optional[float]:
        return None if value is None else float(value) / 1_000_000.0

    return {
        "request_latency_ns": request_completed_ns - request_started_ns,
        "request_latency_ms": ms(request_completed_ns - request_started_ns),
        "ttft_ns": ttft_ns,
        "ttft_ms": ms(ttft_ns),
        "prefill_ns": prefill_ns,
        "prefill_ms": ms(prefill_ns),
        "semantic_lookup_ns": semantic_lookup_ns,
        "semantic_lookup_ms": ms(semantic_lookup_ns),
        "link_plan_build_ns": link_plan_build_ns,
        "link_plan_build_ms": ms(link_plan_build_ns),
        "link_ns": link_ns,
        "link_ms": ms(link_ns),
        "first_token_sampling_ns": first_token_sampling_ns,
        "first_token_sampling_ms": ms(first_token_sampling_ns),
        "decode_ns": decode_ns,
        "decode_ms": ms(decode_ns),
        "itl_ns": itl_ns,
        "itl_ms": ms(itl_ns),
        "itl_p50_ms": ms(percentile(ordered_gaps, 0.50)),
        "itl_p95_ms": ms(percentile(ordered_gaps, 0.95)),
        "output_tokens_per_second": output_tps,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "linked_tokens": linked,
        "reused_tokens": reused_tokens,
        "recomputed_tokens": recomputed_tokens,
        "cache_reuse_ratio": cache_ratio,
    }
