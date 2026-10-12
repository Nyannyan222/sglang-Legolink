"""Semantic execution admission. No physical positions or latency claims.

The pure validator is mirrored into the standalone SGLang runtime; a regression
test pins its source so the Windows host need not import SGLang.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class FramingSpec:
    framing_layout_version: str
    model_id: str
    tokenizer_fingerprint: str
    chat_template_sha256: str
    profile_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RepairDecision:
    selected_k: int
    windows: tuple[int, ...]
    policy: str
    policy_version: str
    framing_layout_version: str
    quality_profile_id: str | None
    evidence_id: str | None
    reason: str
    workload_class: str

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["windows"] = list(self.windows)
        return result


@dataclass(frozen=True)
class RepairQualityProfile:
    """Measured correctness for an exact window vector, independent of latency.

    No interpolation across K, framing or workload. Hardware timing belongs in
    CompileCostProfile. Evidence must be reviewed before installing a profile.
    """
    profile_id: str
    evidence_id: str
    framing_spec: Mapping[str, Any]
    workload_class: str
    selected_k: int
    windows: tuple[int, ...]
    metric: str
    measured_value: float
    required_threshold: float
    status: str

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["windows"] = list(self.windows)
        return result


def validate_execution_intent(wire, profile=None, *, require_formal=False):
    """Validate policy, then consistency. `profile` is deployment-owned trust.

    canonical_input_hashes are reviewed canonical input evidence, never learned
    from the incoming request. An identity label plus a matching hash is not a
    framing proof. Admission requires that input in the installed profile too.
    """
    import math

    mode = wire.get("execution_mode", "legacy_debug")
    if mode not in ("formal", "legacy_debug"):
        raise ValueError("unknown execution_mode")
    if require_formal and mode != "formal":
        raise ValueError("formal deployment rejects legacy_debug execution")
    formal = require_formal or mode == "formal"
    framing = wire.get("framing_spec")
    repair = wire.get("repair_decision")
    if not formal and framing is None and repair is None:
        return
    required_framing = {"framing_layout_version", "model_id", "tokenizer_fingerprint",
                        "chat_template_sha256", "profile_id"}
    if not isinstance(framing, dict) or set(framing) != required_framing:
        raise ValueError("missing or malformed framing_spec")
    if any(not isinstance(v, str) or not v.strip() for v in framing.values()):
        raise ValueError("framing identity must be nonempty")
    version = framing["framing_layout_version"]
    if version != "noc-single-chat-template-split-v1":
        raise ValueError("unsupported/legacy framing version")
    template = framing["chat_template_sha256"]
    if len(template) != 64 or any(c not in "0123456789abcdef" for c in template):
        raise ValueError("invalid chat template hash")
    if wire.get("physical_token_layout_version") != version:
        raise ValueError("framing and physical layout version disagree")
    fields = {"selected_k", "windows", "policy", "policy_version",
              "framing_layout_version", "quality_profile_id", "evidence_id",
              "reason", "workload_class"}
    if not isinstance(repair, dict) or set(repair) != fields:
        raise ValueError("missing or malformed RepairDecision")
    k, windows = repair["selected_k"], repair["windows"]
    if type(k) is not int or k < 0:
        raise ValueError("selected_k must be a nonnegative integer")
    if (not isinstance(windows, list) or len(windows) != len(wire.get("cache_keys", []))
            or not windows or any(type(v) is not int or v < 0 or v > k for v in windows)):
        raise ValueError("invalid repair window vector")
    if repair["policy_version"] != "repair-decision-v1":
        raise ValueError("unsupported repair policy version")
    if repair["framing_layout_version"] != version:
        raise ValueError("repair/framing incompatibility")
    if repair["policy"] not in ("quality_guarded", "fixed_unvalidated_debug"):
        raise ValueError("unknown repair policy")
    if repair["quality_profile_id"] is not None and not isinstance(
        repair["quality_profile_id"], str
    ):
        # Used as a dict key below; a list or dict would raise TypeError.
        raise ValueError("quality_profile_id must be a string or null")
    if not all(isinstance(repair[x], str) and repair[x].strip()
               for x in ("reason", "workload_class")):
        raise ValueError("missing repair provenance")
    if type(wire.get("recompute_initial_tokens")) is not int or wire["recompute_initial_tokens"] != k:
        raise ValueError("repair K was replaced or defaulted")
    if wire.get("recompute_tokens_per_block") != windows:
        raise ValueError("repair windows were replaced or defaulted")
    if (wire.get("recompute_first_block_tokens") != windows[0]
            or wire.get("recompute_first_block") is not bool(windows[0])):
        raise ValueError("first-block repair disagrees with decision")
    if "legolink_k" in wire and wire["legolink_k"] != k:
        raise ValueError("legacy K alias disagrees with decision")
    if formal or repair["policy"] == "quality_guarded":
        if repair["policy"] != "quality_guarded":
            raise ValueError("formal execution requires measured repair quality")
        if not isinstance(profile, dict) or profile.get("framing_spec") != framing:
            raise ValueError("missing/incompatible deployment framing profile")
        if wire.get("expected_complete_token_ids_sha256") not in profile.get("canonical_input_hashes", []):
            raise ValueError("input has no reviewed canonical framing evidence")
        quality = profile.get("repair_quality_profiles", {}).get(repair["quality_profile_id"])
        if not isinstance(quality, dict):
            raise ValueError("missing repair quality profile")
        for name in ("selected_k", "windows", "workload_class", "evidence_id"):
            if quality.get(name) != repair[name]:
                raise ValueError("incompatible repair quality " + name)
        if (quality.get("framing_spec") != framing or quality.get("status") != "PASS"
                or quality.get("profile_id") != repair["quality_profile_id"]
                or not repair["evidence_id"] or not quality.get("metric")):
            raise ValueError("unvalidated repair quality provenance")
        measured, threshold = quality.get("measured_value"), quality.get("required_threshold")
        if (type(measured) not in (float, int) or type(threshold) not in (float, int)
                or not math.isfinite(measured) or not math.isfinite(threshold)
                or measured < threshold):
            raise ValueError("repair quality threshold not satisfied")
