"""Finite Current-LLM boundary output; explanations never determine geometry.

Token bound requires a byte-level tokenizer with nonempty output tokens and the
SGLang xgrammar fixed-whitespace grammar. Formal evaluations verify both, rather
than claiming arbitrary JSON whitespace or arbitrary tokenizers have this bound.
"""
from __future__ import annotations

from typing import Sequence

from .atomic_units import AtomicUnit, UnitGroupingError

GUARDRAIL_VERSION = "current-llm-fixed-cuts-v3"
TOKEN_TERMINATION_MARGIN = 1


def cuts_schema(units: Sequence[AtomicUnit]) -> dict:
    count = max(0, len(units) - 1)
    return {'type': 'object', 'additionalProperties': False, 'required': ['cuts'],
            'properties': {'cuts': {'type': 'array', 'items': {'type': 'boolean'},
                                    'minItems': count, 'maxItems': count}}}


def cuts_output_byte_bound(units: Sequence[AtomicUnit]) -> int:
    """Fixed-whitespace JSON: 32-byte escaped-key envelope, false=5 bytes.

    Even a fully escaped \"cuts\" key fits {key: []} in 32 bytes. For B>0 add
    5B bytes for booleans and 2(B-1) for ', '. The serving grammar actually
    emits the literal key (12-byte envelope); the extra 20 bytes are conservative.
    Arbitrary whitespace is excluded by the verified fixed-whitespace grammar.
    """
    count = max(0, len(units) - 1)
    return 32 + 5 * count + 2 * max(0, count - 1)


def cuts_max_output_tokens(units: Sequence[AtomicUnit]) -> int:
    # Verified Qwen byte-BPE: each non-stop token emits >=1 byte, plus one EOS.
    return cuts_output_byte_bound(units) + TOKEN_TERMINATION_MARGIN


def decode_cuts_response(response: object, units: Sequence[AtomicUnit]) -> dict:
    """Strict wire decoding, then host-assigned ordered IDs for the shared validator.

    No normalization, coercion, sorting, deduplication, or geometry repair.
    The fixed diagnostic is excluded from canonical text and every identity.
    """
    if not isinstance(response, dict) or set(response) != {'cuts'} or type(response['cuts']) is not list:
        raise UnitGroupingError('cuts_schema', 'Return only an object containing a cuts array.')
    count = max(0, len(units) - 1)
    cuts = response['cuts']
    if len(cuts) != count:
        raise UnitGroupingError('cuts_length', f'Expected exactly {count} Boolean cuts; received {len(cuts)}.')
    if any(type(value) is not bool for value in cuts):
        raise UnitGroupingError('cuts_type', 'Every cuts entry must be a JSON Boolean, without coercion.')
    return {'split_points': [{'after_unit_id': units[i].unit_id,
                             'reason': 'Host-decoded semantic boundary.'}
                            for i, selected in enumerate(cuts) if selected]}


def cuts_prompt(prompt: str) -> str:
    """Replace only the encoding instructions; preserve rules 1-4 verbatim."""
    from .segmentation_prompts import OUTPUT_CONTRACT
    old = ('should end. Return those boundaries as split_points with after_unit_id and a concise\n'
           'diagnostic reason.')
    new = ('should end. Return those decisions as a fixed-position Boolean vector cuts.\n'
           'The host supplies the boundary IDs; do not generate IDs or reasons.')
    contract = '''5. Preserve original order using one Boolean for each legal internal boundary.
6. For N atomic units, return exactly N-1 Booleans; for N <= 1 return [].
7. cuts[i] (zero-based) decides the boundary after the (i+1)-th atomic unit:
true selects that boundary, false does not. The host closes the final segment.
8. Do not return boundary IDs or segment start/end ranges.
9. If the whole source has one coherent purpose, return all false.
10. Return only {"cuts": [bool, ...]}. Do not output reasons, source text,
character offsets, source IDs, context fields, titles, objectives, mutability,
or reuse flags. Those belong exclusively to the host.

Example: for U0001...U0006, if purposes are U0001-U0002, U0003-U0005 and U0006,
return {"cuts": [false, true, false, false, true]}. The host constructs
all three exhaustive ranges automatically.
'''
    if old not in prompt or not prompt.endswith(OUTPUT_CONTRACT):
        raise ValueError('Unknown segmentation prompt encoding; cannot preserve semantic instructions.')
    return prompt.replace(old, new, 1)[:-len(OUTPUT_CONTRACT)] + contract

