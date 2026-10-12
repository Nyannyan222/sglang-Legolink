"""Encoding-only reference, strict failures and finite Boolean output; CPU only."""
import itertools
import json

import pytest

from noc_context_manager.atomic_units import AtomicUnit, AtomicUnitBuilder, UnitGroupingError
from noc_context_manager.current_llm_guardrails import (
    cuts_schema, cuts_output_byte_bound, cuts_max_output_tokens, decode_cuts_response,
)
from noc_context_manager.semantic_segmentation import CurrentLLMSegmenter
from noc_context_manager.semantic_segmentation import OpenAICompatibleSemanticSegmenter
from noc_context_manager.segmentation_validation import SemanticSegmentationPipeline, SemanticSegmentationValidator
from noc_context_manager.segmentation_prompts import SEGMENTATION_PROMPTS
from noc_context_manager.session import ExtractedSource, FieldExtraction
from noc_context_manager.tokenizer import RegexTokenizer

pytestmark = [pytest.mark.cpu, pytest.mark.unit]


def units(n):
    return [AtomicUnit(f'U{i:04d}', 'source', 'workspace_evidence', i-1, i, 'x', 1)
            for i in range(1, n+1)]


def make_segmenter(transport, cls=CurrentLLMSegmenter, **options):
    return cls(api_base='http://fixture.invalid/v1', model='fixture', transport=transport,
               semantic_classification_enabled=False, retries=0, cache_dir=None, **options)


def fixture(n=4):
    src = ExtractedSource('workspace_evidence', 'event', 'source',
                          '\n\n'.join(f'topic{i} '*90 for i in range(n)))
    return src, FieldExtraction('task', 1, {src.context_field: (src,)})


@pytest.mark.parametrize('n', [0, 1, 2, 5, 12, 14, 10001])
def test_exact_vector_schema_and_finite_bound(n):
    values = units(n); count = max(0, n-1); schema = cuts_schema(values)
    assert schema['required'] == ['cuts'] and schema['additionalProperties'] is False
    assert schema['properties'] == {'cuts': {'type': 'array', 'items': {'type': 'boolean'},
                                            'minItems': count, 'maxItems': count}}
    key = '"'+''.join('\\u%04x'%ord(c) for c in 'cuts')+'"'
    maximum = '{'+key+': ['+', '.join(['false']*count)+']}'
    assert json.loads(maximum) == {'cuts': [False]*count}
    assert len(maximum.encode()) == cuts_output_byte_bound(values)
    assert len(maximum.encode())+1 == cuts_max_output_tokens(values)
    assert cuts_max_output_tokens(values) == (33 if not count else 31+7*count)


@pytest.mark.parametrize('n', [0, 1, 2, 5])
def test_every_small_legal_vector_maps_to_existing_ordered_candidate_ids(n):
    values = units(n)
    for flags in itertools.product((False, True), repeat=max(0, n-1)):
        decoded = decode_cuts_response({'cuts': list(flags)}, values)
        assert [r['after_unit_id'] for r in decoded['split_points']] == [
            u.unit_id for u, selected in zip(values[:-1], flags) if selected]


@pytest.mark.parametrize('value,code', [
    ({'cuts': []}, 'cuts_length'), ({'cuts': [True]*4}, 'cuts_length'),
    ({'cuts': [0, True, False]}, 'cuts_type'), ({'cuts': [1, True, False]}, 'cuts_type'),
    ({'cuts': ['true', True, False]}, 'cuts_type'), ({'cuts': [None, True, False]}, 'cuts_type'),
    ({'cuts': [0.0, True, False]}, 'cuts_type'), ({'cuts': [[], True, False]}, 'cuts_type'),
    ({'cuts': [False]*3, 'reason': 'extra'}, 'cuts_schema'),
    ({'split_points': []}, 'cuts_schema'), ({'cuts': 'false'}, 'cuts_schema'),
    (None, 'cuts_schema')])
def test_malformed_formal_response_rejected_without_coercion(value, code):
    with pytest.raises(UnitGroupingError) as error:
        decode_cuts_response(value, units(4))
    assert error.value.code == code


@pytest.mark.parametrize('text', ['', 'one unit'])
def test_empty_single_field_has_no_model_call_and_empty_formal_vector(text):
    def forbidden(_):
        pytest.fail('host-trivial field must not call provider')
    s = make_segmenter(forbidden)
    source = ExtractedSource('workspace_evidence', 'event', 'source', text)
    proposals = s.segment_field(source.context_field, [source])
    assert len(proposals) == bool(text) and not s.request_measurements
    assert decode_cuts_response({'cuts': []}, AtomicUnitBuilder(RegexTokenizer()).build(source)) == {'split_points': []}
    if text:
        assert s.response_diagnostics[0]['raw_response'] == {'cuts': []}


@pytest.mark.parametrize('variant', list(SEGMENTATION_PROMPTS))
def test_semantic_instruction_rules_unchanged_for_every_prompt_variant(variant):
    original = SEGMENTATION_PROMPTS[variant]
    s = make_segmenter(lambda _: None, prompt_variant=variant)
    assert original.split('\n1.', 1)[1].split('\n5.', 1)[0] == s.segmentation_system_prompt.split('\n1.', 1)[1].split('\n5.', 1)[0]
    assert 'return {"split_points"' not in s.segmentation_system_prompt
    assert 'do not generate IDs or reasons' in s.segmentation_system_prompt


@pytest.mark.parametrize('n,flags', [(2, (False,)), (2, (True,))] +
                         [(4, flags) for flags in itertools.product((False, True), repeat=3)])
def test_exact_reconstruction_and_all_identities_match_legacy_partition(n, flags):
    source, extraction = fixture(n); calls = []
    def current(payload):
        calls.append(payload)
        return {'cuts': list(flags)}
    old_wire = {'split_points': [{'after_unit_id': f'U{i:04d}', 'reason': 'historical rationale'}
                               for i, cut in enumerate(flags, 1) if cut]}
    def run(segmenter):
        return SemanticSegmentationPipeline(segmenter, validator=SemanticSegmentationValidator(
            tokenizer=RegexTokenizer())).run(extraction, model_fingerprint='model:fixture')
    new = run(make_segmenter(current)); old = run(make_segmenter(lambda _: old_wire, OpenAICompatibleSemanticSegmenter))
    again = run(make_segmenter(current))
    assert not new.report.fallback_used and not old.report.fallback_used
    values = AtomicUnitBuilder(RegexTokenizer()).build(source)
    endpoints = [0]+[values[i].end_char for i, cut in enumerate(flags) if cut]+[len(source.raw_text)]
    assert [(b.source_spans[0].start, b.source_spans[0].end) for b in new.sub_contexts] == list(zip(endpoints, endpoints[1:]))
    assert ''.join(b.raw_text for b in new.sub_contexts) == source.raw_text
    for key in ('source_spans', 'raw_text', 'canonical_text', 'sub_context_id', 'semantic_identity', 'cache_key'):
        assert [getattr(b, key) for b in new.sub_contexts] == [getattr(b, key) for b in old.sub_contexts] == [getattr(b, key) for b in again.sub_contexts]
    assert calls[0]['response_format']['json_schema']['schema'] == cuts_schema(values)
    assert calls[0]['max_tokens'] == cuts_max_output_tokens(values)

