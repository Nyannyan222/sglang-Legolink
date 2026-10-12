"""A linked block is remapped by its position offset, never by its ordinal.

The runtime used to decide this with `if block_index:` -- skip the RoPE remap
for the first block, remap the rest. That is only correct while the first block
starts at global position 0. Under normalized framing it starts behind a chat
prefix, so its cached rows stayed at block-local angles and every reused row
sat at a uniform position offset. Full repair overwrote them and hid it;
partial repair left them, and retrieval broke.

These drive the real `_copy_linked_kv` / `_copy_linked_kv_batched_blocks` and
compare the pool against `tests/pic/reference/rope_reference.py`, which rotates
raw K to the required global position without going through any production
helper.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch

from .reference.rope_reference import rotate_neox
from .support import FakeRotary, fixed_generator, load_srt_module

pytestmark = [pytest.mark.unit, pytest.mark.cpu]

ROTARY_DIM = 8
HEADS = 2


class RecordingPool:
    """Minimal MHA-style pool: one layer, dense slots, nothing quantized."""

    def __init__(self, slots: int = 64):
        self.dtype = torch.float32
        self.store_dtype = torch.float32
        self.key = torch.zeros(slots, HEADS, ROTARY_DIM)
        self.value = torch.zeros(slots, HEADS, ROTARY_DIM)

    def get_key_buffer(self, layer_id):
        assert layer_id == 0
        return self.key

    def get_value_buffer(self, layer_id):
        assert layer_id == 0
        return self.value


def link_blocks(monkeypatch, tmp_path, blocks, *, batched=None):
    """Link `blocks` (list of (start, token_count)) and report what landed.

    Returns the raw K each block was built from, the pool, and the trace, so a
    test can compare against an independently rotated expectation.
    """

    module = load_srt_module("pic_runtime", "pic_runtime.py")
    rotary = FakeRotary(max_position=4096, rotary_dim=ROTARY_DIM)
    runtime = module.PICRuntime.__new__(module.PICRuntime)
    runtime.model_runner = SimpleNamespace(start_layer=0, end_layer=1)
    runtime._rotary_by_layer = {0: rotary}
    runtime._blocks = OrderedDict()
    runtime._legolink_cache = None
    runtime._active_trace_context = {"request_id": "remap-test"}

    generator = fixed_generator()
    raw_keys = {}
    ranges = []
    cache_keys = []
    for index, (start, count) in enumerate(blocks):
        cache_key = "block-%d" % index
        raw = torch.randn(count, HEADS, ROTARY_DIM, generator=generator)
        raw_keys[cache_key] = raw
        # The cached block is stored rotated to its own block-local positions,
        # which is what compilation produces.
        runtime._blocks[cache_key] = module.PICBlockSnapshot(
            cache_key=cache_key,
            token_ids=tuple(range(count)),
            keys={0: rotate_neox(raw, torch.arange(count))},
            values={0: torch.randn(count, HEADS, ROTARY_DIM, generator=generator)},
        )
        ranges.append(
            {"block_id": cache_key, "start_token": start, "end_token": start + count}
        )
        cache_keys.append(cache_key)

    pool = RecordingPool()
    total = max(block_range["end_token"] for block_range in ranges)
    destination_slots = torch.arange(total, dtype=torch.long)

    trace_path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("SGLANG_PIC_TRACE_PATH", str(trace_path))
    monkeypatch.setenv("SGLANG_PIC_LEGOLINK_ONLY", "0")
    monkeypatch.setenv("SGLANG_PIC_RADIXCACHE_OWNS_BLOCKS", "0")
    if batched is not None:
        monkeypatch.setenv("SGLANG_PIC_BATCH_BLOCK_KV_LINK", "1" if batched else "0")
    runtime._copy_linked_kv(
        pool,
        destination_slots,
        {"cache_keys": cache_keys, "block_ranges": ranges},
    )

    events = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    linked = [event for event in events if event["event"] == "kv_block_linked"]
    return raw_keys, pool, linked, ranges


def block_rows(pool, block_range):
    return pool.key[block_range["start_token"] : block_range["end_token"]]


def expected_global(raw, start):
    """What the pool must hold: raw K rotated to its request-global positions."""

    return rotate_neox(raw, torch.arange(raw.shape[0]) + start)


# --- Case A: first block already at global 0 -------------------------------


def test_first_block_at_offset_zero_is_not_remapped(monkeypatch, tmp_path):
    raw_keys, pool, linked, ranges = link_blocks(
        monkeypatch, tmp_path, [(0, 6)], batched=False
    )
    rows = block_rows(pool, ranges[0])
    torch.testing.assert_close(
        rows, expected_global(raw_keys["block-0"], 0), atol=2e-5, rtol=2e-5
    )
    assert linked[0]["rope_remapped"] is False


def test_offset_zero_first_block_is_copied_bit_for_bit(monkeypatch, tmp_path):
    """Case F. Local == global, so a remap would be a no-op that still costs a
    rounding step. The legacy layout must keep the cached bytes exactly."""

    raw_keys, pool, _, ranges = link_blocks(
        monkeypatch, tmp_path, [(0, 6)], batched=False
    )
    cached = rotate_neox(raw_keys["block-0"], torch.arange(6))
    assert torch.equal(block_rows(pool, ranges[0]), cached)


# --- Case B: first block at a nonzero offset (the regression) --------------


def test_first_block_at_nonzero_offset_is_remapped(monkeypatch, tmp_path):
    raw_keys, pool, linked, ranges = link_blocks(
        monkeypatch, tmp_path, [(24, 6)], batched=False
    )
    raw = raw_keys["block-0"]
    rows = block_rows(pool, ranges[0])
    torch.testing.assert_close(rows, expected_global(raw, 24), atol=2e-5, rtol=2e-5)
    assert linked[0]["rope_remapped"] is True


def test_first_block_at_nonzero_offset_does_not_keep_block_local_angles(
    monkeypatch, tmp_path
):
    """The exact failure: the pool held K_local where K_global was required."""

    raw_keys, pool, _, ranges = link_blocks(
        monkeypatch, tmp_path, [(24, 6)], batched=False
    )
    block_local = rotate_neox(raw_keys["block-0"], torch.arange(6))
    assert not torch.allclose(block_rows(pool, ranges[0]), block_local, atol=1e-3)


# --- Case C: later blocks, both layouts ------------------------------------


def test_later_block_is_remapped_when_the_first_starts_at_zero(monkeypatch, tmp_path):
    raw_keys, pool, linked, ranges = link_blocks(
        monkeypatch, tmp_path, [(0, 6), (6, 5)]
    )
    torch.testing.assert_close(
        block_rows(pool, ranges[1]),
        expected_global(raw_keys["block-1"], 6),
        atol=2e-5,
        rtol=2e-5,
    )
    assert [event["rope_remapped"] for event in linked] == [False, True]


def test_every_block_is_remapped_when_the_first_starts_at_an_offset(
    monkeypatch, tmp_path
):
    """Multi-block under normalized framing: fixing block 0 must not disturb
    block 1, and block 1's own offset is 24 + len(block 0), not len(block 0)."""

    raw_keys, pool, linked, ranges = link_blocks(
        monkeypatch, tmp_path, [(24, 6), (30, 5)]
    )
    for index, block_range in enumerate(ranges):
        torch.testing.assert_close(
            block_rows(pool, block_range),
            expected_global(raw_keys["block-%d" % index], block_range["start_token"]),
            atol=2e-5,
            rtol=2e-5,
        )
    assert [event["rope_remapped"] for event in linked] == [True, True]


def test_batched_and_per_block_paths_agree(monkeypatch, tmp_path):
    """The two link implementations must not disagree about the first block."""

    _, batched_pool, _, _ = link_blocks(
        monkeypatch, tmp_path, [(24, 6), (30, 5)], batched=True
    )
    _, per_block_pool, _, _ = link_blocks(
        monkeypatch, tmp_path, [(24, 6), (30, 5)], batched=False
    )
    torch.testing.assert_close(
        batched_pool.key, per_block_pool.key, atol=2e-5, rtol=2e-5
    )


# --- Case D: the trace reports the decision, not the ordinal ---------------


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize(
    ("blocks", "expected"),
    [
        ([(0, 6), (6, 5)], [False, True]),
        ([(24, 6), (30, 5)], [True, True]),
    ],
)
def test_trace_rope_remapped_follows_the_offset_not_the_index(
    monkeypatch, tmp_path, blocks, expected, batched
):
    _, _, linked, _ = link_blocks(monkeypatch, tmp_path, blocks, batched=batched)
    assert [event["block_index"] for event in linked] == [0, 1]
    assert [event["rope_remapped"] for event in linked] == expected


@pytest.mark.parametrize("batched", [False, True])
def test_rope_remap_trace_is_emitted_for_an_offset_first_block(
    monkeypatch, tmp_path, batched
):
    _, _, linked, _ = link_blocks(
        monkeypatch, tmp_path, [(24, 6), (30, 5)], batched=batched
    )
    assert linked[0]["global_offset"] == 24
    assert linked[0]["rope_remapped"] is True
