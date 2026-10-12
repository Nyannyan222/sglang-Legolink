from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from .reference.repair_reference import final_row_logits
from .support import FakeRotary, fixed_generator, make_pic_runtime

pytestmark = [pytest.mark.unit, pytest.mark.cpu]


def test_grouped_repair_returns_only_final_request_logits():
    _, runtime = make_pic_runtime(FakeRotary())
    all_logits = torch.tensor(
        [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
    )
    output = SimpleNamespace(
        logits_output=SimpleNamespace(next_token_logits=all_logits.clone())
    )

    selected = runtime._select_final_grouped_request_output(
        output, grouped_batch_size=3
    )

    assert torch.equal(
        selected.logits_output.next_token_logits, all_logits[-1:]
    )
    assert not torch.equal(
        selected.logits_output.next_token_logits, all_logits[:1]
    )


def test_final_row_reference_matches_sequential_final_logits():
    hidden_rows = [
        torch.randn(2, 4, generator=fixed_generator(11)),
        torch.randn(1, 4, generator=fixed_generator(12)),
        torch.randn(3, 4, generator=fixed_generator(13)),
    ]
    lm_head = torch.randn(4, 7, generator=fixed_generator(14))
    grouped = torch.cat(hidden_rows)

    actual = final_row_logits(grouped, [2, 1, 3], lm_head)
    expected = hidden_rows[-1][-1:] @ lm_head

    torch.testing.assert_close(actual, expected)


def test_final_row_selector_fails_closed_on_shape_mismatch():
    module, runtime = make_pic_runtime(FakeRotary())
    output = SimpleNamespace(next_token_logits=torch.zeros(2, 4))

    with pytest.raises(module.PICRuntimeError, match="batch size"):
        runtime._select_final_grouped_request_output(output, grouped_batch_size=3)

