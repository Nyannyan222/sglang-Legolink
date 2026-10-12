"""L1_: a Link asking for input logprobs must be refused at admission.

Audit finding L-1 (`legolink-link-input-logprobs-assert-fatal-result-processor`,
CWE-617). `PICRuntime.run_link` rebuilds its inner extend batches with
`return_logprob=False`, so the logits output it returns has no
`input_token_logprobs`. The scheduler's result processor still counts
`extend_input_len - extend_logprob_start_len` input logprobs for the request and
asserts the field is present; the assertion fires in the event loop and the
scheduler process answers by signalling its parent. OpenAI `echo=true` with
`logprobs` sets `logprob_start_len=0`, and `/generate` lets the client set it.

The fix refuses the request at the scheduler's admission boundary, as a
request-scoped abort, instead of serving it without the logprobs it asked for.
The GPU path (`run_link`, the result processor) cannot run here, so the tests
pin (a) the decision function against an independent reference of the quantity
that reaches the assertion, and (b) the wiring in the scheduler source.
"""

from __future__ import annotations

import ast
from types import SimpleNamespace

import pytest

from tests.pic.support import SRT_ROOT, load_srt_module

pytestmark = [pytest.mark.unit, pytest.mark.cpu]


@pytest.fixture(scope="module")
def pic():
    return load_srt_module("pic_runtime", "pic_runtime.py")


def link_sp():
    return SimpleNamespace(
        custom_params={"pic_link": {"cache_keys": ["k"], "recompute_initial_tokens": 2}}
    )


def compile_sp():
    return SimpleNamespace(
        custom_params={"sub_context_metadata": {"cache_key": "k"}}
    )


def native_sp():
    return SimpleNamespace(custom_params=None)


def check(pic, sampling_params, *, return_logprob, start, length=8):
    return pic.validate_pic_link_logprob_request(
        sampling_params,
        return_logprob=return_logprob,
        logprob_start_len=start,
        input_length=length,
    )


# --- the five cases the task names ------------------------------------------------


def test_L1_link_with_logprob_start_len_zero_is_refused(pic):
    """The OpenAI `echo=true` + `logprobs` shape."""

    message = check(pic, link_sp(), return_logprob=True, start=0)
    assert message and "input-token logprobs" in message


def test_L1_link_with_a_nonzero_input_logprob_start_is_refused(pic):
    for start in (1, 3, 7):
        assert check(pic, link_sp(), return_logprob=True, start=start), start


def test_L1_a_normal_link_without_input_logprobs_is_admitted(pic):
    # logprobs off
    assert check(pic, link_sp(), return_logprob=False, start=-1) is None
    # logprobs off, a stale start the scheduler would have reset anyway
    assert check(pic, link_sp(), return_logprob=False, start=0) is None
    # output logprobs only: "from the end"
    assert check(pic, link_sp(), return_logprob=True, start=-1) is None
    # start exactly at the end of the input: zero input logprobs
    assert check(pic, link_sp(), return_logprob=True, start=8, length=8) is None


def test_L1_a_non_pic_request_with_input_logprobs_is_untouched(pic):
    assert check(pic, native_sp(), return_logprob=True, start=0) is None
    assert check(pic, native_sp(), return_logprob=True, start=3) is None
    # A Compile is a native prefill and does return input logprobs.
    assert check(pic, compile_sp(), return_logprob=True, start=0) is None


def test_L1_a_malformed_pic_link_is_not_mistaken_for_a_link(pic):
    """A non-object `pic_link` is not a Link plan; the decision function must not raise."""

    for bad in (None, 5, "x", [], [1]):
        sampling_params = SimpleNamespace(custom_params={"pic_link": bad})
        assert check(pic, sampling_params, return_logprob=True, start=0) is None


# --- against an independent reference of what reaches the assertion --------------


def reference_num_input_logprobs(*, start, input_length):
    """`Req.set_extend_input_len` for a Link: no prefix, one unchunked extend.

    Written from the scheduler's contract, not from the helper under test: a
    start of -1 means "from the end of the fill ids"; otherwise the absolute
    start, clipped to the extend length. A Link request has a private radix
    namespace, so the prefix is empty.
    """

    prefix = 0
    logprob_start = input_length if start == -1 else max(start, prefix)
    extend_logprob_start = min(logprob_start - prefix, input_length)
    return input_length - extend_logprob_start


@pytest.mark.parametrize("length", [1, 2, 8, 33])
def test_L1_refusal_is_exactly_when_the_result_processor_would_assert(pic, length):
    for start in range(-1, length + 1):
        wanted = reference_num_input_logprobs(start=start, input_length=length) > 0
        refused = (
            check(pic, link_sp(), return_logprob=True, start=start, length=length)
            is not None
        )
        assert refused == wanted, (start, length)


# --- request-scoped, not process-wide -----------------------------------------------


def scheduler_function(name):
    source = (SRT_ROOT / "managers" / "scheduler.py").read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def test_L1_the_scheduler_refuses_the_request_instead_of_raising():
    function = scheduler_function("handle_generate_request")
    calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "validate_pic_link_logprob_request"
    ]
    assert len(calls) == 1, "expected exactly one admission-time logprob check"

    # The check is followed, in the same block, by the abort-and-return pattern
    # every other admission rejection in this function uses.
    body = function.body
    index = next(
        i
        for i, statement in enumerate(body)
        if isinstance(statement, ast.Assign)
        and "validate_pic_link_logprob_request" in ast.unparse(statement)
    )
    following = ast.unparse(body[index + 1])
    assert "if error_msg" in following
    assert "set_finish_with_abort(error_msg)" in following
    assert "return" in following


def test_L1_the_check_runs_after_the_effective_logprob_start_is_known():
    source = ast.unparse(scheduler_function("handle_generate_request"))
    start_assigned = source.index("req.logprob_start_len = recv_req.logprob_start_len")
    bound_checked = source.index("req.logprob_start_len > len(req.origin_input_ids)")
    link_checked = source.index("validate_pic_link_logprob_request(")
    queued = source.index("self.grammar_manager.process_req_with_grammar(req)")
    assert start_assigned < bound_checked < link_checked < queued


def test_L1_the_refusal_is_a_bad_request_abort_not_an_exception(pic):
    """`set_finish_with_abort` turns the request into a 400 with logprobs off."""

    source = (SRT_ROOT / "managers" / "schedule_batch.py").read_text(encoding="utf-8")
    abort = source[source.index("def set_finish_with_abort") :]
    abort = abort[: abort.index("def update_reasoning_tokens")]
    assert "self.return_logprob = False" in abort
    assert "HTTPStatus.BAD_REQUEST" in abort
