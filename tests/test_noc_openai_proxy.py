import json
import sys
import types


class FakeBaseLLM:
    def __init__(self, **kwargs):
        if kwargs:
            raise TypeError(f"unexpected BaseLLM kwargs: {sorted(kwargs)}")


class FakeOutputLengthExceededError(Exception):
    def __init__(self, message, truncated_response=None):
        super().__init__(message)
        self.truncated_response = truncated_response


class FakeTerminus:
    def __init__(self, **kwargs):
        self._llm = None


def install_stubs():
    definitions = {
        "terminal_bench": {},
        "terminal_bench.agents": {},
        "terminal_bench.agents.base_agent": {
            "AgentResult": type("AgentResult", (), {})
        },
        "terminal_bench.agents.terminus_1": {"Terminus": FakeTerminus},
        "terminal_bench.llms": {},
        "terminal_bench.llms.base_llm": {
            "BaseLLM": FakeBaseLLM,
            "OutputLengthExceededError": FakeOutputLengthExceededError,
        },
        "terminal_bench.terminal": {},
        "terminal_bench.terminal.tmux_session": {
            "TmuxSession": type("TmuxSession", (), {})
        },
    }
    for name, attrs in definitions.items():
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module


install_stubs()


from terminal_bench_integration.noc_openai_proxy import (
    LEGOLINK_ROUTE,
    TERMINAL_BENCH_JSON_GUARDRAIL,
    NOCOpenAIProxy,
    ProxyConfig,
    message_content_to_text,
    messages_to_prompt_and_history,
    normalize_model_name,
    request_session_key,
    terminal_bench_guardrail_replan_prompt,
    terminal_bench_guardrail_safe_diagnostic_response,
    terminal_bench_guardrail_violation,
    terminal_bench_inject_verification,
    terminal_bench_repair_json_response,
    terminal_bench_verification_probe,
)


class Headers(dict):
    pass


def test_normalize_litellm_model_prefix() -> None:
    assert normalize_model_name("openai/Qwen/Qwen2.5-3B-Instruct") == (
        "Qwen/Qwen2.5-3B-Instruct"
    )
    assert normalize_model_name("Qwen/Qwen2.5-3B-Instruct") == (
        "Qwen/Qwen2.5-3B-Instruct"
    )


def test_message_content_to_text_supports_openai_content_parts() -> None:
    content = [
        {"type": "text", "text": "first"},
        {"type": "image_url", "image_url": {"url": "ignored"}},
        {"content": "second"},
    ]
    assert message_content_to_text(content) == "first\nsecond"


def test_messages_to_prompt_and_history_uses_last_message_as_prompt() -> None:
    prompt, history = messages_to_prompt_and_history(
        [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "state"},
            {"role": "assistant", "content": "previous"},
            {"role": "user", "content": "next action?"},
        ]
    )
    assert prompt == "next action?"
    assert history == [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "state"},
        {"role": "assistant", "content": "previous"},
    ]


def test_session_key_prefers_explicit_header() -> None:
    key = request_session_key(
        headers=Headers({"X-NOC-Session-ID": "task/attempt:1"}),
        payload={"user": "ignored"},
        messages=[{"role": "user", "content": "hello"}],
        model="model",
    )
    assert key.startswith("x-noc-session-id_task_attempt_1_model")


def test_session_key_without_an_explicit_identity_is_opaque_not_content_derived() -> None:
    """Replaces the test that pinned a deterministic first-message hash.

    That determinism was the L-4 finding: anyone who knew the opening prompt
    computed the same key and joined the session. A request that names no
    session now gets a fresh random id; continuing a conversation without an
    explicit identity is the session registry's job
    (tests/test_security_l4_proxy_session_isolation.py).
    """

    kwargs = {
        "headers": Headers({}),
        "payload": {},
        "messages": [{"role": "user", "content": "same first prompt"}],
        "model": "model",
    }
    assert request_session_key(**kwargs).startswith("derived-")
    assert request_session_key(**kwargs) != request_session_key(**kwargs)


def test_proxy_route_constant_is_optimized_legolink_route() -> None:
    assert LEGOLINK_ROUTE == "selected_context_radixcache_legolink"


def test_terminal_bench_json_guardrail_targets_confirmation_loop() -> None:
    assert "task_complete" in TERMINAL_BENCH_JSON_GUARDRAIL
    assert "do not rerun" in TERMINAL_BENCH_JSON_GUARDRAIL
    assert "printf" in TERMINAL_BENCH_JSON_GUARDRAIL
    assert "Do not assume helper scripts already exist" in (
        TERMINAL_BENCH_JSON_GUARDRAIL
    )
    assert "No such file or directory" in TERMINAL_BENCH_JSON_GUARDRAIL
    assert "python3 - <<'PY'" in TERMINAL_BENCH_JSON_GUARDRAIL
    assert "do not use a long python3 -c one-liner" in (
        TERMINAL_BENCH_JSON_GUARDRAIL
    )
    assert "cat > /tmp/solve.py <<'PY'" in TERMINAL_BENCH_JSON_GUARDRAIL
    assert "parse correctly" in TERMINAL_BENCH_JSON_GUARDRAIL
    assert "schema.json" in TERMINAL_BENCH_JSON_GUARDRAIL
    assert "csv.DictReader" in TERMINAL_BENCH_JSON_GUARDRAIL
    assert "observed fieldnames" in TERMINAL_BENCH_JSON_GUARDRAIL
    assert "never guess column names" in TERMINAL_BENCH_JSON_GUARDRAIL
    assert "KeyError" in TERMINAL_BENCH_JSON_GUARDRAIL
    assert "never use json.load on a .csv file" in TERMINAL_BENCH_JSON_GUARDRAIL
    assert "JSONDecodeError" in TERMINAL_BENCH_JSON_GUARDRAIL


def test_terminal_bench_guardrail_violation_detects_known_bad_commands() -> None:
    assert terminal_bench_guardrail_violation("python3 -c '" + "x" * 950) == (
        "unsafe python3 -c command with multi-statement or compound Python "
        "detected; use a heredoc or write a temporary Python file instead"
    )
    assert terminal_bench_guardrail_violation(
        "python3 -c 'items=[]; for row in rows: items.append(row)'"
    ) == (
        "unsafe python3 -c command with multi-statement or compound Python "
        "detected; use a heredoc or write a temporary Python file instead"
    )
    assert terminal_bench_guardrail_violation(
        "{\"commands\":[{\"keystrokes\":\"python3 - <<'PY'\\n\"}]}"
    ) == (
        "heredoc opener detected without a standalone closing PY marker in "
        "the same keystrokes string"
    )
    assert terminal_bench_guardrail_violation(
        "{\"commands\":[{\"keystrokes\":\"cat > /tmp/solve.py <<'PY' && "
        "python3 /tmp/solve.py\\n\"}]}"
    ) == (
        "empty or malformed heredoc detected; put the script body after the "
        "heredoc opener and a standalone closing PY marker before running it"
    )
    assert terminal_bench_guardrail_violation(
        "{\"commands\":[{\"keystrokes\":\"python3 - <<'PY'\\nprint(1)\\nPY\\n\"}]}"
    ) == (
        "interactive python heredoc detected; Terminal-Bench JSON commands "
        "must put the complete heredoc body and closing PY marker inside one "
        "keystrokes string"
    )
    assert terminal_bench_guardrail_violation(
        'python3 -c "json.load(open(\\"/app/data/input.csv\\"))"'
    ) == "CSV input is being parsed with json.load instead of csv.DictReader"
    assert terminal_bench_guardrail_violation(
        "python3 /app/process_csv.py /app/data/a.csv /tmp/a.json"
    ) == "helper script path is assumed before verifying or creating the file"
    assert terminal_bench_guardrail_violation(
        "python3 /app/transform_csv_to_json.py"
    ) == "helper script path is assumed before verifying or creating the file"


def test_terminal_bench_guardrail_violation_ignores_analysis_mentions() -> None:
    response = json.dumps(
        {
            "analysis": "I should not use python3 -c or json.load on a .csv file.",
            "plan": "Use safe commands.",
            "commands": [{"keystrokes": "printf '%s\\n' ok\n", "duration": 0.1}],
        }
    )
    assert terminal_bench_guardrail_violation(response) is None


def test_terminal_bench_guardrail_replan_prompt_preserves_parser_contract() -> None:
    prompt = terminal_bench_guardrail_replan_prompt("bad command")
    assert "same parser format" in prompt
    assert "cat > /tmp/solve.py <<'PY'" in prompt
    assert "single commands[i].keystrokes string" in prompt
    assert "Return only the JSON object" in prompt
    assert "Do not repeat" in prompt


def test_terminal_bench_guardrail_safe_diagnostic_response_is_parser_safe() -> None:
    response = terminal_bench_guardrail_safe_diagnostic_response("bad command")
    parsed = json.loads(response)

    assert parsed["commands"]
    command = parsed["commands"][0]["keystrokes"]
    assert "NOC_GUARDRAIL_BLOCKED: bad command" in command
    assert "NOC_GUARDRAIL_NEXT_ACTION:" in command
    assert "find /app -maxdepth 2 -type f" in command
    assert "sed -n '1,8p'" in command
    assert terminal_bench_guardrail_violation(response) is None


def test_terminal_bench_repair_json_response_converts_python_c_to_script() -> None:
    original = json.dumps(
        {
            "analysis": "write file",
            "plan": "run python",
            "commands": [
                {
                    "keystrokes": (
                        "python3 -c 'import json; with open(\"/tmp/out.json\", "
                        "\"w\") as f: json.dump({\"ok\": True}, f)'"
                    ),
                    "duration": 1.0,
                }
            ],
        }
    )

    repaired, reason = terminal_bench_repair_json_response(original)
    parsed = json.loads(repaired)
    command = parsed["commands"][0]["keystrokes"]

    assert reason
    assert "cat > /tmp/solve.py <<'PY'" in command
    assert "\nwith open" in command
    assert "\nPY\npython3 /tmp/solve.py\n" in command
    assert terminal_bench_guardrail_violation(repaired) is None


def test_proxy_error_log_is_compact_and_prompt_safe(tmp_path) -> None:
    proxy = NOCOpenAIProxy(
        ProxyConfig(
            host="127.0.0.1",
            port=32200,
            model="Qwen/Qwen2.5-3B-Instruct",
            patched_api_base="http://127.0.0.1:32102/v1",
            original_api_base="http://127.0.0.1:32102/v1",
            variant=LEGOLINK_ROUTE,
            operation_mode="strict_experimental",
            segmentation_mode="deterministic",
            selection_mode="rule",
            compile_policy_mode="legacy_fixed_64",
            context_token_budget=24000,
            legolink_k=128,
            max_tokens=1024,
            temperature=0.0,
            tokenizer_mode="sglang",
            artifacts_root=tmp_path,
            timeout=1200.0,
            api_key="",
            allow_streaming_fallback=False,
            terminal_bench_json_guardrail=True,
        )
    )
    proxy.record_error(
        request_path="/v1/chat/completions",
        payload={
            "model": "openai/Qwen/Qwen2.5-3B-Instruct",
            "messages": [{"role": "user", "content": "secret prompt body"}],
            "max_tokens": 128,
        },
        error=RuntimeError("backend connection refused"),
    )

    error_log = tmp_path / "proxy_errors.jsonl"
    record = error_log.read_text(encoding="utf-8")
    assert "backend connection refused" in record
    assert "secret prompt body" not in record


# --------------------------------------------------------------------------
# Observation verification probes
# --------------------------------------------------------------------------
#
# The failure these pin down is concrete. On the 2026-08-27 paired run the
# agent solved hello-world on episode 1 with `echo 'Hello, world!' >
# /app/hello.txt`, then emitted the byte-identical command seven more times,
# reasoning each time that "no files or directories exist within it", because
# a successful redirect prints nothing and the observation it got back was an
# empty prompt line. The trial scored unresolved with the task already done.


def test_verification_probe_makes_a_silent_redirect_observable() -> None:
    probe = terminal_bench_verification_probe(
        "echo 'Hello, world!' > /app/hello.txt\n"
    )

    assert probe is not None
    assert "NOC_VERIFY" in probe
    assert "ls -ld '/app/hello.txt'" in probe
    # Existence alone does not prove the contents are right; the probe makes an
    # empty file visible instead of leaving a blank pane for the agent to
    # over-interpret.
    assert "NOC_VERIFY_FILE_BYTES" in probe
    assert "NOC_VERIFY_FILE_HEAD_BEGIN" in probe
    assert "NOC_VERIFY_FILE_HEAD_END" in probe
    assert "head -c 400 '/app/hello.txt'" in probe


def test_verification_probe_covers_silent_mutators() -> None:
    assert "ls -ld '/app'" in (terminal_bench_verification_probe("mkdir -p /app\n") or "")
    assert "ls -ld '/app/run.sh'" in (
        terminal_bench_verification_probe("chmod 644 /app/run.sh\n") or ""
    )


def test_verification_probe_shows_the_copy_destination_not_the_source() -> None:
    probe = terminal_bench_verification_probe("cp /app/a.csv /app/b.csv\n")

    assert probe is not None
    # cp/mv/ln put the destination last; showing the source proves nothing.
    assert "head -c 400 '/app/b.csv'" in probe
    assert "head -c 400 '/app/a.csv'" not in probe


def test_verification_probe_falls_back_to_the_workdir_for_generated_scripts() -> None:
    # A generated script writes files the proxy cannot name, so the evidence
    # has to be the task directory. This is the count-dataset-tokens shape.
    probe = terminal_bench_verification_probe("python3 /tmp/solve.py\n")

    assert probe is not None
    assert "ls -la '/app'" in probe


def test_verification_probe_skips_commands_that_change_nothing() -> None:
    for read_only in (
        "cat /app/schema.json\n",
        "ls -la /app\n",
        "grep foo /app/x.txt 2>&1\n",
    ):
        assert terminal_bench_verification_probe(read_only) is None


def test_verification_probe_ignores_mode_arguments_and_devnull() -> None:
    # chmod 644 puts a mode in argument position, and /dev/null is not evidence.
    assert "'644'" not in (terminal_bench_verification_probe("chmod 644 /app/f.txt\n") or "")
    assert terminal_bench_verification_probe("cmd > /dev/null\n") is None


def test_inject_verification_extends_duration_so_the_probe_is_captured() -> None:
    original = json.dumps(
        {
            "analysis": "write it",
            "plan": "echo",
            "commands": [
                {"keystrokes": "echo 'Hello, world!' > /app/hello.txt\n", "duration": 1.0}
            ],
        }
    )

    injected, reason = terminal_bench_inject_verification(original)
    command = json.loads(injected)["commands"][0]

    assert reason == "/app/hello.txt"
    assert command["keystrokes"].startswith("echo 'Hello, world!' > /app/hello.txt\n")
    assert "NOC_VERIFY" in command["keystrokes"]
    # The harness captures the pane after `duration`; a probe that has not
    # printed by then would be invisible and the whole thing pointless.
    assert command["duration"] > 1.0


def test_inject_verification_is_idempotent() -> None:
    original = json.dumps(
        {"commands": [{"keystrokes": "mkdir -p /app\n", "duration": 1.0}]}
    )

    once, first_reason = terminal_bench_inject_verification(original)
    twice, second_reason = terminal_bench_inject_verification(once)

    assert first_reason
    # A replan can route the same content through again; probing twice would
    # double the block and inflate the observation for no extra evidence.
    assert second_reason is None
    assert twice == once


def test_inject_verification_leaves_read_only_responses_untouched() -> None:
    original = json.dumps(
        {"commands": [{"keystrokes": "cat /app/schema.json\n", "duration": 1.0}]}
    )

    injected, reason = terminal_bench_inject_verification(original)

    assert reason is None
    assert injected == original


def test_probe_output_never_trips_the_guardrail() -> None:
    original = json.dumps(
        {"commands": [{"keystrokes": "echo x > /app/out.json\n", "duration": 1.0}]}
    )

    injected, _ = terminal_bench_inject_verification(original)

    # The probe runs after the guardrail check, so it must not be the thing
    # that would have failed it.
    assert terminal_bench_guardrail_violation(injected) is None


def test_guardrail_tells_the_model_to_read_the_probe_instead_of_repeating() -> None:
    assert "NOC_VERIFY" in TERMINAL_BENCH_JSON_GUARDRAIL
    assert "instead of running the same command again" in TERMINAL_BENCH_JSON_GUARDRAIL


# --------------------------------------------------------------------------
# One guardrail, both agent loops
# --------------------------------------------------------------------------


def test_guardrail_and_probes_live_in_the_shared_module() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "terminal_bench_integration"
    shared = (root / "terminal_bench_guardrail.py").read_text(encoding="utf-8")
    proxy = (root / "noc_openai_proxy.py").read_text(encoding="utf-8")
    agent = (root / "sglang_pic_agent.py").read_text(encoding="utf-8")

    # The definitions live in one place.
    assert "def terminal_bench_verification_probe" in shared
    assert "TERMINAL_BENCH_JSON_GUARDRAIL = (" in shared

    # And nowhere else. This is the whole point: the guardrail used to be
    # proxy-only, so it reached the paired runner and any-agent runs but never
    # sglang_pic_agent -- which is what the H100 formal runs execute. A fix
    # landed in one loop and silently missed the other.
    assert "def terminal_bench_verification_probe" not in proxy
    assert "TERMINAL_BENCH_JSON_GUARDRAIL = (" not in proxy
    assert "def terminal_bench_verification_probe" not in agent

    # Both loops import it.
    assert "from .terminal_bench_guardrail import" in proxy
    assert "from terminal_bench_integration.terminal_bench_guardrail import" in agent


def test_both_loops_apply_the_same_probe_to_the_same_command() -> None:
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "terminal_bench_integration"
    spec = importlib.util.spec_from_file_location(
        "shared_guardrail_under_test", root / "terminal_bench_guardrail.py"
    )
    shared = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(shared)

    keystrokes = "echo 'Hello, world!' > /app/hello.txt\n"

    # The proxy re-exports the same object rather than a copy, so the two loops
    # cannot diverge in what they consider a silent state change.
    assert terminal_bench_verification_probe.__doc__ == (
        shared.terminal_bench_verification_probe.__doc__
    )
    assert terminal_bench_verification_probe(keystrokes) == (
        shared.terminal_bench_verification_probe(keystrokes)
    )
