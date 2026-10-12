"""Terminal-Bench command guardrail and observation verification probes.

Shared by both agent loops in this package, which is the point of the module.
The guardrail and the probes used to live only in :mod:`noc_openai_proxy`, so
they applied to the paired Agent Quality runner and to any-agent proxy runs but
never to ``sglang_pic_agent``, which is what the H100 formal runs actually
execute. A fix landed in one loop silently missed the other; the 2026-08-28
H100 run had neither the guardrail nor the probes.

Everything here is a pure function of the model's response text, with no proxy,
session, or HTTP dependency, so either loop can apply it.
"""

from __future__ import annotations

import json
import re

TERMINAL_BENCH_JSON_GUARDRAIL = (
    "\n\nNOC proxy note for Terminal-Bench JSON agents: If the previous command "
    "batch already completed the requested file or state change and the harness "
    "asks for confirmation, do not rerun the same shell commands. Return an empty "
    'commands array with "task_complete": true. Prefer exact file writes such as '
    "printf over repeated append commands when exact file contents are required. "
    "Do not assume helper scripts already exist in /app; inspect the filesystem "
    "before calling script paths. If a command fails with No such file or directory, "
    "do not repeat the same failing command. Instead create the missing file or use "
    "a self-contained command such as python3 - <<'PY' with only standard-library "
    "code. For non-trivial Python, do not use a long python3 -c one-liner; write a "
    "temporary .py file with cat > /tmp/solve.py <<'PY' or use a heredoc so shell "
    "quotes cannot corrupt dictionary keys. Do not overwrite task-provided input "
    "files such as schema.json or files "
    "under /app/data unless the task explicitly asks you to modify them. For CSV "
    "inputs, inspect the exact header and at least one sample row, then parse with "
    "csv.DictReader using the observed fieldnames; never guess column names and "
    "never use json.load on a .csv file. If Python raises KeyError, JSONDecodeError, "
    "FileNotFoundError, or another Traceback, inspect the relevant file/header and "
    "change the command instead of repeating it. Before marking "
    "task_complete, verify required output files exist, are non-empty, and parse "
    "correctly when the task expects JSON/CSV/text output. "
    "The proxy appends a read-only NOC_VERIFY block after commands that change "
    "state silently, so the terminal output already shows whether the file now "
    "exists and what it contains. Read that block before deciding what to do "
    "next: a file with 0 bytes or an empty NOC_VERIFY_FILE_HEAD block does not "
    "prove correct contents and must be fixed before completion. If the block "
    'shows the required state, return "task_complete": true with an empty '
    "commands array instead of running the same command again."
)
TERMINAL_BENCH_GUARDRAIL_REPLAN_PREFIX = (
    "Your previous response cannot be sent to the Terminal-Bench shell because "
    "it violates the NOC proxy safety/correctness guardrail: "
)
TERMINAL_BENCH_SAFE_DIAGNOSTIC_COMMAND = """pwd
find /app -maxdepth 2 -type f | sort
for f in /app/schema.json /app/data/*.csv; do
  if [ -f "$f" ]; then
    echo "===== $f ====="
    sed -n '1,8p' "$f"
  fi
done
"""


def _shell_single_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _terminal_bench_command_texts(content: str) -> list[str] | None:
    json_content = _extract_first_json_object(content)
    if json_content is None:
        return None
    try:
        parsed = json.loads(json_content)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    commands = parsed.get("commands")
    if not isinstance(commands, list):
        return None
    command_texts: list[str] = []
    for command in commands:
        if isinstance(command, dict) and isinstance(command.get("keystrokes"), str):
            command_texts.append(command["keystrokes"])
    return command_texts


def _terminal_bench_command_violation(command_text: str) -> str | None:
    lowered = command_text.lower()
    if "<<'py' &&" in lowered or '<<"py" &&' in lowered:
        return (
            "empty or malformed heredoc detected; put the script body after the "
            "heredoc opener and a standalone closing PY marker before running it"
        )
    has_closing_heredoc = "\npy\n" in lowered or "\\npy\\n" in lowered
    if "<<'py'" in lowered and not has_closing_heredoc:
        return (
            "heredoc opener detected without a standalone closing PY marker in "
            "the same keystrokes string"
        )
    if "python3 - <<'py'" in lowered:
        return (
            "interactive python heredoc detected; Terminal-Bench JSON commands "
            "must put the complete heredoc body and closing PY marker inside one "
            "keystrokes string"
        )
    if "python3 -c" in lowered and (
        len(command_text) > 900
        or "; for " in lowered
        or "; with " in lowered
        or "\nfor " in lowered
        or "\nwith " in lowered
    ):
        return (
            "unsafe python3 -c command with multi-statement or compound Python "
            "detected; use a heredoc or write a temporary Python file instead"
        )
    if ".csv" in lowered and "json.load" in lowered:
        return "CSV input is being parsed with json.load instead of csv.DictReader"
    if re.search(
        r"(/app/)?[A-Za-z0-9_-]*(process|transform|convert)[A-Za-z0-9_-]*\.py\b",
        command_text,
    ):
        return "helper script path is assumed before verifying or creating the file"
    if (
        "cat /app/schema.json" in command_text
        and command_text.count("cat /app/schema.json") > 1
    ):
        return "repeated schema inspection without a changed follow-up action"
    return None


def terminal_bench_guardrail_violation(content: str) -> str | None:
    """Detect agent responses that match known Terminal-Bench failure loops."""

    command_texts = _terminal_bench_command_texts(content)
    if command_texts is not None:
        for command_text in command_texts:
            violation = _terminal_bench_command_violation(command_text)
            if violation is not None:
                return violation
        return None
    return _terminal_bench_command_violation(content)


def terminal_bench_guardrail_replan_prompt(reason: str) -> str:
    return (
        TERMINAL_BENCH_GUARDRAIL_REPLAN_PREFIX
        + reason
        + "\n\nReturn a corrected response in the same parser format requested by "
        "the agent. Use only commands that you expect to run successfully. For "
        "structured data tasks, inspect headers first, then use a heredoc such as "
        "cat > /tmp/solve.py <<'PY' followed by python3 /tmp/solve.py. If the "
        "agent parser expects JSON, the complete shell command, including every "
        "newline, the script body, the closing PY marker, and python3 /tmp/solve.py, "
        "must be inside a single commands[i].keystrokes string. Do not put raw "
        "script text as a separate JSON array item. Do not emit an empty heredoc "
        "like cat > /tmp/solve.py <<'PY' && python3 /tmp/solve.py. Return only "
        "the JSON object, with no prose before or after it. Do not repeat a "
        "command that already failed."
    )


def terminal_bench_guardrail_safe_diagnostic_response(reason: str) -> str:
    """Return a parser-safe fallback command after repeated unsafe replans."""

    command = (
        f"printf '%s\\n' {_shell_single_quote('NOC_GUARDRAIL_BLOCKED: ' + reason)}\n"
        "printf '%s\\n' 'NOC_GUARDRAIL_NEXT_ACTION: Do not repeat the blocked "
        "command. Inspect the real files below, then write a complete temporary "
        "script file before running it.'\n"
        f"{TERMINAL_BENCH_SAFE_DIAGNOSTIC_COMMAND}"
    )
    return json.dumps(
        {
            "analysis": (
                "The previous candidate response was blocked by the NOC proxy "
                f"guardrail: {reason}. I will inspect the actual task files and "
                "CSV headers before attempting another write."
            ),
            "plan": (
                "Inspect /app, schema.json, and data CSV headers/sample rows; "
                "use the observed output in the next action."
            ),
            "commands": [
                {
                    "keystrokes": command,
                    "duration": 1.0,
                }
            ],
        },
        ensure_ascii=False,
    )


def _extract_first_json_object(content: str) -> str | None:
    """Extract the first balanced JSON object from mixed model output."""

    json_start = -1
    brace_count = 0
    in_string = False
    escape_next = False
    for index, char in enumerate(content):
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            if brace_count == 0:
                json_start = index
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0 and json_start >= 0:
                return content[json_start : index + 1]
    return None


def _normalize_python_one_liner(code: str) -> str:
    """Make common shell-safe Python one-liners valid as a script file."""

    code = code.strip()
    for keyword in ("with", "for", "if", "try", "from", "import", "print"):
        code = re.sub(rf";\s*({keyword}\b)", r"\n\1", code)
    return code


def _repair_python_c_command(keystrokes: str) -> str | None:
    """Convert a risky python3 -c shell command into a temp script heredoc."""

    match = re.search(
        r"(?P<prefix>.*?)(?P<python>python3\s+-c\s+)(?P<quote>['\"])(?P<code>.*?)(?P=quote)(?P<suffix>.*)$",
        keystrokes.strip(),
        flags=re.DOTALL,
    )
    if not match:
        return None

    prefix = match.group("prefix").strip()
    code = _normalize_python_one_liner(match.group("code"))
    suffix = match.group("suffix").strip()
    commands: list[str] = []
    if prefix:
        prefix = re.sub(r"\s*&&\s*$", "", prefix)
        if prefix:
            commands.append(prefix)
    commands.extend(
        [
            "cat > /tmp/solve.py <<'PY'",
            code,
            "PY",
            f"python3 /tmp/solve.py{(' ' + suffix) if suffix else ''}",
        ]
    )
    return "\n".join(commands) + "\n"


def terminal_bench_repair_json_response(content: str) -> tuple[str, str | None]:
    """Repair known bad Terminus JSON command shapes before parser execution."""

    json_content = _extract_first_json_object(content)
    if json_content is None:
        return content, None
    try:
        parsed = json.loads(json_content)
    except json.JSONDecodeError:
        return content, None
    if not isinstance(parsed, dict):
        return content, None
    commands = parsed.get("commands")
    if not isinstance(commands, list):
        return content, None

    changed = False
    reasons: list[str] = []
    for command in commands:
        if not isinstance(command, dict):
            continue
        keystrokes = command.get("keystrokes")
        if not isinstance(keystrokes, str):
            continue
        violation = terminal_bench_guardrail_violation(
            json.dumps({"commands": [command]}, ensure_ascii=False)
        )
        repaired = None
        if violation and "python3 -c" in keystrokes.lower():
            repaired = _repair_python_c_command(keystrokes)
        if repaired is None:
            continue
        command["keystrokes"] = repaired
        command.setdefault("duration", 1.0)
        changed = True
        reasons.append(violation or "repaired python3 -c command")

    if not changed:
        return content, None

    parsed["analysis"] = (
        str(parsed.get("analysis") or "")
        + "\n\nNOC proxy command repair: converted unsafe python3 -c command(s) "
        "into complete /tmp/solve.py heredoc command(s)."
    ).strip()
    parsed["plan"] = (
        str(parsed.get("plan") or "")
        + "\n\nExecute the repaired script command and verify outputs before "
        "marking the task complete."
    ).strip()
    return json.dumps(parsed, ensure_ascii=False), "; ".join(reasons)


# ---------------------------------------------------------------------------
# Observation verification probes
# ---------------------------------------------------------------------------
#
# A Terminal-Bench observation contains only what the shell printed. The
# commands these tasks actually need -- a redirect, mkdir, chmod, cp, tar, or a
# generated /tmp/solve.py -- all succeed silently, so the observation after a
# successful write is a bare prompt line indistinguishable from the one before
# it. The model has no evidence that anything happened.
#
# On the 2026-08-27 paired run this cost hello-world every trial: the agent
# wrote the correct file on episode 1, then repeated the byte-identical command
# seven more times, each time reasoning that "no files or directories exist
# within it", until max_episodes ran out and the trial scored unresolved.
#
# The probe appends read-only evidence commands after a silent state change so
# the next observation shows the new state. It lives in the proxy, so both
# Agent Quality arms receive it byte-identically and the paired comparison
# still changes exactly one variable.

NOC_VERIFY_MARKER = "NOC_VERIFY"
_VERIFY_MAX_PATHS = 3
_VERIFY_HEAD_BYTES = 400
# Extra seconds granted to a probed command. The harness captures the pane
# after `duration`, so a probe that has not printed yet would be invisible --
# which would defeat the entire point of adding it.
_VERIFY_DURATION_BONUS = 1.0
_VERIFY_TEXT_SUFFIXES = (
    ".txt",
    ".json",
    ".jsonl",
    ".csv",
    ".tsv",
    ".md",
    ".py",
    ".sh",
    ".yaml",
    ".yml",
)
# A write redirect, ignoring 2>&1 style descriptor duplication.
_VERIFY_REDIRECT = re.compile(r"(?<![0-9&>])>>?\s*(?P<path>[^\s;|&<>()]+)")
_VERIFY_MUTATORS = re.compile(
    r"\b(?P<verb>mkdir|touch|cp|mv|chmod|chown|ln)\b(?P<args>[^;|&\n]*)"
)
# A generated script writes files this proxy cannot name, so the evidence has
# to be the task working directory instead. /app is the Terminal-Bench task
# root, the same assumption TERMINAL_BENCH_SAFE_DIAGNOSTIC_COMMAND already makes.
_VERIFY_SCRIPT_RUN = re.compile(r"\b(?:python3?|bash|sh)\s+/tmp/[A-Za-z0-9_.-]+")
_VERIFY_WORKDIR = "/app"


def _verify_path_candidate(token: str) -> str | None:
    """Accept only tokens that genuinely look like a filesystem path."""

    token = token.strip().strip("'\"")
    if not token or token[0] in "-+$":
        return None
    if token in {"/dev/null", "/dev/stdout", "/dev/stderr"}:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_./~-]+", token):
        return None
    # chmod 644 /app/f puts a mode in argument position; a path has a separator
    # or an extension, a mode has neither.
    if "/" not in token and "." not in token:
        return None
    return token


def _verify_candidate_paths(keystrokes: str) -> list[str]:
    """Collect the paths a command writes to, in first-seen order."""

    paths: list[str] = []

    def add(token: str) -> None:
        path = _verify_path_candidate(token)
        if path is not None and path not in paths:
            paths.append(path)

    for match in _VERIFY_REDIRECT.finditer(keystrokes):
        add(match.group("path"))
    for match in _VERIFY_MUTATORS.finditer(keystrokes):
        for token in match.group("args").split():
            add(token)
    return paths[:_VERIFY_MAX_PATHS]


def terminal_bench_verification_probe(keystrokes: str) -> str | None:
    """Return read-only commands that make a silent state change observable."""

    if NOC_VERIFY_MARKER in keystrokes:
        return None
    paths = _verify_candidate_paths(keystrokes)
    runs_generated_script = bool(_VERIFY_SCRIPT_RUN.search(keystrokes))
    if not paths and not runs_generated_script:
        return None
    lines = [
        "printf '%s\\n' "
        + _shell_single_quote(f"{NOC_VERIFY_MARKER}: state after the command above")
    ]
    if paths:
        quoted = " ".join(_shell_single_quote(path) for path in paths)
        lines.append(f"ls -ld {quoted} 2>&1")
    if runs_generated_script:
        lines.append(f"ls -la {_shell_single_quote(_VERIFY_WORKDIR)} 2>&1 | head -n 20")
    # cp/mv/ln put the destination last, and a redirect has only one path, so
    # the last path is the one worth showing the contents of.
    for path in reversed(paths):
        if not path.lower().endswith(_VERIFY_TEXT_SUFFIXES):
            continue
        quoted_path = _shell_single_quote(path)
        lines.extend(
            [
                f"if [ -f {quoted_path} ]; then",
                "  printf '%s ' 'NOC_VERIFY_FILE_BYTES'",
                f"  wc -c < {quoted_path}",
                "  printf '%s\\n' 'NOC_VERIFY_FILE_HEAD_BEGIN'",
                f"  head -c {_VERIFY_HEAD_BYTES} {quoted_path}",
                "  printf '\\n%s\\n' 'NOC_VERIFY_FILE_HEAD_END'",
                "fi",
            ]
        )
        break
    return "\n".join(lines) + "\n"


def terminal_bench_inject_verification(content: str) -> tuple[str, str | None]:
    """Append observation evidence to silent state-changing commands."""

    json_content = _extract_first_json_object(content)
    if json_content is None:
        return content, None
    try:
        parsed = json.loads(json_content)
    except json.JSONDecodeError:
        return content, None
    if not isinstance(parsed, dict):
        return content, None
    commands = parsed.get("commands")
    if not isinstance(commands, list):
        return content, None

    probed: list[str] = []
    for command in commands:
        if not isinstance(command, dict):
            continue
        keystrokes = command.get("keystrokes")
        if not isinstance(keystrokes, str) or not keystrokes.strip():
            continue
        probe = terminal_bench_verification_probe(keystrokes)
        if probe is None:
            continue
        if not keystrokes.endswith("\n"):
            keystrokes += "\n"
        command["keystrokes"] = keystrokes + probe
        try:
            duration = float(command.get("duration", 1.0))
        except (TypeError, ValueError):
            duration = 1.0
        command["duration"] = duration + _VERIFY_DURATION_BONUS
        # A generated script contributes no named path, so record the working
        # directory rather than dropping the reason and reporting no change.
        probed.extend(_verify_candidate_paths(keystrokes) or [_VERIFY_WORKDIR])

    if not probed:
        return content, None
    return json.dumps(parsed, ensure_ascii=False), ", ".join(probed)
