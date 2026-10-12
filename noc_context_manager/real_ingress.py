"""Step 1 adapters: external structured information -> canonical ``SessionEvent`` tuples.

ARCHITECTURE_STAGE: 1 -- Agent Information -> Canonical SessionEvent
CANONICAL_ENTRYPOINT: the functions of this module (one adapter per event type that has a real source)

Step 1 is deterministic and does **no** semantic classification. An adapter may only

* normalize an external schema (an OpenAI request, a Terminus turn) into events,
* give an event the ``EventType`` its *structure* names (a ``tool`` role message is a tool output, a
  ``tools`` entry is a tool definition, the last message of a request is the current request),
* keep the raw text exactly as the framework produced it,
* attach provenance (``ingress``, ``message_role``, ``tool_call_id`` ...) and a deterministic ``source_id``.

Which *field* an event becomes (Step 2), which namespace and root (Step 6) are decided elsewhere. Nothing
here reads text to guess a role: the evidence is the message ``role``, the call position, the request
fields, or a structure the agent that built the prompt declared and that this module re-verifies byte for byte.

Provenance contract (``PROVENANCE_CONTRACT``):

    event_id          derived from (event_type, source_id, raw_text)  -> deduplication
    source_id         deterministic, content-derived                  -> identity of one logical message
    sequence_number   acceptance order                                -> ordering
    session_revision  one per accepted batch                          -> revision / supersession
    raw_text          exact bytes                                     -> source reconstruction (spans)
    timestamp, task_id, metadata (tool_call_id, ingress, ...)         -> provenance only

Semantic identity is ``context_field`` + authoritative token ids and the cache key adds the model,
tokenizer and policy versions (``subcontext_store.semantic_identity``). A request id, a session id, a tool-call
id, a timestamp or a revision therefore never moves a reusable block's identity.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .session import EventType

INGRESS_VERSION = "real-ingress-v1"

#: ``(event_type, source_id, raw_text, metadata)`` -- what ``SessionEventCollector.add_events`` takes.
EventTuple = tuple[EventType, str, str, dict[str, Any]]

KIND_TOOL_OUTPUT = "tool_output"
KIND_AGENT_DIRECTIVE = "agent_directive"

PROVENANCE_CONTRACT: dict[str, dict[str, Any]] = {
    "event_id": {"deduplication": True, "ordering": False, "source_reconstruction": False, "semantic_identity": False, "cache_identity": False,
                 "note": "digest of (event_type, source_id, raw_text); the same triple is accepted once"},
    "task_id": {"deduplication": False, "ordering": False, "source_reconstruction": False, "semantic_identity": False, "cache_identity": False,
                "note": "provenance; one collector per task"},
    "session_revision": {"deduplication": False, "ordering": True, "source_reconstruction": False, "semantic_identity": False, "cache_identity": False,
                         "note": "one per accepted batch; a later revision of a source_id supersedes the earlier one"},
    "event_type": {"deduplication": True, "ordering": False, "source_reconstruction": False, "semantic_identity": True, "cache_identity": True,
                   "note": "decides the functional field, hence context_field, which semantic identity contains"},
    "source_id": {"deduplication": True, "ordering": False, "source_reconstruction": True, "semantic_identity": False, "cache_identity": False,
                  "note": "identifies one logical message; spans address text by it; excluded from semantic identity"},
    "raw_text": {"deduplication": True, "ordering": False, "source_reconstruction": True, "semantic_identity": True, "cache_identity": True,
                 "note": "exact bytes; the authoritative token ids of a block are derived from it"},
    "timestamp": {"deduplication": False, "ordering": False, "source_reconstruction": False, "semantic_identity": False, "cache_identity": False,
                  "note": "wall clock; volatile; never read by a decision"},
    "sequence_number": {"deduplication": False, "ordering": True, "source_reconstruction": False, "semantic_identity": False, "cache_identity": False,
                        "note": "acceptance order; the presentation order of sources"},
    "metadata": {"deduplication": False, "ordering": False, "source_reconstruction": False, "semantic_identity": False, "cache_identity": False,
                 "note": "provenance (ingress, message_role, tool_call_id, tool_name, call ordinal); hidden/oracle flags and path filter a source"},
}


# --- deterministic helpers ------------------------------------------------------------------------------


def canonical_json(value: Any) -> str:
    """JSON with sorted keys and no insignificant whitespace: dictionary order never changes the text."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_digest(*parts: str, size: int = 20) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:size]


def message_source_id(role: str, content: str) -> str:
    """The id the first agent integration used for a chat message; kept so existing sessions still dedupe."""

    return "message-" + content_digest(role, content)


def _text(content: Any) -> str:
    """OpenAI text or multi-part content as plain text (``None`` is the empty string)."""

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
                elif "content" in item:
                    parts.append(str(item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def _maybe_json(value: Any) -> Any:
    """A tool-call ``arguments`` string is JSON text; decode it so its key order cannot move the identity."""

    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


# --- tool definitions -------------------------------------------------------------------------------------


def _tool_name(tool: Mapping[str, Any]) -> str:
    function = tool.get("function") if isinstance(tool.get("function"), Mapping) else None
    return str((function or tool).get("name") or "")


def canonical_tool_schema(tool: Mapping[str, Any]) -> str:
    """One tool definition as deterministic text. Every key the client sent is kept; only the order is fixed."""

    return canonical_json(tool)


def tool_definition_events(tools: Sequence[Mapping[str, Any]] | None, *, functions: Sequence[Mapping[str, Any]] | None = None) -> list[EventTuple]:
    """``tools`` (and the legacy ``functions``) of an OpenAI request -> one ``TOOL_DEFINITION`` per tool.

    One event per tool: a tool added, removed or reordered between requests never invalidates the others, and
    a tool's source id is its *name*, so a changed schema under the same name supersedes the old revision while
    an unchanged one is deduplicated. Two tools sharing a name keep distinct ids (the schema digest is appended).
    """

    declared: list[Mapping[str, Any]] = [dict(item) for item in (tools or [])]
    declared += [{"type": "function", "function": dict(item)} for item in (functions or [])]
    names = [_tool_name(item) for item in declared]
    events: list[EventTuple] = []
    for index, tool in enumerate(declared):
        text = canonical_tool_schema(tool)
        name = names[index] or f"tool-{index}"
        source_id = f"tool-schema:{name}"
        if names.count(names[index]) > 1:
            source_id += f"#{content_digest(text, size=8)}"
        events.append(
            (
                EventType.TOOL_DEFINITION,
                source_id,
                text,
                {"ingress": "openai.tools", "ingress_version": INGRESS_VERSION, "tool_name": name},
            )
        )
    return events


# --- tool calls and outputs --------------------------------------------------------------------------------


def tool_call_event(call_id: str, tool_name: str, arguments: Any, *, ingress: str) -> EventTuple:
    """A tool call as ``{"arguments":..., "tool_name":...}``. The execution-only call id is metadata, never text."""

    text = canonical_json({"tool_name": tool_name, "arguments": _maybe_json(arguments)})
    return (
        EventType.TOOL_CALL,
        "tool-call:" + content_digest(text),
        text,
        {"ingress": ingress, "ingress_version": INGRESS_VERSION, "tool_call_id": call_id, "tool_name": tool_name},
    )


def tool_output_event(call_id: str, tool_name: str, content: str, *, ingress: str, extra: Mapping[str, Any] | None = None) -> EventTuple:
    """A tool result. The raw text is exactly what the tool returned."""

    return (
        EventType.TOOL_OUTPUT,
        "tool-output:" + content_digest(tool_name, content),
        content,
        {
            "ingress": ingress,
            "ingress_version": INGRESS_VERSION,
            "tool_call_id": call_id,
            "tool_name": tool_name,
            **dict(extra or {}),
        },
    )


# --- the ledger: what the session already holds ---------------------------------------------------------


@dataclass
class SourceLedger:
    """``source_id -> (latest event type, digest of its text)`` of one session.

    Decides skip / supersede from structure and exact text equality, never from reading the text: a source the
    session already holds *with the same text* is an exact repeat and is skipped whatever type it was ingested
    as (a first prompt that comes back as history stays the instruction); a source whose text changed (a tool
    schema revised under its name) is a new revision and is admitted.
    """

    latest: dict[str, tuple[EventType, str]] = field(default_factory=dict)

    @classmethod
    def from_events(cls, events: Iterable[Any]) -> "SourceLedger":
        ledger = cls()
        for event in sorted(events, key=lambda item: item.sequence_number):
            ledger.latest[event.source_id] = (event.event_type, content_digest(event.raw_text, size=32))
        return ledger

    def type_of(self, source_id: str) -> EventType | None:
        held = self.latest.get(source_id)
        return None if held is None else held[0]

    def admit(self, events: list[EventTuple], event: EventTuple) -> None:
        """Append ``event`` unless the session (or this batch) already holds that source with that text."""

        _kind, source_id, text, _metadata = event
        digest = content_digest(text, size=32)
        held = self.latest.get(source_id)
        if held is not None and held[1] == digest:
            return
        if any(item[1] == source_id and content_digest(item[2], size=32) == digest for item in events):
            return
        events.append(event)


# --- OpenAI-compatible requests -------------------------------------------------------------------------


def _chat_event_type(role: str) -> EventType:
    return EventType.TASK_INSTRUCTION if role in {"system", "developer"} else EventType.CONVERSATION_MESSAGE


def openai_request_events(
    messages: Sequence[Mapping[str, Any]],
    *,
    tools: Sequence[Mapping[str, Any]] | None = None,
    functions: Sequence[Mapping[str, Any]] | None = None,
    ledger: SourceLedger,
    first_turn: bool,
) -> list[EventTuple]:
    """One OpenAI chat request -> the events it adds to a session.

    Evidence used: the message ``role``, the message *position* (only the last message is the current
    request), ``tool_calls`` / ``tool_call_id`` and the ``tools`` field. The current request is a
    ``USER_REQUEST`` only from the second call of a session on: the first prompt of a session (one that holds no
    event yet) is the task prompt by the framework's own definition (it is ``TASK_INSTRUCTION``, as before). A user message that was
    the current request in an earlier call and is history now is re-emitted as a ``CONVERSATION_MESSAGE``
    under the same ``source_id`` -- the extractor keeps the latest revision of a source.
    """

    events: list[EventTuple] = []
    for event in tool_definition_events(tools, functions=functions):
        ledger.admit(events, event)
    call_names: dict[str, str] = {}
    last = len(messages) - 1
    for index, message in enumerate(messages):
        role = str(message.get("role") or "user")
        current = index == last
        content = _text(message.get("content"))
        calls = message.get("tool_calls") or []
        if role == "assistant" or calls:
            if content.strip():
                _admit_message(events, ledger, role, content, EventType.CONVERSATION_MESSAGE)
            for call in calls:
                function = call.get("function") if isinstance(call.get("function"), Mapping) else call
                name = str(function.get("name") or "")
                call_id = str(call.get("id") or "")
                call_names[call_id] = name
                ledger.admit(events, tool_call_event(call_id, name, function.get("arguments"), ingress="openai.messages"))
            continue
        if role in {"tool", "function"}:
            call_id = str(message.get("tool_call_id") or "")
            name = str(message.get("name") or call_names.get(call_id) or "")
            ledger.admit(
                events,
                tool_output_event(call_id, name, content, ingress="openai.messages", extra={"message_role": role}),
            )
            continue
        if role == "user" and current and not first_turn:
            _admit_message(events, ledger, role, content, EventType.USER_REQUEST)
        elif role == "user" and current and first_turn:
            _admit_message(events, ledger, role, content, EventType.TASK_INSTRUCTION)
        else:
            _admit_message(events, ledger, role, content, _chat_event_type(role))
    return events


def _admit_message(events: list[EventTuple], ledger: SourceLedger, role: str, content: str, kind: EventType) -> None:
    source_id = message_source_id(role, content)
    # A user message the session holds as the (then current) request is history now: supersede it.
    if kind is EventType.CONVERSATION_MESSAGE and ledger.type_of(source_id) is EventType.USER_REQUEST:
        events.append((kind, source_id, content, {"ingress": "openai.messages", "ingress_version": INGRESS_VERSION, "message_role": role, "settled_from": "user_request"}))
        return
    ledger.admit(
        events,
        (kind, source_id, content, {"ingress": "openai.messages", "ingress_version": INGRESS_VERSION, "message_role": role}),
    )


# --- Terminus turns (call(prompt, message_history)) ---------------------------------------------------------


@dataclass(frozen=True)
class PromptPart:
    """A piece of one prompt, declared by the agent that assembled it."""

    kind: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


def verified_parts(prompt: str, parts: Sequence[PromptPart] | None) -> tuple[PromptPart, ...] | None:
    """The declared parts if they re-assemble ``prompt`` byte for byte, else ``None``.

    The agent is trusted to say *what* it assembled, never to say it correctly: a declaration that does not
    join to the prompt is discarded and the prompt is treated as unstructured.
    """

    if not parts or any(part.kind not in {KIND_TOOL_OUTPUT, KIND_AGENT_DIRECTIVE} for part in parts):
        return None
    if "".join(part.text for part in parts) != prompt:
        return None
    return tuple(part for part in parts if part.text)


def split_around(prompt: str, needle: str, *, needle_kind: str = KIND_TOOL_OUTPUT, needle_metadata: Mapping[str, Any] | None = None) -> tuple[PromptPart, ...] | None:
    """``prompt`` as directive / ``needle`` / directive, when the agent built it around exactly one ``needle``."""

    if not needle or prompt.count(needle) != 1:
        return None
    before, after = prompt.split(needle, 1)
    parts = [PromptPart(KIND_AGENT_DIRECTIVE, before), PromptPart(needle_kind, needle, needle_metadata or {}), PromptPart(KIND_AGENT_DIRECTIVE, after)]
    return tuple(part for part in parts if part.text)


def part_event(part: PromptPart, *, ingress: str, current: bool) -> EventTuple:
    if part.kind == KIND_TOOL_OUTPUT:
        meta = dict(part.metadata)
        return tool_output_event(
            str(meta.pop("tool_call_id", "")),
            str(meta.pop("tool_name", "terminal")),
            part.text,
            ingress=ingress,
            extra=meta,
        )
    # An agent directive is text the agent addressed to the model in the user turn: the current request.
    # The whitespace at the seams between declared parts is the framework's own join, not content, and a block
    # that starts or ends with newlines does not tokenize the same next to the stitcher's separator (a token
    # crosses the edge). It is shaved here; a tool result keeps its exact bytes.
    text = part.text.strip()
    return (
        EventType.USER_REQUEST if current else EventType.CONVERSATION_MESSAGE,
        message_source_id("user", text),
        text,
        {"ingress": ingress, "ingress_version": INGRESS_VERSION, "message_role": "user", "prompt_part": part.kind, **dict(part.metadata)},
    )


class PromptRegistry:
    """Remembers how each declared prompt was split, so it can be settled when it returns as history.

    A prompt that was ingested as parts comes back, one call later, as *one* user message in
    ``message_history``. It must not be ingested again as a message (that would duplicate its text); its
    request-type parts are re-emitted as conversation instead.
    """

    def __init__(self) -> None:
        self._parts: dict[str, tuple[PromptPart, ...]] = {}

    @staticmethod
    def key(prompt: str) -> str:
        return content_digest(prompt, size=32)

    def remember(self, prompt: str, parts: tuple[PromptPart, ...]) -> None:
        self._parts[self.key(prompt)] = parts

    def lookup(self, prompt: str) -> tuple[PromptPart, ...] | None:
        return self._parts.get(self.key(prompt))


def terminus_turn_events(
    prompt: str,
    history: Sequence[Any],
    *,
    ledger: SourceLedger,
    registry: PromptRegistry,
    declared: Sequence[PromptPart] | None,
    first_turn: bool,
    message_value: Any,
    ingress: str = "terminus.call",
) -> tuple[list[EventTuple], dict[str, Any]]:
    """The events one ``call(prompt, message_history)`` of a Terminus agent adds, and what was structured.

    Terminus gives the LLM only ``prompt`` (the current user turn) and ``message_history`` (user/assistant
    messages). There is no system message, no tool schema and no structured tool result at this boundary; the
    only structure is the agent's own declaration of how it assembled ``prompt`` (``declared``), re-verified.
    """

    events: list[EventTuple] = []
    for message in history:
        role = str(message_value(message, "role", "user"))
        content = str(message_value(message, "content"))
        previous_parts = registry.lookup(content) if role == "user" else None
        if previous_parts is not None:
            # Already ingested as parts: settle the request-type ones, leave the tool outputs where they are.
            for part in previous_parts:
                if part.kind == KIND_AGENT_DIRECTIVE:
                    event = part_event(part, ingress=ingress, current=False)
                    if event[2].strip() and ledger.type_of(event[1]) is EventType.USER_REQUEST:
                        events.append(event)
            continue
        kind = _chat_event_type(role)
        source_id = message_source_id(role, content)
        if ledger.type_of(source_id) is EventType.USER_REQUEST and kind is EventType.CONVERSATION_MESSAGE:
            events.append((kind, source_id, content, {"ingress": ingress, "ingress_version": INGRESS_VERSION, "message_role": role, "settled_from": "user_request"}))
        else:
            ledger.admit(events, (kind, source_id, content, {"ingress": ingress, "ingress_version": INGRESS_VERSION, "message_role": role}))

    parts = verified_parts(prompt, declared)
    structure = {"declared": declared is not None, "verified": parts is not None, "parts": [part.kind for part in parts] if parts else []}
    if parts is not None and not first_turn:
        registry.remember(prompt, parts)
        present = [part for part in parts if part.text.strip()]
        for index, part in enumerate(present):
            # Only the end of the user turn can be the current request. Agent text that is followed by more of the
            # same prompt (a label before an observation) is framing: it goes in as conversation at once, because
            # a request that is re-typed later takes a later sequence number and would land after the content it
            # introduced.
            ledger.admit(events, part_event(part, ingress=ingress, current=index == len(present) - 1))
        return events, structure
    if first_turn:
        # The framework's first prompt is the task prompt: instruction and initial environment in one string.
        kind = EventType.TASK_INSTRUCTION
    else:
        kind = EventType.USER_REQUEST
    source_id = message_source_id("user", prompt)
    ledger.admit(events, (kind, source_id, prompt, {"ingress": ingress, "ingress_version": INGRESS_VERSION, "message_role": "user"}))
    return events, structure


__all__ = [
    "INGRESS_VERSION",
    "KIND_AGENT_DIRECTIVE",
    "KIND_TOOL_OUTPUT",
    "PROVENANCE_CONTRACT",
    "EventTuple",
    "PromptPart",
    "PromptRegistry",
    "SourceLedger",
    "canonical_json",
    "canonical_tool_schema",
    "content_digest",
    "message_source_id",
    "openai_request_events",
    "part_event",
    "split_around",
    "terminus_turn_events",
    "tool_call_event",
    "tool_definition_events",
    "tool_output_event",
    "verified_parts",
]
