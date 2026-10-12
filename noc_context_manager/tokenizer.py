"""Token counting abstractions shared by the NOC context pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
from pathlib import Path
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Any, Protocol, runtime_checkable

from .token_accounting import (
    EstimatedTokenCount,
    ModelTokenCount,
    TokenAccountingError,
    TokenUnit,
    require_authoritative_tokenizer,
)


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_./:+-]+|[\u4e00-\u9fff]|[^\s]")


@runtime_checkable
class TextTokenizer(Protocol):
    """Minimal tokenizer contract needed by context management."""

    @property
    def fingerprint(self) -> str: ...

    def tokenize(self, text: str) -> Sequence[Any]: ...

    def count(self, text: str) -> int: ...


@runtime_checkable
class ChatTemplateTokenizer(TextTokenizer, Protocol):
    """Tokenizer that can render the server/model chat-template token layout."""

    def tokenize_messages(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        reasoning_effort: str | None = None,
        continue_final_message: bool = False,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> Sequence[int]: ...


class RegexTokenizer:
    #: An approximation, and it says so. Nothing that decides runtime behaviour,
    #: cache placement or compile eligibility may consume its counts.
    token_unit = TokenUnit.ESTIMATED

    """Dependency-free estimator used when a model tokenizer is unavailable."""

    fingerprint = "regex-ascii-cjk-punct-v1"

    def tokenize(self, text: str) -> tuple[str, ...]:
        return tuple(TOKEN_PATTERN.findall(text))

    def count(self, text: str) -> int:
        return len(TOKEN_PATTERN.findall(text))


class CallableTokenizer:
    """Adapter for a model tokenizer's encode-like callable.

    Estimated unless the caller both declares it authoritative and names the
    model, because an arbitrary callable cannot be assumed to be the serving
    model's tokenizer. Wrapping a real one is a claim the wrapper has to make
    on purpose -- and one it must be able to back with a model id.
    """

    #: Overwritten per instance in __init__; the class default is the safe one.
    token_unit = TokenUnit.ESTIMATED
    backend_name = "callable"

    def __init__(
        self,
        encode: Callable[[str], Sequence[Any]],
        *,
        fingerprint: str,
        model_id: str = "",
        authoritative: bool = False,
    ) -> None:
        if not fingerprint.strip():
            raise ValueError("tokenizer fingerprint must not be empty")
        if authoritative and not model_id.strip():
            raise TokenAccountingError(
                "an authoritative CallableTokenizer must name the model whose "
                "tokenizer it wraps; a fingerprint alone cannot establish that"
            )
        self._encode = encode
        self._fingerprint = fingerprint
        self.model_id = model_id
        self.token_unit = (
            TokenUnit.MODEL if authoritative else TokenUnit.ESTIMATED
        )

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def tokenize(self, text: str) -> tuple[Any, ...]:
        return tuple(self._encode(text))

    def count(self, text: str) -> int:
        return len(self._encode(text))


class HuggingFaceTokenizer:
    token_unit = TokenUnit.MODEL
    backend_name = "transformers"

    """Use the same tokenizer implementation and vocabulary as the model."""

    def __init__(
        self,
        backend: Any,
        *,
        model_id: str,
        revision: str | None = None,
    ) -> None:
        self.backend = backend
        self.model_id = model_id
        metadata = {
            "model_id": model_id,
            "revision": revision
            or getattr(backend, "init_kwargs", {}).get("_commit_hash"),
            "tokenizer_class": (
                f"{type(backend).__module__}.{type(backend).__qualname__}"
            ),
            "name_or_path": str(getattr(backend, "name_or_path", model_id)),
            "vocab_size": int(getattr(backend, "vocab_size", -1)),
            "special_tokens_map": getattr(backend, "special_tokens_map", {}),
        }
        encoded = json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        self._fingerprint = "hf-" + hashlib.sha256(encoded).hexdigest()[:24]

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        revision: str | None = None,
        local_files_only: bool = True,
        trust_remote_code: bool = True,
    ) -> "HuggingFaceTokenizer":
        from transformers import AutoTokenizer

        kwargs: dict[str, Any] = {
            "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code,
        }
        if revision:
            kwargs["revision"] = revision
        backend = AutoTokenizer.from_pretrained(model_id, **kwargs)
        return cls(backend, model_id=model_id, revision=revision)

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def tokenize(self, text: str) -> tuple[int, ...]:
        return tuple(
            int(value)
            for value in self.backend.encode(text, add_special_tokens=False)
        )

    def count(self, text: str) -> int:
        return len(self.backend.encode(text, add_special_tokens=False))

    def tokenize_messages(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        reasoning_effort: str | None = None,
        continue_final_message: bool = False,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> tuple[int, ...]:
        if continue_final_message:
            raise ValueError(
                "HuggingFace chat-token audit does not support "
                "continue_final_message"
            )
        kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": False,
        }
        if tools is not None:
            kwargs["tools"] = list(tools)
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        if chat_template_kwargs:
            kwargs.update(chat_template_kwargs)
        values = self.backend.apply_chat_template(list(messages), **kwargs)
        if isinstance(values, dict):
            values = values.get("input_ids")
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError(
                "HuggingFace apply_chat_template returned invalid token ids"
            )
        return tuple(int(value) for value in values)


class TokenizersJSONChatTokenizer:
    token_unit = TokenUnit.MODEL
    backend_name = "tokenizers-json"

    """Dependency-light tokenizer.json backend with chat-template rendering."""

    def __init__(
        self,
        *,
        tokenizer_json: Path,
        tokenizer_config_json: Path,
        model_id: str,
        revision: str | None = None,
    ) -> None:
        from jinja2 import Environment
        from tokenizers import Tokenizer

        self.model_id = model_id
        self.tokenizer_json = tokenizer_json
        self.tokenizer_config_json = tokenizer_config_json
        self.backend = Tokenizer.from_file(str(tokenizer_json))
        config = json.loads(tokenizer_config_json.read_text(encoding="utf-8"))
        template = str(config.get("chat_template") or "")
        if not template.strip():
            raise ValueError(f"tokenizer config has no chat_template: {tokenizer_config_json}")
        self._chat_template = template
        self._environment = Environment(autoescape=False)
        metadata = {
            "backend": "tokenizers-json-chat-template-v1",
            "model_id": model_id,
            "revision": revision or _snapshot_revision(tokenizer_json),
            "tokenizer_json_sha256": _file_sha256(tokenizer_json),
            "tokenizer_config_sha256": _file_sha256(tokenizer_config_json),
        }
        encoded = json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        self._fingerprint = "tokjson-" + hashlib.sha256(encoded).hexdigest()[:24]

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str,
        *,
        revision: str | None = None,
    ) -> "TokenizersJSONChatTokenizer":
        root = _resolve_tokenizer_root(model_id_or_path, revision=revision)
        tokenizer_json = root / "tokenizer.json" if root.is_dir() else root
        tokenizer_config_json = (
            tokenizer_json.parent / "tokenizer_config.json"
        )
        if not tokenizer_json.exists():
            raise FileNotFoundError(f"tokenizer.json not found for {model_id_or_path}")
        if not tokenizer_config_json.exists():
            raise FileNotFoundError(
                f"tokenizer_config.json not found beside {tokenizer_json}"
            )
        return cls(
            tokenizer_json=tokenizer_json,
            tokenizer_config_json=tokenizer_config_json,
            model_id=model_id_or_path,
            revision=revision,
        )

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def tokenize(self, text: str) -> tuple[int, ...]:
        return tuple(
            int(value)
            for value in self.backend.encode(text, add_special_tokens=False).ids
        )

    def count(self, text: str) -> int:
        return len(self.tokenize(text))

    def tokenize_messages(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        reasoning_effort: str | None = None,
        continue_final_message: bool = False,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> tuple[int, ...]:
        if continue_final_message:
            raise ValueError(
                "tokenizers-json chat-token audit does not support "
                "continue_final_message"
            )
        values: dict[str, Any] = {
            "messages": [dict(message) for message in messages],
            "tools": list(tools) if tools is not None else None,
            "tool_choice": tool_choice,
            "reasoning_effort": reasoning_effort,
            "add_generation_prompt": True,
        }
        if chat_template_kwargs:
            values.update(chat_template_kwargs)
        rendered = self._environment.from_string(self._chat_template).render(**values)
        return self.tokenize(rendered)


class SGLangHTTPTokenizer:
    token_unit = TokenUnit.MODEL
    backend_name = "sglang-http"

    """Use the tokenizer owned by the running SGLang model server."""

    def __init__(
        self,
        *,
        api_base: str,
        model_id: str,
        api_key: str = "",
        timeout: float = 1200,
        transport: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        if not api_base.strip():
            raise ValueError("SGLang tokenizer mode requires api_base")
        self.api_base = api_base.rstrip("/")
        self.model_id = model_id
        self.api_key = api_key
        self.timeout = timeout
        self._transport = transport or self._post
        encoded = json.dumps(
            {
                "contract": "sglang-v1-tokenize-no-special-v1",
                "model_id": model_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self._fingerprint = "sglang-" + hashlib.sha256(encoded).hexdigest()[:24]

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.api_base}/tokenize",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("SGLang /tokenize returned a non-object response")
        return value

    @staticmethod
    def _response_token_ids(response: dict[str, Any]) -> tuple[int, ...]:
        values = response.get("tokens")
        if not isinstance(values, list) or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in values
        ):
            raise ValueError("SGLang /tokenize response has invalid token ids")
        count = response.get("count")
        if count is not None and count != len(values):
            raise ValueError("SGLang /tokenize count does not match token ids")
        return tuple(values)

    @lru_cache(maxsize=4096)
    def _tokenize_payload_cached(self, serialized_payload: str) -> tuple[int, ...]:
        payload = json.loads(serialized_payload)
        return self._response_token_ids(self._transport(payload))

    def _tokenize_payload(self, payload: dict[str, Any]) -> tuple[int, ...]:
        serialized = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return self._tokenize_payload_cached(serialized)

    def tokenize(self, text: str) -> tuple[int, ...]:
        return self._tokenize_payload(
            {
                "model": self.model_id,
                "prompt": text,
                "add_special_tokens": False,
            }
        )

    def count(self, text: str) -> int:
        return len(self.tokenize(text))

    def tokenize_messages(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        reasoning_effort: str | None = None,
        continue_final_message: bool = False,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> tuple[int, ...]:
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": [dict(message) for message in messages],
            "continue_final_message": continue_final_message,
        }
        if tools is not None:
            payload["tools"] = list(tools)
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
        if chat_template_kwargs is not None:
            payload["chat_template_kwargs"] = dict(chat_template_kwargs)
        return self._tokenize_payload(payload)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot_revision(path: Path) -> str | None:
    parts = path.resolve().parts
    if "snapshots" not in parts:
        return None
    index = parts.index("snapshots")
    if index + 1 >= len(parts):
        return None
    return parts[index + 1]


def _hf_cache_roots() -> list[Path]:
    roots = []
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        value = os.environ.get(name)
        if value:
            roots.append(Path(value))
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    return roots


def _resolve_tokenizer_root(model_id_or_path: str, *, revision: str | None = None) -> Path:
    path = Path(model_id_or_path)
    if path.exists():
        return path
    repo_dir_name = "models--" + model_id_or_path.replace("/", "--")
    for cache_root in _hf_cache_roots():
        snapshots = cache_root / repo_dir_name / "snapshots"
        if revision:
            candidate = snapshots / revision
            if candidate.exists():
                return candidate
        if snapshots.exists():
            candidates = [
                candidate
                for candidate in snapshots.iterdir()
                if (candidate / "tokenizer.json").exists()
                and (candidate / "tokenizer_config.json").exists()
            ]
            if candidates:
                return max(candidates, key=lambda item: item.stat().st_mtime)
    raise FileNotFoundError(
        f"could not resolve a local tokenizer cache for {model_id_or_path}"
    )


#: The estimating tokenizer. Named for what it does rather than for being the
#: default, because "default" is what let it stand in for the model's tokenizer
#: on paths that needed the model's tokenizer.
ESTIMATE_TOKENIZER: TextTokenizer = RegexTokenizer()

#: Compatibility alias. Still the regex estimator, and still not authoritative
#: for anything -- `require_authoritative_tokenizer` rejects it. Prefer
#: `ESTIMATE_TOKENIZER` in new code so the name states the unit.
DEFAULT_TOKENIZER: TextTokenizer = ESTIMATE_TOKENIZER


def load_text_tokenizer(
    model_id: str,
    *,
    mode: str = "model",
    tokenizer_path: str | None = None,
    revision: str | None = None,
    local_files_only: bool = True,
    trust_remote_code: bool = True,
    api_base: str | None = None,
    api_key: str = "",
    timeout: float = 1200,
) -> TextTokenizer:
    normalized = mode.strip().lower()
    if normalized == "regex":
        return RegexTokenizer()
    if normalized == "model":
        return HuggingFaceTokenizer.from_pretrained(
            tokenizer_path or model_id,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
    if normalized == "tokenizers":
        return TokenizersJSONChatTokenizer.from_pretrained(
            tokenizer_path or model_id,
            revision=revision,
        )
    if normalized == "sglang":
        return SGLangHTTPTokenizer(
            api_base=api_base or "",
            model_id=model_id,
            api_key=api_key,
            timeout=timeout,
        )
    raise ValueError(f"unknown tokenizer mode: {mode}")


def tokenize_text(
    text: str,
    tokenizer: TextTokenizer | None = None,
) -> tuple[Any, ...]:
    """Tokens from `tokenizer`, or from the estimator when none is given.

    Kept permissive because its callers use it for shape and preview, never for
    a count that decides anything. `count_tokens` is where the unit mattered
    and that one no longer guesses.
    """

    return tuple((tokenizer or ESTIMATE_TOKENIZER).tokenize(text))


def estimate_tokens(
    text: str, tokenizer: TextTokenizer | None = None
) -> EstimatedTokenCount:
    """A cheap approximation, labelled as one.

    Legitimate uses: budgeting a segmentation pass before the authoritative
    tokenizer runs, sizing a preflight, previewing a report. Not legitimate:
    anything that reaches the compile floor, a context budget, a link plan, or
    a benchmark table presented as actual tokens.
    """

    chosen = tokenizer or ESTIMATE_TOKENIZER
    return EstimatedTokenCount(
        value=chosen.count(text),
        tokenizer_fingerprint=str(getattr(chosen, "fingerprint", "unknown-tokenizer")),
    )


def count_model_tokens(text: str, tokenizer: TextTokenizer) -> ModelTokenCount:
    """What the serving model's tokenizer counts. No fallback, by construction.

    `tokenizer` is positional and required. There is deliberately no default:
    the whole defect this replaces was a default that silently supplied the
    regex estimator to callers who believed they were counting model tokens.
    """

    identity = require_authoritative_tokenizer(
        tokenizer, purpose="count_model_tokens"
    )
    return ModelTokenCount(
        value=tokenizer.count(text),
        tokenizer_fingerprint=identity.fingerprint,
        model_id=identity.model_id,
    )


def count_tokens(
    text: str,
    tokenizer: TextTokenizer | None = None,
) -> int:
    """A bare integer, from an explicitly named tokenizer.

    The unit-free return type is why this now refuses to pick a tokenizer for
    you: an `int` cannot carry its own unit, so the caller has to have decided.
    New code should prefer `estimate_tokens` or `count_model_tokens`, which
    return a value that knows what it is.
    """

    if tokenizer is None:
        raise TokenAccountingError(
            "count_tokens no longer defaults to the regex estimator, because "
            "the returned int cannot say which unit it is in. Pass a tokenizer "
            "explicitly, or call estimate_tokens()/count_model_tokens()."
        )
    return tokenizer.count(text)
