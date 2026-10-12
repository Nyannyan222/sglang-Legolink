"""A host block catalog that outlives one process -- for live multi-agent Terminal-Bench runs.

`tb run` builds a fresh agent per trial, and each agent session builds its own `CacheCatalog`, so the
host forgets at every task boundary which immutable blocks the patched runtime already holds. Agent B,
run as a separate trial or a separate `tb run`, would Compile a block agent A compiled minutes before.

`PersistentCacheCatalog` is the minimum to keep that knowledge: the same interface, backed by an
append-only JSONL file that every session sharing it reads before it answers. It holds only what
`CacheCatalog` holds -- compiled immutable block keys and their compatibility metadata -- never a
conversation, a manifest, a mutable suffix or an output. Gate 1 reads its entries exactly as before.

It records host intent, not runtime residency: if the runtime has lost a block, the strict Link fails
closed (and residency recovery, when enabled, recompiles it), exactly as with the in-memory catalog.
Deleting the file is a CATALOG_RESET.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from .compile_policy import CacheCatalog


class PersistentCacheCatalog(CacheCatalog):
    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._offset = 0
        self._file_guard = threading.Lock()
        self._refresh()

    def _refresh(self) -> None:
        """Apply whatever other sessions appended since the last read."""

        with self._file_guard, self.path.open("rb") as handle:
            handle.seek(self._offset)
            data = handle.read()
            # Only whole lines are consumed. A peer's append can be visible half written (a long line, or one whose write
            # was split by the OS); reading the fragment as a line would drop the event for good, because the offset would
            # already be past it when its second half arrives. The fragment stays unread until its newline is there.
            complete = data[: data.rfind(b"\n") + 1]
            self._offset += len(complete)
            lines = complete.decode("utf-8", errors="replace").split("\n")
        with self._guard:
            for line in lines:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key, op = event.get("cache_key"), event.get("op")
                if op == "compiled":
                    self._entries[key] = {**event.get("metadata", {}), "stale": False}
                elif op == "stale" and key in self._entries:
                    self._entries[key]["stale"] = True
                elif op == "invalidate":
                    self._entries.pop(key, None)

    def _append(self, event: dict) -> None:
        with self._file_guard, self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, separators=(",", ":"), default=str) + "\n")

    def contains(self, cache_key: str) -> bool:
        self._refresh()
        return super().contains(cache_key)

    def metadata(self, cache_key: str):
        self._refresh()
        return super().metadata(cache_key)

    def find_by_physical_identity(self, digest: str):
        self._refresh()
        return super().find_by_physical_identity(digest)

    def live_entries(self):
        self._refresh()
        return super().live_entries()

    def active_by_namespace(self):
        self._refresh()
        return super().active_by_namespace()

    def mark_compiled(self, cache_key: str, **metadata: object) -> None:
        # Append the row `mark_compiled` just wrote, not `self.metadata(cache_key)`: that call refreshes from the file
        # first, and a peer's `invalidate` of this very key appended since the last read would pop the row, so the
        # event appended here would carry empty metadata and every session replaying the file would hold a row with
        # no namespace, physical identity or token hash.
        row = super().mark_compiled(cache_key, **metadata)
        self._append({"op": "compiled", "cache_key": cache_key,
                      "metadata": {k: v for k, v in row.items() if k != "stale"}})

    def mark_stale(self, cache_key: str) -> None:
        super().mark_stale(cache_key)
        self._append({"op": "stale", "cache_key": cache_key})

    def invalidate(self, cache_key: str) -> None:
        super().invalidate(cache_key)
        self._append({"op": "invalidate", "cache_key": cache_key})


__all__ = ["PersistentCacheCatalog"]
