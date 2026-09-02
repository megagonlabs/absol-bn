"""On-disk SQLite cache for LLM chat-completion responses.

Keyed by the full request payload (model, messages, max tokens, temperature,
reasoning_effort), so any change to prompts, model name, or sampling params
automatically misses. Intended scope: one cache file per (run, dataset),
shared across model variants and trials so repeated prompts (column_grouping,
parent_ordering, etc.) are computed once.
"""

import hashlib
import json
import os
import sqlite3
import threading
from typing import Any, Dict, List, Optional


def make_key(
    *,
    model: str,
    messages: List[Dict[str, str]],
    max_completion_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    reasoning_effort: Optional[str] = None,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_completion_tokens,
        "temperature": temperature,
        "reasoning_effort": reasoning_effort,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class LlmCache:
    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        # The sqlite3 module's per-connection state (cursors, implicit transactions)
        # is not safe to share across threads even with check_same_thread=False, so we
        # serialize all operations on the connection with a Python-level lock. Cache
        # calls are sub-millisecond, so this does not bottleneck threaded LLM fan-out.
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False, timeout=30.0)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS responses ("
            "key TEXT PRIMARY KEY, "
            "content TEXT NOT NULL, "
            "usage TEXT)"
        )
        self._conn.commit()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT content, usage FROM responses WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        content, usage_json = row
        return {"content": content, "usage": json.loads(usage_json) if usage_json else None}

    def put(self, key: str, content: str, usage: Optional[Dict[str, int]] = None) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses (key, content, usage) VALUES (?, ?, ?)",
                (key, content, json.dumps(usage) if usage else None),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
