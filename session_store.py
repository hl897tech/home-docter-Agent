from __future__ import annotations
from typing import Dict, List, Tuple
import time

History = List[Tuple[str, str]]  # (role, content)

class SessionStore:
    def __init__(self, ttl_seconds: int = 60 * 60 * 6, max_turns: int = 20):
        self.ttl = ttl_seconds
        self.max_turns = max_turns
        self._data: Dict[str, Dict] = {}

    def get(self, session_id: str) -> History:
        now = time.time()
        s = self._data.get(session_id)
        if (s is None) or (now - s["ts"] > self.ttl):
            self._data[session_id] = {"ts": now, "history": []}
        else:
            s["ts"] = now
        return self._data[session_id]["history"]

    def append(self, session_id: str, role: str, content: str):
        hist = self.get(session_id)
        hist.append((role, content))

        # 只保留最近 max_turns 轮（每轮两条：user+assistant）
        keep = self.max_turns * 2
        if len(hist) > keep:
            del hist[:-keep]

STORE = SessionStore()