"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Start a request trace, retaining a timestamp for latency tracking."""
        key = request_id or user_id
        started = time.monotonic()
        self._open[key] = started
        if not hasattr(self, "_pending"):
            self._pending = {}
        self._pending[key] = {
            "request_id": key,
            "user_id": user_id,
            "input": text,
            "started_at": utc_now_iso(),
        }
        return key

    def record_output(
        self, *, user_id: str, text: str, blocked: bool = False,
        layer: str | None = None, request_id: str | None = None,
    ):
        """Complete a trace, preserving the actual enforcement outcome."""
        key = request_id or user_id
        started = self._open.pop(key, None)
        base = getattr(self, "_pending", {}).pop(key, {})
        base.update({
            "request_id": key,
            "user_id": user_id,
            "output": text,
            "blocked": bool(blocked),
            "layer": layer,
            "finished_at": utc_now_iso(),
            "latency_ms": round((time.monotonic() - started) * 1000, 2)
            if started is not None else None,
        })
        self.logs.append(base)
        return base

    def export_json(self, filepath: str | None = None):
        """Export real recorded entries; never manufacture placeholder results."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.logs, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
