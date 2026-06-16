"""Append-only tool call audit log."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import PROJECT_ROOT, resolve_path
from app.utils.logging import get_logger

LOGGER = get_logger("tools")
LOG_PATH = PROJECT_ROOT / "logs" / "tool_calls.jsonl"


def log_tool_event(
    event: str,
    *,
    call_id: str = "",
    name: str = "",
    args: dict[str, Any] | None = None,
    success: bool | None = None,
    detail: str = "",
) -> None:
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "call_id": call_id,
        "name": name,
        "args": args or {},
        "success": success,
        "detail": detail[:2000],
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    LOGGER.info("%s %s %s", event, name, call_id)


def read_recent_log(limit: int = 20) -> str:
    if not LOG_PATH.exists():
        return "No tool calls logged yet."
    lines = LOG_PATH.read_text(encoding="utf-8").strip().splitlines()
    recent = lines[-limit:]
    if not recent:
        return "No tool calls logged yet."
    parts = [f"Recent tool log ({len(recent)} entries):", ""]
    for line in recent:
        try:
            entry = json.loads(line)
            parts.append(
                f"  [{entry.get('ts', '?')}] {entry.get('event')} "
                f"{entry.get('name', '')} — {entry.get('detail', '')[:120]}"
            )
        except json.JSONDecodeError:
            parts.append(f"  {line[:120]}")
    return "\n".join(parts)
