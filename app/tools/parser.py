"""Parse tool call JSON blocks from assistant replies."""

from __future__ import annotations

import json
import re
from typing import Any

from app.tools.models import ToolCall

_TOOL_BLOCK_RE = re.compile(r"```tool\s*\n?(.*?)\n?```", re.DOTALL | re.IGNORECASE)


def parse_tool_calls(text: str) -> tuple[str, list[ToolCall], str | None]:
    """Return (display_text, tool_calls, parse_error)."""
    match = _TOOL_BLOCK_RE.search(text)
    if not match:
        return text, [], None

    display = (text[: match.start()] + text[match.end() :]).strip()
    raw_json = match.group(1).strip()
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as error:
        return display, [], f"Invalid tool JSON: {error}"

    calls = _extract_calls(data)
    if not calls and isinstance(data, dict) and data.get("name"):
        calls = [_tool_call_from_dict(data)]

    return display, calls, None


def _extract_calls(data: Any) -> list[ToolCall]:
    if isinstance(data, dict) and "tools" in data:
        items = data.get("tools")
    elif isinstance(data, list):
        items = data
    else:
        return []

    calls: list[ToolCall] = []
    if not isinstance(items, list):
        return calls
    for item in items:
        if isinstance(item, dict):
            call = _tool_call_from_dict(item)
            if call is not None:
                calls.append(call)
    return calls


def _tool_call_from_dict(data: dict[str, Any]) -> ToolCall | None:
    name = str(data.get("name", "")).strip()
    if not name:
        return None
    args = data.get("args")
    if not isinstance(args, dict):
        args = {k: v for k, v in data.items() if k not in ("name", "rationale")}
    rationale = str(data.get("rationale", "")).strip()
    return ToolCall(name=name, args=dict(args), rationale=rationale)
