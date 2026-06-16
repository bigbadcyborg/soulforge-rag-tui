"""Data models for the tool harness."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ToolRisk(str, Enum):
    READ = "read"
    WRITE = "write"
    SHELL = "shell"
    ACTION = "action"


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    rationale: str = ""


@dataclass
class PendingToolCall:
    call_id: str
    call: ToolCall
    risk: ToolRisk
    requires_approval: bool

    @staticmethod
    def create(call: ToolCall, risk: ToolRisk, *, requires_approval: bool) -> PendingToolCall:
        return PendingToolCall(
            call_id=uuid.uuid4().hex[:8],
            call=call,
            risk=risk,
            requires_approval=requires_approval,
        )


@dataclass
class ToolResult:
    call_id: str
    name: str
    success: bool
    output: str = ""
    error: str = ""
    status: str = "executed"  # proposed | auto | approved | rejected | failed

    def summary(self, max_len: int = 500) -> str:
        if self.success:
            text = self.output or "(no output)"
        else:
            text = self.error or "Tool failed"
        if len(text) > max_len:
            return text[: max_len - 3] + "..."
        return text


@dataclass
class ToolTurnResult:
    display_text: str
    auto_results: list[ToolResult]
    pending: list[PendingToolCall]
    parse_error: str | None = None

    @property
    def has_pending(self) -> bool:
        return bool(self.pending)
