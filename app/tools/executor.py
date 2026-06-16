"""Execute tool calls with permission checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from app.core.config import AppConfig
from app.memory.memory_reviewer import MemorySuggestion
from app.rag.retriever import Retriever
from app.skills.skill_crystallizer import SkillSuggestion
from app.tasks.task_manager import TaskManager
from app.tools.handlers import bridge, fs, shell
from app.tools.models import PendingToolCall, ToolCall, ToolResult
from app.tools.permissions import is_tool_available, requires_approval, tool_risk
from app.tools.tool_log import log_tool_event


@dataclass
class ToolExecutionContext:
    config: AppConfig
    task_manager: TaskManager
    turn_count: int = 0
    retriever: Retriever | None = None
    on_memory_suggestion: Callable[[MemorySuggestion], None] | None = None
    on_skill_suggestion: Callable[[SkillSuggestion], None] | None = None


class ToolExecutor:
    """Runs tool calls; never raises to callers."""

    def __init__(self, context: ToolExecutionContext) -> None:
        self.context = context

    def classify(self, call: ToolCall) -> PendingToolCall:
        risk = tool_risk(call.name)
        needs_approval = requires_approval(self.context.config, call.name)
        return PendingToolCall.create(call, risk, requires_approval=needs_approval)

    def execute(self, pending: PendingToolCall) -> ToolResult:
        call = pending.call
        config = self.context.config
        if not config.features.tools:
            return ToolResult(
                call_id=pending.call_id,
                name=call.name,
                success=False,
                error="Tools feature is disabled",
                status="failed",
            )
        if not is_tool_available(config, call.name):
            return ToolResult(
                call_id=pending.call_id,
                name=call.name,
                success=False,
                error=f"Tool unavailable: {call.name}",
                status="failed",
            )
        try:
            output = self._dispatch(call)
            log_tool_event(
                "executed",
                call_id=pending.call_id,
                name=call.name,
                args=call.args,
                success=True,
                detail=output,
            )
            return ToolResult(
                call_id=pending.call_id,
                name=call.name,
                success=True,
                output=output,
                status="approved" if pending.requires_approval else "auto",
            )
        except Exception as error:  # noqa: BLE001
            message = str(error)
            log_tool_event(
                "failed",
                call_id=pending.call_id,
                name=call.name,
                args=call.args,
                success=False,
                detail=message,
            )
            return ToolResult(
                call_id=pending.call_id,
                name=call.name,
                success=False,
                error=message,
                status="failed",
            )

    def _dispatch(self, call: ToolCall) -> str:
        config = self.context.config
        name = call.name
        args = call.args

        if name == "read_file":
            return fs.read_file(config, args)
        if name == "list_dir":
            return fs.list_dir(config, args)
        if name == "write_file":
            return fs.write_file(config, args)
        if name == "run_command":
            return shell.run_command(config, args)
        if name == "search_docs":
            return self._search_docs(args)
        if name == "create_task":
            return bridge.create_task(
                config, args, self.context.task_manager
            )
        if name == "update_memory":
            if self.context.on_memory_suggestion is None:
                raise RuntimeError("Memory bridge not configured")
            return bridge.update_memory(
                args,
                turn_count=self.context.turn_count,
                on_suggestion=self.context.on_memory_suggestion,
            )
        if name == "create_skill":
            if self.context.on_skill_suggestion is None:
                raise RuntimeError("Skill bridge not configured")
            return bridge.create_skill(
                args, on_suggestion=self.context.on_skill_suggestion
            )
        raise ValueError(f"Unknown tool: {name}")

    def _search_docs(self, args: dict[str, Any]) -> str:
        query = str(args.get("query", "")).strip()
        if not query:
            raise ValueError("search_docs requires query")
        retriever = self.context.retriever
        if retriever is None:
            raise RuntimeError("RAG retriever not available")
        chunks = retriever.retrieve(query)
        if not chunks:
            return "No matching documents found."
        lines = []
        for chunk in chunks[:5]:
            lines.append(
                f"[{chunk.source} chunk {chunk.chunk_index}]\n{chunk.document[:500]}"
            )
        return "\n\n---\n\n".join(lines)
