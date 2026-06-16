"""Tool registry: definitions and catalog."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app.core.config import AppConfig
from app.tools.models import ToolRisk
from app.tools.permissions import is_tool_available, tool_risk

ToolHandler = Callable[..., str]

TOOL_DESCRIPTIONS: dict[str, str] = {
    "read_file": "Read a text file under readRoots (args: path)",
    "list_dir": "List directory entries under readRoots (args: path)",
    "search_docs": "Search indexed docs via RAG (args: query)",
    "write_file": "Write a file under writeRoots — requires approval (args: path, content)",
    "run_command": "Run allowlisted shell command — requires approval (args: command)",
    "create_task": "Create Kanban task — requires approval (args: title, description?, column?)",
    "update_memory": "Queue memory update suggestion — requires approval (args: section, proposed_content, rationale?)",
    "create_skill": "Queue skill draft — requires approval (args: name, content or trigger/procedure/validation)",
}

TOOL_EXAMPLE_ARGS: dict[str, str] = {
    "read_file": '{"path": "docs/example.md"}',
    "list_dir": '{"path": "docs"}',
    "search_docs": '{"query": "search terms"}',
    "write_file": '{"path": "app/memory/note.md", "content": "hello"}',
    "run_command": '{"command": "git status"}',
    "create_task": '{"title": "My task", "description": "", "column": "backlog"}',
    "update_memory": '{"section": "user", "proposed_content": "facts", "rationale": "test"}',
    "create_skill": '{"name": "my_skill", "trigger": "...", "procedure": "...", "validation": "..."}',
}


@dataclass(frozen=True)
class ToolDef:
    name: str
    risk: ToolRisk
    description: str


def list_tool_defs(config: AppConfig) -> list[ToolDef]:
    defs: list[ToolDef] = []
    for name, description in TOOL_DESCRIPTIONS.items():
        if is_tool_available(config, name) or name in (
            "read_file",
            "list_dir",
            "write_file",
            "run_command",
            "search_docs",
            "create_task",
            "update_memory",
            "create_skill",
        ):
            defs.append(
                ToolDef(name=name, risk=tool_risk(name), description=description)
            )
    return defs


def format_tools_catalog(config: AppConfig) -> str:
    lines = [
        f"Tools feature: {'on' if config.features.tools else 'off'}",
        f"allowWrite: {config.tools.allow_write}",
        f"allowShell: {config.tools.allow_shell}",
        f"autoApproveReadOnly: {config.tools.auto_approve_read_only}",
        "",
        "Registered tools:",
    ]
    for tool_def in list_tool_defs(config):
        available = is_tool_available(config, tool_def.name)
        flag = "available" if available else "unavailable"
        lines.append(
            f"  {tool_def.name} [{tool_def.risk.value}] ({flag}) — {tool_def.description}"
        )
    return "\n".join(lines)
