"""Permission checks and path sandboxing for tools."""

from __future__ import annotations

import os
import shlex
from pathlib import Path

from app.core.config import PROJECT_ROOT, AppConfig
from app.tools.models import ToolRisk


def resolve_sandbox_path(raw: str) -> Path:
    """Resolve a user-supplied path under the project root."""
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    resolved = path.resolve()
    if not str(resolved).startswith(str(PROJECT_ROOT.resolve())):
        raise PermissionError(f"Path escapes project root: {raw}")
    return resolved


def _is_under(path: Path, roots: list[Path]) -> bool:
    resolved = path.resolve()
    for root in roots:
        root_resolved = root.resolve()
        try:
            resolved.relative_to(root_resolved)
            return True
        except ValueError:
            continue
    return False


def check_read_path(config: AppConfig, raw: str) -> Path:
    path = resolve_sandbox_path(raw)
    if not _is_under(path, config.tools.read_root_paths):
        raise PermissionError(f"Read not allowed outside readRoots: {path}")
    return path


def check_write_path(config: AppConfig, raw: str) -> Path:
    if not config.tools.allow_write:
        raise PermissionError("write_file disabled (tools.allowWrite: false)")
    path = resolve_sandbox_path(raw)
    if not _is_under(path, config.tools.write_root_paths):
        raise PermissionError(f"Write not allowed outside writeRoots: {path}")
    protected = {PROJECT_ROOT / "config.yaml", PROJECT_ROOT / "models"}
    if path in protected or str(path).startswith(str((PROJECT_ROOT / "models").resolve())):
        raise PermissionError(f"Protected path cannot be written: {path}")
    return path


def check_shell_command(config: AppConfig, command: str) -> list[str]:
    if not config.tools.allow_shell:
        raise PermissionError("run_command disabled (tools.allowShell: false)")
    command = command.strip()
    if not command:
        raise PermissionError("Empty command")
    allowlist = config.tools.shell_allowlist
    if not allowlist:
        raise PermissionError("shellAllowlist is empty — no commands permitted")
    if not any(command.startswith(prefix) for prefix in allowlist):
        raise PermissionError(f"Command not on shellAllowlist: {command}")
    return shlex.split(command)


def tool_risk(name: str) -> ToolRisk:
    mapping = {
        "read_file": ToolRisk.READ,
        "list_dir": ToolRisk.READ,
        "search_docs": ToolRisk.READ,
        "write_file": ToolRisk.WRITE,
        "run_command": ToolRisk.SHELL,
        "create_task": ToolRisk.ACTION,
        "update_memory": ToolRisk.ACTION,
        "create_skill": ToolRisk.ACTION,
    }
    return mapping.get(name, ToolRisk.ACTION)


def is_tool_available(config: AppConfig, name: str) -> bool:
    if not config.features.tools:
        return False
    if name == "search_docs" and not config.features.rag:
        return False
    if name == "create_task" and not config.features.kanban:
        return False
    if name == "update_memory" and not config.features.memory:
        return False
    if name == "create_skill" and not config.features.skills:
        return False
    if name == "write_file" and not config.tools.allow_write:
        return False
    if name == "run_command" and not config.tools.allow_shell:
        return False
    return name in {
        "read_file",
        "list_dir",
        "search_docs",
        "write_file",
        "run_command",
        "create_task",
        "update_memory",
        "create_skill",
    }


def requires_approval(config: AppConfig, name: str) -> bool:
    risk = tool_risk(name)
    if risk == ToolRisk.READ:
        return not config.tools.auto_approve_read_only
    return True
