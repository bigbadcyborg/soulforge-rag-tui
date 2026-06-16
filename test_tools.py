"""Tests for iteration 13 optional tool harness."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

import pytest

from app.core.chat_controller import ChatController
from app.core.config import (
    AppConfig,
    DEFAULT_CONFIG_PATH,
    ToolsConfig,
    load_config,
    save_tools,
    tools_to_yaml_dict,
)
from app.tools.executor import ToolExecutionContext, ToolExecutor
from app.tools.handlers import fs
from app.tools.models import ToolCall, PendingToolCall, ToolRisk
from app.tools.parser import parse_tool_calls
from app.tools.permissions import (
    check_read_path,
    requires_approval,
    resolve_sandbox_path,
)
from app.tools.registry import format_tools_catalog
from app.tools.tool_log import LOG_PATH, log_tool_event, read_recent_log


def _tools_config(tmp_path: Path, **overrides) -> AppConfig:
    data = {
        "model": {
            "chatModelPath": str(tmp_path / "chat.gguf"),
            "embeddingModelPath": str(tmp_path / "embed.gguf"),
        },
        "features": {"tools": True, "rag": False, "memory": True, "kanban": True},
        "tools": {
            "allowWrite": False,
            "allowShell": False,
            "readRoots": [str(tmp_path / "docs"), str(tmp_path / "app")],
            "writeRoots": [str(tmp_path / "app" / "memory")],
            "maxReadBytes": 100,
            "shellAllowlist": ["git status"],
            "autoApproveReadOnly": True,
        },
        "memory": {
            "userFile": str(tmp_path / "app" / "memory" / "user.md"),
            "memoryFile": str(tmp_path / "app" / "memory" / "memory.md"),
            "sessionFile": str(tmp_path / "app" / "memory" / "session.md"),
        },
        "skills": {
            "activePath": str(tmp_path / "app" / "skills" / "active"),
            "archivedPath": str(tmp_path / "app" / "skills" / "archived"),
            "registryPath": str(tmp_path / "app" / "skills" / "registry.json"),
        },
        "tasks": {"kanbanPath": str(tmp_path / "app" / "tasks" / "kanban.json")},
        "sessions": {"storePath": str(tmp_path / "app" / "sessions")},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and key in data:
            data[key].update(value)
        else:
            data[key] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    (tmp_path / "docs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "app" / "memory").mkdir(parents=True, exist_ok=True)
    (tmp_path / "app" / "tasks").mkdir(parents=True, exist_ok=True)
    (tmp_path / "app" / "skills" / "active").mkdir(parents=True, exist_ok=True)
    (tmp_path / "app" / "skills" / "archived").mkdir(parents=True, exist_ok=True)
    (tmp_path / "app" / "sessions").mkdir(parents=True, exist_ok=True)
    (tmp_path / "app" / "tasks" / "kanban.json").write_text(
        json.dumps({"version": 1, "tasks": {c: [] for c in ("backlog", "in_progress", "blocked", "done")}}),
        encoding="utf-8",
    )
    return load_config(path)


def test_path_sandbox_blocks_escape(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.tools.permissions.PROJECT_ROOT", tmp_path.resolve()
    )
    with pytest.raises(PermissionError):
        check_read_path(
            _tools_config(tmp_path),
            "../../outside.txt",
        )


def test_read_file_respects_max_bytes(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("app.core.config.PROJECT_ROOT", tmp_path.resolve())
    config = _tools_config(tmp_path)
    doc = tmp_path / "docs" / "big.txt"
    doc.write_text("x" * 200, encoding="utf-8")
    with patch("app.tools.handlers.fs.check_read_path", return_value=doc):
        output = fs.read_file(config, {"path": "docs/big.txt"})
    assert "[truncated" in output


def test_write_file_blocked_when_disabled(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("app.core.config.PROJECT_ROOT", tmp_path.resolve())
    config = _tools_config(tmp_path)
    with pytest.raises(PermissionError):
        fs.write_file(config, {"path": "app/memory/x.md", "content": "hi"})


def test_run_command_blocked_without_shell(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("app.core.config.PROJECT_ROOT", tmp_path.resolve())
    from app.tools.handlers import shell

    config = _tools_config(tmp_path)
    with pytest.raises(PermissionError):
        shell.run_command(config, {"command": "git status"})


def test_parser_extracts_and_strips_tool_block() -> None:
    text = 'Hello\n```tool\n{"tools": [{"name": "read_file", "args": {"path": "docs/a.md"}}]}\n```\nDone'
    display, calls, error = parse_tool_calls(text)
    assert error is None
    assert len(calls) == 1
    assert calls[0].name == "read_file"
    assert "```tool" not in display
    assert "Hello" in display


def test_parser_invalid_json_no_crash() -> None:
    text = "```tool\n{bad json\n```"
    display, calls, error = parse_tool_calls(text)
    assert calls == []
    assert error is not None


def test_requires_approval_for_write(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("app.core.config.PROJECT_ROOT", tmp_path.resolve())
    config = _tools_config(tmp_path)
    assert requires_approval(config, "write_file") is True
    assert requires_approval(config, "read_file") is False


def test_executor_create_task_on_approve(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("app.core.config.PROJECT_ROOT", tmp_path.resolve())
    config = _tools_config(tmp_path)
    from app.tasks.task_manager import TaskManager

    mgr = TaskManager(config)
    ctx = ToolExecutionContext(config=config, task_manager=mgr)
    executor = ToolExecutor(ctx)
    pending = PendingToolCall.create(
        ToolCall(name="create_task", args={"title": "Test task"}),
        ToolRisk.ACTION,
        requires_approval=True,
    )
    result = executor.execute(pending)
    assert result.success
    board = mgr.list_board()
    assert any(
        task.title == "Test task"
        for tasks in board.values()
        for task in tasks
    )


def test_update_memory_does_not_write_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("app.core.config.PROJECT_ROOT", tmp_path.resolve())
    config = _tools_config(tmp_path)
    user_path = config.memory.user_path
    before = user_path.read_text(encoding="utf-8") if user_path.exists() else ""
    queued = []

    def on_suggestion(suggestion) -> None:
        queued.append(suggestion)

    from app.tasks.task_manager import TaskManager
    from app.tools.handlers import bridge

    bridge.update_memory(
        {"section": "user", "proposed_content": "New facts", "rationale": "test"},
        turn_count=1,
        on_suggestion=on_suggestion,
    )
    after = user_path.read_text(encoding="utf-8") if user_path.exists() else ""
    assert before == after
    assert len(queued) == 1


def test_tool_log_appends_line(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("app.tools.tool_log.LOG_PATH", tmp_path / "tool_calls.jsonl")
    log_tool_event("test", name="read_file", detail="hello")
    assert (tmp_path / "tool_calls.jsonl").exists()
    monkeypatch.setattr("app.tools.tool_log.LOG_PATH", tmp_path / "tool_calls.jsonl")
    view = read_recent_log()
    assert "read_file" in view


def test_controller_process_reply_tools_off(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("app.core.config.PROJECT_ROOT", tmp_path.resolve())
    config = _tools_config(tmp_path, features={"tools": False})
    controller = ChatController(config)
    raw = 'Hi ```tool\n{"tools": []}\n```'
    result = controller.process_assistant_reply(raw)
    assert result.display_text == raw.strip()
    assert not result.pending


def test_get_tools_status_includes_flags(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("app.core.config.PROJECT_ROOT", tmp_path.resolve())
    config = _tools_config(tmp_path)
    controller = ChatController(config)
    status = controller.get_tools_status()
    assert "allowWrite: False" in status
    assert "read_file" in status


def test_risky_tools_always_pending(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("app.core.config.PROJECT_ROOT", tmp_path.resolve())
    config = _tools_config(tmp_path, tools={"allowWrite": True, "autoApproveReadOnly": True})
    controller = ChatController(config)
    raw = (
        'Answer\n```tool\n'
        '{"tools": [{"name": "write_file", "args": {"path": "app/memory/x.md", "content": "x"}}]}\n'
        "```"
    )
    result = controller.process_assistant_reply(raw)
    assert result.pending
    assert not result.auto_results


def test_tools_to_yaml_dict_roundtrip() -> None:
    tools = ToolsConfig(
        allow_write=True,
        allow_shell=True,
        shell_allowlist=["git status"],
    )
    data = tools_to_yaml_dict(tools)
    assert data["allowWrite"] is True
    assert data["shellAllowlist"] == ["git status"]


def test_save_tools_persists_allowlist(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("app.core.config.PROJECT_ROOT", tmp_path.resolve())
    config = _tools_config(tmp_path)
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr("app.core.config.DEFAULT_CONFIG_PATH", config_path)
    config.tools.shell_allowlist.append("git status")
    save_tools(config, config_path)
    reloaded = load_config(config_path)
    assert "git status" in reloaded.tools.shell_allowlist


def test_add_shell_allowlist_dedupes(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("app.core.config.PROJECT_ROOT", tmp_path.resolve())
    config = _tools_config(tmp_path)
    config.tools.shell_allowlist.clear()
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr("app.core.config.DEFAULT_CONFIG_PATH", config_path)
    save_tools(config, config_path)
    controller = ChatController(config)
    first = controller.add_shell_allowlist_entry("git status")
    second = controller.add_shell_allowlist_entry("git status")
    assert "Added" in first
    assert "Already" in second
    assert controller.config.tools.shell_allowlist.count("git status") == 1


def test_run_tool_test_list_dir(tmp_path, monkeypatch) -> None:
    root = tmp_path.resolve()
    monkeypatch.setattr("app.core.config.PROJECT_ROOT", root)
    monkeypatch.setattr("app.tools.permissions.PROJECT_ROOT", root)
    (tmp_path / "docs").mkdir(parents=True, exist_ok=True)
    config = _tools_config(tmp_path)
    controller = ChatController(config)
    result = controller.run_tool_test("list_dir", {"path": "docs"})
    assert result.success
    assert result.status == "manual_test"


def test_run_tool_test_shell_requires_allowlist(tmp_path, monkeypatch) -> None:
    root = tmp_path.resolve()
    monkeypatch.setattr("app.core.config.PROJECT_ROOT", root)
    monkeypatch.setattr("app.tools.handlers.shell.PROJECT_ROOT", root)
    config = _tools_config(
        tmp_path,
        tools={"allowShell": True, "shellAllowlist": []},
    )
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr("app.core.config.DEFAULT_CONFIG_PATH", config_path)
    controller = ChatController(config)
    fail = controller.run_tool_test("run_command", {"command": "git status"})
    assert not fail.success
    controller.add_shell_allowlist_entry("git status")
    ok = controller.run_tool_test("run_command", {"command": "git status"})
    assert ok.success


def test_remove_shell_allowlist_entry(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("app.core.config.PROJECT_ROOT", tmp_path.resolve())
    config = _tools_config(
        tmp_path,
        tools={"allowShell": True, "shellAllowlist": ["git status"]},
    )
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr("app.core.config.DEFAULT_CONFIG_PATH", config_path)
    controller = ChatController(config)
    message = controller.remove_shell_allowlist_entry("git status")
    assert "Removed" in message
    assert "git status" not in controller.config.tools.shell_allowlist


def test_get_tools_menu_data(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("app.core.config.PROJECT_ROOT", tmp_path.resolve())
    config = _tools_config(tmp_path)
    controller = ChatController(config)
    data = controller.get_tools_menu_data()
    assert data["tools_enabled"] is True
    assert any(t["name"] == "read_file" for t in data["tool_defs"])
