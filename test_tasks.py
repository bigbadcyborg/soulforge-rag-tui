"""Smoke tests for iteration 10 Kanban task manager."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from app.core.chat_controller import ChatController
from app.core.config import load_config
from app.tasks.task_manager import (
    COLUMNS,
    TaskManager,
    normalize_column,
)
from app.tasks.task_suggester import (
    ACTION_CREATE,
    ACTION_MOVE,
    TaskSuggestion,
    _validate_suggestion,
)


def _make_config(tmp_path):
    config = MagicMock()
    config.tasks.kanban_path = str(tmp_path / "kanban.json")
    return config


def _make_controller(tmp_path, kanban_enabled: bool = True) -> ChatController:
    config = load_config()
    config.tasks.kanban_path = str(tmp_path / "kanban.json")
    config.features.kanban = kanban_enabled
    controller = ChatController(config)
    controller.task_manager = TaskManager(config)
    return controller


def test_normalize_column_aliases() -> None:
    assert normalize_column("progress") == "in_progress"
    assert normalize_column("wip") == "in_progress"
    assert normalize_column("done") == "done"
    assert normalize_column("invalid") is None


def test_create_and_list_task(tmp_path) -> None:
    mgr = TaskManager(_make_config(tmp_path))
    task = mgr.create_task("Fix CUDA script", "Update start script")
    assert task is not None
    assert task.title == "Fix CUDA script"
    assert task.description == "Update start script"

    board = mgr.list_board()
    assert len(board["backlog"]) == 1
    assert board["backlog"][0].id == task.id


def test_move_task_across_columns(tmp_path) -> None:
    mgr = TaskManager(_make_config(tmp_path))
    task = mgr.create_task("Move me")
    assert task is not None

    for column in ("in_progress", "blocked", "done", "backlog"):
        assert mgr.move_task(task.id, column) is True
        located = mgr.get_task(task.id)
        assert located is not None
        assert located[0] == column


def test_move_rejects_unknown_column(tmp_path) -> None:
    mgr = TaskManager(_make_config(tmp_path))
    task = mgr.create_task("Stay put")
    assert task is not None
    assert mgr.move_task(task.id, "invalid") is False


def test_update_and_delete_task(tmp_path) -> None:
    mgr = TaskManager(_make_config(tmp_path))
    task = mgr.create_task("Old title")
    assert task is not None

    assert mgr.update_task(task.id, title="New title", description="Notes") is True
    located = mgr.get_task(task.id)
    assert located is not None
    _, updated = located
    assert updated.title == "New title"
    assert updated.description == "Notes"

    assert mgr.delete_task(task.id) is True
    assert mgr.get_task(task.id) is None


def test_get_task_prefix_match(tmp_path) -> None:
    mgr = TaskManager(_make_config(tmp_path))
    task = mgr.create_task("Prefix test")
    assert task is not None
    assert mgr.get_task(task.id[:4]) is not None


def test_corrupt_file_recovery(tmp_path) -> None:
    path = tmp_path / "kanban.json"
    path.write_text("{not valid json", encoding="utf-8")
    mgr = TaskManager(_make_config(tmp_path))
    board = mgr.list_board()
    for column in COLUMNS:
        assert board[column] == []


def test_format_board_summary_truncates(tmp_path) -> None:
    mgr = TaskManager(_make_config(tmp_path))
    for index in range(8):
        mgr.create_task(f"Task {index}")
    summary = mgr.format_board_summary(max_tasks_per_column=3)
    assert "... and 5 more" in summary


def test_empty_title_rejected(tmp_path) -> None:
    mgr = TaskManager(_make_config(tmp_path))
    assert mgr.create_task("   ") is None


def test_persisted_json_structure(tmp_path) -> None:
    mgr = TaskManager(_make_config(tmp_path))
    mgr.create_task("Persisted")
    path = tmp_path / "kanban.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert len(data["tasks"]["backlog"]) == 1


def test_validate_create_suggestion(tmp_path) -> None:
    mgr = TaskManager(_make_config(tmp_path))
    suggestion = _validate_suggestion(
        {
            "action": "create",
            "title": "Write docs",
            "description": "Update README",
            "target_column": "backlog",
            "rationale": "User asked to document setup",
        },
        mgr,
    )
    assert suggestion is not None
    assert suggestion.action == ACTION_CREATE
    assert suggestion.title == "Write docs"


def test_validate_move_requires_existing_task(tmp_path) -> None:
    mgr = TaskManager(_make_config(tmp_path))
    task = mgr.create_task("Existing")
    assert task is not None
    suggestion = _validate_suggestion(
        {
            "action": "move",
            "task_id": task.id,
            "target_column": "done",
            "rationale": "Completed",
        },
        mgr,
    )
    assert suggestion is not None
    assert suggestion.action == ACTION_MOVE
    assert _validate_suggestion(
        {"action": "move", "task_id": "missing", "target_column": "done", "rationale": "x"},
        mgr,
    ) is None


def test_controller_create_and_move(tmp_path) -> None:
    controller = _make_controller(tmp_path)
    outcome = controller.create_task_direct("Ship Kanban")
    assert outcome.success is True
    board = controller.task_manager.list_board()
    task = board["backlog"][0]
    move_outcome = controller.move_task_direct(task.id, "in_progress")
    assert move_outcome.success is True
    located = controller.task_manager.get_task(task.id)
    assert located is not None
    assert located[0] == "in_progress"


def test_controller_disabled_kanban(tmp_path) -> None:
    controller = _make_controller(tmp_path, kanban_enabled=False)
    outcome = controller.create_task_direct("Nope")
    assert outcome.success is False
    assert "disabled" in outcome.message.lower()


def test_accept_create_suggestion(tmp_path) -> None:
    controller = _make_controller(tmp_path)
    suggestion = TaskSuggestion(
        suggestion_id="sugg001",
        action=ACTION_CREATE,
        task_id="",
        title="From suggestion",
        description="",
        target_column="backlog",
        rationale="Test",
    )
    controller.pending_task_suggestions = [suggestion]
    outcome = controller.accept_task_suggestion("sugg001")
    assert outcome.success is True
    assert len(controller.task_manager.list_board()["backlog"]) == 1


def test_prune_stale_move_suggestion(tmp_path) -> None:
    controller = _make_controller(tmp_path)
    suggestion = TaskSuggestion(
        suggestion_id="sugg002",
        action=ACTION_MOVE,
        task_id="deadbeef",
        title="",
        description="",
        target_column="done",
        rationale="Gone",
    )
    controller.pending_task_suggestions = [suggestion]
    visible = controller._visible_task_suggestions()
    assert visible == []


@patch("app.core.chat_controller.generate_suggestions")
def test_run_task_suggest_stores_results(mock_generate, tmp_path) -> None:
    controller = _make_controller(tmp_path)
    mock_generate.return_value = [
        TaskSuggestion(
            suggestion_id="abc",
            action=ACTION_CREATE,
            task_id="",
            title="Suggested",
            description="",
            target_column="backlog",
            rationale="Because",
        )
    ]
    result = controller.run_task_suggest()
    assert result.has_suggestions is True
    assert len(controller.pending_task_suggestions) == 1

