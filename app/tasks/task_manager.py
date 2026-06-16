"""Task Manager: CRUD and column moves for the local Kanban board."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import date
from typing import Any

from app.core.config import AppConfig, resolve_path

COLUMNS: tuple[str, ...] = ("backlog", "in_progress", "blocked", "done")

COLUMN_LABELS: dict[str, str] = {
    "backlog": "Backlog",
    "in_progress": "In Progress",
    "blocked": "Blocked",
    "done": "Done",
}

COLUMN_ALIASES: dict[str, str] = {
    "backlog": "backlog",
    "todo": "backlog",
    "in_progress": "in_progress",
    "inprogress": "in_progress",
    "progress": "in_progress",
    "wip": "in_progress",
    "working": "in_progress",
    "blocked": "blocked",
    "done": "done",
    "complete": "done",
    "completed": "done",
}


def normalize_column(name: str) -> str | None:
    """Map user-facing column names to canonical keys."""
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    return COLUMN_ALIASES.get(key)


@dataclass
class Task:
    id: str
    title: str
    description: str = ""
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Task:
        return cls(
            id=str(data.get("id", "")),
            title=str(data.get("title", "")),
            description=str(data.get("description", "")),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
        )


def _empty_board() -> dict[str, Any]:
    return {
        "version": 1,
        "tasks": {column: [] for column in COLUMNS},
    }


class TaskManager:
    """Manages tasks stored in kanban.json."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.kanban_path = resolve_path(config.tasks.kanban_path)
        self.ensure_file()

    def ensure_file(self) -> None:
        """Create an empty board file if missing."""
        self.kanban_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.kanban_path.exists():
            self._save_board(_empty_board())

    def _load_board(self) -> dict[str, Any]:
        if not self.kanban_path.exists():
            return _empty_board()
        try:
            with open(self.kanban_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError):
            return _empty_board()

        if not isinstance(data, dict):
            return _empty_board()

        tasks_section = data.get("tasks")
        if not isinstance(tasks_section, dict):
            return _empty_board()

        normalized = _empty_board()
        normalized["version"] = data.get("version", 1)
        for column in COLUMNS:
            column_tasks = tasks_section.get(column, [])
            if not isinstance(column_tasks, list):
                column_tasks = []
            normalized["tasks"][column] = [
                task for task in column_tasks if isinstance(task, dict)
            ]
        return normalized

    def _save_board(self, board: dict[str, Any]) -> None:
        self.kanban_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.kanban_path, "w", encoding="utf-8") as handle:
            json.dump(board, handle, indent=2)

    def list_board(self) -> dict[str, list[Task]]:
        board = self._load_board()
        return {
            column: [Task.from_dict(item) for item in board["tasks"].get(column, [])]
            for column in COLUMNS
        }

    def get_task(self, task_id: str) -> tuple[str, Task] | None:
        needle = task_id.strip().lower()
        for column, tasks in self.list_board().items():
            for task in tasks:
                if task.id.lower() == needle or task.id.lower().startswith(needle):
                    return column, task
        return None

    def create_task(
        self,
        title: str,
        description: str = "",
        column: str = "backlog",
    ) -> Task | None:
        title = title.strip()
        if not title:
            return None

        target = normalize_column(column) or column
        if target not in COLUMNS:
            return None

        today = date.today().isoformat()
        task = Task(
            id=uuid.uuid4().hex[:8],
            title=title,
            description=description.strip(),
            created_at=today,
            updated_at=today,
        )

        board = self._load_board()
        board["tasks"][target].append(task.to_dict())
        self._save_board(board)
        return task

    def move_task(self, task_id: str, to_column: str) -> bool:
        target = normalize_column(to_column) or to_column
        if target not in COLUMNS:
            return False

        located = self.get_task(task_id)
        if located is None:
            return False

        from_column, task = located
        if from_column == target:
            return True

        board = self._load_board()
        source_tasks = board["tasks"][from_column]
        board["tasks"][from_column] = [
            item for item in source_tasks if item.get("id") != task.id
        ]

        task.updated_at = date.today().isoformat()
        board["tasks"][target].append(task.to_dict())
        self._save_board(board)
        return True

    def update_task(
        self,
        task_id: str,
        *,
        title: str | None = None,
        description: str | None = None,
    ) -> bool:
        located = self.get_task(task_id)
        if located is None:
            return False

        column, task = located
        if title is not None:
            title = title.strip()
            if not title:
                return False
            task.title = title
        if description is not None:
            task.description = description.strip()
        task.updated_at = date.today().isoformat()

        board = self._load_board()
        board["tasks"][column] = [
            task.to_dict() if item.get("id") == task.id else item
            for item in board["tasks"][column]
        ]
        self._save_board(board)
        return True

    def delete_task(self, task_id: str) -> bool:
        located = self.get_task(task_id)
        if located is None:
            return False

        column, task = located
        board = self._load_board()
        board["tasks"][column] = [
            item for item in board["tasks"][column] if item.get("id") != task.id
        ]
        self._save_board(board)
        return True

    def format_board_summary(self, max_tasks_per_column: int = 5) -> str:
        """Compact text summary for prompts and CLI."""
        board = self.list_board()
        lines: list[str] = []
        for column in COLUMNS:
            tasks = board[column]
            label = COLUMN_LABELS[column]
            if not tasks:
                lines.append(f"{label}: (empty)")
                continue
            shown = tasks[:max_tasks_per_column]
            task_lines = [f"  [{task.id}] {task.title}" for task in shown]
            extra = len(tasks) - len(shown)
            if extra > 0:
                task_lines.append(f"  ... and {extra} more")
            lines.append(f"{label} ({len(tasks)}):")
            lines.extend(task_lines)
        return "\n".join(lines)

    def format_board_view(self) -> str:
        """Detailed board listing for /tasks CLI output."""
        board = self.list_board()
        parts: list[str] = ["Kanban Board", "=" * 40]
        total = sum(len(tasks) for tasks in board.values())
        parts.append(f"Total tasks: {total}\n")

        for column in COLUMNS:
            tasks = board[column]
            parts.append(f"{COLUMN_LABELS[column]} ({len(tasks)})")
            parts.append("-" * 30)
            if not tasks:
                parts.append("  (empty)")
            else:
                for task in tasks:
                    desc = f" — {task.description}" if task.description else ""
                    parts.append(f"  [{task.id}] {task.title}{desc}")
            parts.append("")
        return "\n".join(parts).rstrip()
