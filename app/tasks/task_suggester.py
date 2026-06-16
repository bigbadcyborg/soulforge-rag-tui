"""Task suggester: LLM-assisted Kanban update suggestions from conversation."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any

from app.core.config import AppConfig
from app.core.model_runtime import ModelRuntime
from app.memory.memory_reviewer import format_conversation_window
from app.tasks.task_manager import COLUMNS, COLUMN_LABELS, TaskManager, normalize_column

ACTION_CREATE = "create"
ACTION_MOVE = "move"
ACTION_UPDATE = "update"
ACTION_DELETE = "delete"

VALID_ACTIONS = {ACTION_CREATE, ACTION_MOVE, ACTION_UPDATE, ACTION_DELETE}

SUGGEST_COMPLETION_OVERRIDES = {
    "temperature": 0.2,
    "max_tokens": 1024,
    "stop": ["</s>"],
}

SUGGEST_RETRY_PROMPT = (
    "Your previous response was invalid. Reply with ONLY valid JSON:\n"
    '{"suggestions": [{"action": "create|move|update|delete", '
    '"task_id": "optional for create", "title": "...", "description": "...", '
    '"target_column": "backlog|in_progress|blocked|done", "rationale": "..."}]}\n'
    "Use an empty suggestions array if nothing should change."
)


@dataclass
class TaskSuggestion:
    suggestion_id: str
    action: str
    task_id: str
    title: str
    description: str
    target_column: str
    rationale: str


def format_suggestion_view(suggestion: TaskSuggestion, index: int, total: int) -> str:
    """Human-readable preview for approval modal."""
    column_label = COLUMN_LABELS.get(suggestion.target_column, suggestion.target_column)
    lines = [
        f"Action: {suggestion.action.upper()}",
        f"Rationale: {suggestion.rationale}",
    ]
    if suggestion.task_id:
        lines.append(f"Task ID: {suggestion.task_id}")
    if suggestion.title:
        lines.append(f"Title: {suggestion.title}")
    if suggestion.description:
        lines.append(f"Description: {suggestion.description}")
    if suggestion.target_column:
        lines.append(f"Target column: {column_label}")
    lines.append(f"\n({index + 1} of {total})")
    return "\n".join(lines)


def format_suggestions_review(suggestions: list[TaskSuggestion]) -> str:
    if not suggestions:
        return "No pending task suggestions."
    parts = [f"Pending task suggestions ({len(suggestions)}):", ""]
    for index, suggestion in enumerate(suggestions):
        parts.append(format_suggestion_view(suggestion, index, len(suggestions)))
        parts.append("")
    parts.append("Use /task-suggest in TUI or approve individually.")
    return "\n".join(parts).strip()


def _build_suggest_prompt(board_summary: str) -> str:
    columns = ", ".join(f"{key} ({COLUMN_LABELS[key]})" for key in COLUMNS)
    return (
        "You are a task board assistant for a local chatbot. Review the conversation "
        "and current Kanban board. Suggest concrete board updates ONLY when the user "
        "clearly discussed work items, progress, blockers, or completions.\n\n"
        f"Valid columns: {columns}\n\n"
        "CURRENT BOARD:\n"
        f"{board_summary}\n\n"
        "Respond with ONLY valid JSON:\n"
        '{"suggestions": ['
        '{"action": "create", "title": "...", "description": "...", '
        '"target_column": "backlog", "rationale": "..."}, '
        '{"action": "move", "task_id": "abc12345", "target_column": "done", '
        '"rationale": "..."}, '
        '{"action": "update", "task_id": "abc12345", "title": "...", '
        '"description": "...", "rationale": "..."}, '
        '{"action": "delete", "task_id": "abc12345", "rationale": "..."}'
        "]}\n\n"
        "Rules:\n"
        "- Use task_id from the board for move/update/delete.\n"
        "- Prefer create for new work; move when status clearly changed.\n"
        "- Do not duplicate existing tasks.\n"
        "- Return empty suggestions if nothing actionable was discussed.\n"
        "- Never invent tasks unrelated to USER STATEMENTS."
    )


def _parse_suggest_response(raw: str) -> list[dict[str, Any]] | None:
    text = raw.strip()
    if not text:
        return None

    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None

    if not isinstance(data, dict):
        return None
    suggestions = data.get("suggestions", [])
    if not isinstance(suggestions, list):
        return None
    return [item for item in suggestions if isinstance(item, dict)]


def _validate_suggestion(
    item: dict[str, Any],
    task_manager: TaskManager,
) -> TaskSuggestion | None:
    action = str(item.get("action", "")).strip().lower()
    if action not in VALID_ACTIONS:
        return None

    task_id = str(item.get("task_id", "")).strip()
    title = str(item.get("title", "")).strip()
    description = str(item.get("description", "")).strip()
    rationale = str(item.get("rationale", "")).strip() or "Suggested from conversation."
    raw_column = str(item.get("target_column", "backlog")).strip()
    target_column = normalize_column(raw_column) or raw_column

    if action == ACTION_CREATE:
        if not title:
            return None
        if target_column not in COLUMNS:
            target_column = "backlog"
    elif action in (ACTION_MOVE, ACTION_UPDATE, ACTION_DELETE):
        located = task_manager.get_task(task_id) if task_id else None
        if located is None:
            return None
        _, task = located
        task_id = task.id
        if action == ACTION_MOVE:
            if target_column not in COLUMNS:
                return None
        if action == ACTION_UPDATE and not title and not description:
            return None
    else:
        return None

    return TaskSuggestion(
        suggestion_id=uuid.uuid4().hex[:12],
        action=action,
        task_id=task_id,
        title=title,
        description=description,
        target_column=target_column,
        rationale=rationale,
    )


def generate_suggestions(
    runtime: ModelRuntime,
    task_manager: TaskManager,
    messages: list[dict[str, str]],
    config: AppConfig,
) -> list[TaskSuggestion]:
    """Analyze conversation and return validated task suggestions."""
    board_summary = task_manager.format_board_summary()
    window = format_conversation_window(messages, last_n_turns=12)
    system_prompt = _build_suggest_prompt(board_summary)

    llm_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"CONVERSATION:\n{window}"},
    ]

    raw = ""
    for attempt in range(2):
        try:
            response = runtime.create_chat_completion(
                llm_messages,
                stream=False,
                **SUGGEST_COMPLETION_OVERRIDES,
            )
            raw = response["choices"][0]["message"]["content"]
        except Exception as error:  # noqa: BLE001
            print(f"[tasks] Suggestion generation failed: {error}")
            return []

        parsed_items = _parse_suggest_response(raw)
        if parsed_items is not None:
            suggestions: list[TaskSuggestion] = []
            for item in parsed_items:
                suggestion = _validate_suggestion(item, task_manager)
                if suggestion is not None:
                    suggestions.append(suggestion)
            return suggestions

        llm_messages.append({"role": "assistant", "content": raw})
        llm_messages.append({"role": "user", "content": SUGGEST_RETRY_PROMPT})

    return []
