"""Bridge tools that delegate to existing SoulForge managers."""

from __future__ import annotations

from typing import Callable

from app.core.config import AppConfig
from app.memory.memory_reviewer import MemorySuggestion
from app.skills.skill_crystallizer import SkillSuggestion, render_skill_markdown
from app.tasks.task_manager import TaskManager


def create_task(config: AppConfig, args: dict, task_manager: TaskManager) -> str:
    title = str(args.get("title", "")).strip()
    if not title:
        raise ValueError("create_task requires title")
    description = str(args.get("description", "")).strip()
    column = str(args.get("column", "backlog")).strip() or "backlog"
    task = task_manager.create_task(title, description=description, column=column)
    if task is None:
        raise ValueError("Failed to create task")
    return f"Created task {task.id}: {task.title} in {column}"


def update_memory(
    args: dict,
    *,
    turn_count: int,
    on_suggestion: Callable[[MemorySuggestion], None],
) -> str:
    section = str(args.get("section", "user")).strip().lower()
    if section not in ("user", "memory"):
        section = "user"
    proposed = str(args.get("proposed_content", "")).strip()
    if not proposed:
        raise ValueError("update_memory requires proposed_content")
    rationale = str(args.get("rationale", "Tool-proposed memory update")).strip()
    suggestion = MemorySuggestion(
        section=section,
        rationale=rationale,
        proposed_content=proposed,
        turn_count=turn_count,
    )
    on_suggestion(suggestion)
    return (
        f"Memory suggestion queued for {section}.md — "
        "run /memory-review and /memory-accept to save."
    )


def create_skill(
    args: dict,
    *,
    on_suggestion: Callable[[SkillSuggestion], None],
) -> str:
    name = str(args.get("name", "")).strip()
    description = str(args.get("description", "")).strip()
    rationale = str(args.get("rationale", "Tool-proposed skill")).strip()
    trigger = str(args.get("trigger", "")).strip()
    procedure = str(args.get("procedure", "")).strip()
    validation = str(args.get("validation", "")).strip()
    content = str(args.get("content", "")).strip()
    if content:
        proposed_content = content
    else:
        if not all((trigger, procedure, validation)):
            raise ValueError(
                "create_skill requires content or trigger/procedure/validation"
            )
        proposed_content = render_skill_markdown(
            {
                "trigger": trigger,
                "procedure": procedure,
                "validation": validation,
            }
        )
    if not name:
        raise ValueError("create_skill requires name")
    suggestion = SkillSuggestion(
        name=name,
        description=description or name.replace("_", " "),
        rationale=rationale,
        proposed_content=proposed_content,
        fingerprint="tool",
        success_count=0,
    )
    on_suggestion(suggestion)
    return (
        f"Skill draft queued: {name} — "
        "run /skill-accept or review in TUI to save."
    )
