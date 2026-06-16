"""Skill crystallizer: LLM-assisted skill suggestion from successful workflows."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from typing import Any

from app.core.config import AppConfig
from app.core.model_runtime import ModelRuntime
from app.memory.memory_reviewer import format_conversation_window
from app.skills.workflow_observer import WorkflowEntry

CRYSTALLIZE_COMPLETION_OVERRIDES = {
    "temperature": 0.2,
    "max_tokens": 1536,
    "stop": ["</s>"],
}

CRYSTALLIZE_RETRY_PROMPT = (
    "Your previous response was invalid or incomplete. Reply with ONLY valid JSON.\n"
    'Required: {"name": "snake_case", "description": "...", "rationale": "...", '
    '"trigger": "...", "procedure": "...", "validation": "..."}\n'
    "Do not invent steps not present in USER STATEMENTS."
)

REQUIRED_SECTIONS = ("## Trigger", "## Procedure", "## Validation")


@dataclass
class SkillSuggestion:
    name: str
    description: str
    rationale: str
    proposed_content: str
    fingerprint: str
    success_count: int


def validate_skill_content(content: str) -> list[str]:
    """Return validation errors for skill markdown body."""
    errors: list[str] = []
    for section in REQUIRED_SECTIONS:
        if section.lower() not in content.lower():
            errors.append(f"Missing section: {section}")
    return errors


def _slugify_name(raw: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", raw.strip().lower()).strip("_")
    return slug or "workflow_skill"


def render_skill_markdown(
    data: dict[str, Any],
    success_count: int = 0,
    tags: list[str] | None = None,
) -> str:
    """Assemble SKILL-001 markdown from crystallizer fields."""
    name = _slugify_name(str(data.get("name", "workflow_skill")))
    description = str(data.get("description", "")).strip()
    tag_list = tags or list(data.get("tags") or [])
    trigger = str(data.get("trigger", "")).strip()
    procedure = str(data.get("procedure", "")).strip()
    validation = str(data.get("validation", "")).strip()
    today = date.today().isoformat()

    body = (
        f"---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        f"tags: {tag_list}\n"
        f"status: active\n"
        f"successCount: {success_count}\n"
        f"lastUsed: {today}\n"
        f"---\n\n"
        f"## Trigger\n\n{trigger}\n\n"
        f"## Procedure\n\n{procedure}\n\n"
        f"## Validation\n\n{validation}\n"
    )
    return body


def build_crystallize_prompt(
    user_statements: list[str],
    existing_skills: list[str],
    summary_hint: str = "",
) -> str:
    """Build the system prompt for skill crystallization."""
    statements_block = (
        "\n".join(f"- {text}" for text in user_statements)
        if user_statements
        else "(no user statements)"
    )
    skills_block = (
        "\n".join(f"- {name}" for name in existing_skills)
        if existing_skills
        else "(none)"
    )
    hint_block = summary_hint.strip() or "(none)"

    return (
        "You are a skill crystallizer for a local chatbot. Turn repeated successful "
        "workflows into reusable skill files.\n\n"
        "USER STATEMENTS (ONLY source for steps — do not invent anything else):\n"
        f"{statements_block}\n\n"
        f"Summary hint from user: {hint_block}\n\n"
        "Existing skill names (avoid duplicates):\n"
        f"{skills_block}\n\n"
        "Respond with ONLY valid JSON:\n"
        '{"name": "snake_case_slug", "description": "one line summary", '
        '"rationale": "why this should be a skill", '
        '"trigger": "when to use this skill", '
        '"procedure": "numbered steps from user statements", '
        '"validation": "how to verify success", '
        '"tags": ["tag1", "tag2"]}\n\n'
        "Rules:\n"
        "- name must be lowercase snake_case.\n"
        "- procedure must use numbered steps derived from USER STATEMENTS.\n"
        "- Never invent people, tools, or steps not implied by USER STATEMENTS.\n"
        "- Ignore assistant messages entirely.\n"
        "- Skip questions; only crystallize declarative workflow content."
    )


def _parse_crystallize_response(raw: str) -> dict[str, Any] | None:
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

    return data if isinstance(data, dict) else None


def _build_local_fallback(
    user_statements: list[str],
    summary_hint: str,
    success_count: int,
) -> SkillSuggestion:
    """Template fallback when the model response is unusable."""
    summary = summary_hint.strip() or (
        user_statements[0][:80] if user_statements else "Repeated workflow"
    )
    name = _slugify_name(summary)
    procedure_lines = [
        f"{index}. {statement}"
        for index, statement in enumerate(user_statements, start=1)
        if statement.strip()
    ]
    if not procedure_lines:
        procedure_lines = ["1. Follow the workflow described in recent user messages."]

    data = {
        "name": name,
        "description": summary[:120],
        "rationale": "Repeated successful workflow marked by the user.",
        "trigger": summary,
        "procedure": "\n".join(procedure_lines),
        "validation": "User confirms the workflow completed successfully.",
        "tags": [],
    }
    content = render_skill_markdown(data, success_count=success_count)
    return SkillSuggestion(
        name=_slugify_name(str(data["name"])),
        description=str(data["description"]),
        rationale=str(data["rationale"]),
        proposed_content=content,
        fingerprint="",
        success_count=success_count,
    )


def finalize_proposed_skill(
    content: str,
    user_statements: list[str],
    success_count: int,
) -> str:
    """Ensure required sections exist; rebuild from statements if needed."""
    errors = validate_skill_content(content)
    if not errors:
        return content.strip()

    fallback = _build_local_fallback(user_statements, "", success_count)
    return fallback.proposed_content


def _interpret_crystallize_data(
    data: dict[str, Any] | None,
    workflow: WorkflowEntry,
    user_statements: list[str],
    raw: str,
) -> tuple[SkillSuggestion | None, str | None]:
    if not data:
        preview = raw[:200].replace("\n", " ")
        message = f"Could not parse crystallize response: {preview!r}"
        print(f"[skills] {message}")
        return None, message

    name = _slugify_name(str(data.get("name", workflow.summary or "workflow_skill")))
    description = str(data.get("description", workflow.summary)).strip()
    rationale = str(data.get("rationale", "")).strip()
    if not rationale:
        rationale = "Repeated successful workflow marked by the user."

    content = render_skill_markdown(
        {
            "name": name,
            "description": description,
            "trigger": data.get("trigger", workflow.summary),
            "procedure": data.get("procedure", ""),
            "validation": data.get("validation", ""),
            "tags": data.get("tags", []),
        },
        success_count=workflow.success_count,
    )
    content = finalize_proposed_skill(
        content,
        user_statements,
        workflow.success_count,
    )

    return (
        SkillSuggestion(
            name=name,
            description=description,
            rationale=rationale,
            proposed_content=content,
            fingerprint=workflow.fingerprint,
            success_count=workflow.success_count,
        ),
        None,
    )


def _run_crystallize_completion(
    runtime: ModelRuntime,
    messages: list[dict[str, str]],
) -> str:
    response = runtime.create_chat_completion(
        messages,
        stream=False,
        **CRYSTALLIZE_COMPLETION_OVERRIDES,
    )
    return response["choices"][0]["message"]["content"].strip()


def generate_suggestion(
    runtime: ModelRuntime,
    config: AppConfig,
    chat_messages: list[dict[str, str]],
    workflow: WorkflowEntry,
    existing_skill_names: list[str],
) -> tuple[SkillSuggestion | None, str | None]:
    """Run a one-shot crystallization. Returns (suggestion, error_message)."""
    user_statements = workflow.user_statements
    if not user_statements:
        window = config.skills.success_window_turns
        conversation = format_conversation_window(chat_messages, last_n_turns=window)
        user_statements = [
            line[6:].strip()
            for line in conversation.splitlines()
            if line.startswith("User: ")
        ]

    system_prompt = build_crystallize_prompt(
        user_statements,
        existing_skill_names,
        workflow.summary,
    )
    review_messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": "Crystallize this workflow into skill JSON only.",
        },
    ]

    try:
        raw = _run_crystallize_completion(runtime, review_messages)
    except Exception as error:  # noqa: BLE001
        message = f"Skill crystallization failed: {error}"
        print(f"[skills] {message}")
        return None, message

    suggestion, error = _interpret_crystallize_data(
        _parse_crystallize_response(raw),
        workflow,
        user_statements,
        raw,
    )
    if suggestion is not None:
        suggestion.fingerprint = workflow.fingerprint
        return suggestion, None

    if error is not None:
        print(f"[skills] Crystallize retry after: {error}")
        retry_messages = [
            *review_messages,
            {"role": "assistant", "content": raw},
            {"role": "user", "content": CRYSTALLIZE_RETRY_PROMPT},
        ]
        try:
            raw_retry = _run_crystallize_completion(runtime, retry_messages)
        except Exception as retry_error:  # noqa: BLE001
            message = f"Skill crystallize retry failed: {retry_error}"
            print(f"[skills] {message}")
            return _build_local_fallback(
                user_statements,
                workflow.summary,
                workflow.success_count,
            ), None

        suggestion, retry_err = _interpret_crystallize_data(
            _parse_crystallize_response(raw_retry),
            workflow,
            user_statements,
            raw_retry,
        )
        if suggestion is not None:
            suggestion.fingerprint = workflow.fingerprint
            return suggestion, None
        if retry_err:
            print(f"[skills] {retry_err}")

    fallback = _build_local_fallback(
        user_statements,
        workflow.summary,
        workflow.success_count,
    )
    fallback.fingerprint = workflow.fingerprint
    return fallback, None


def format_suggestion_view(suggestion: SkillSuggestion) -> str:
    """Format a pending skill suggestion for display."""
    lines = [
        f"Skill crystallization ({suggestion.success_count} successes)",
        f"Target: {suggestion.name}.md",
        "",
        "Rationale:",
        suggestion.rationale,
        "",
        "Proposed content:",
        suggestion.proposed_content or "(empty)",
        "",
        "Use /skill-accept to save, /skill-reject to discard, or Edit in TUI.",
    ]
    return "\n".join(lines)


def resolve_unique_name(base_name: str, existing_names: set[str]) -> str:
    """Return base_name or base_name-N if a collision exists."""
    if base_name not in existing_names:
        return base_name
    index = 2
    while f"{base_name}-{index}" in existing_names:
        index += 1
    return f"{base_name}-{index}"
