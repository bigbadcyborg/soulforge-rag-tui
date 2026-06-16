"""Workflow observer: track successful chat workflows for skill crystallization."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import AppConfig, resolve_path
from app.memory.memory_reviewer import collect_user_statements

_QUESTION_PREFIXES = (
    "what ",
    "who ",
    "where ",
    "when ",
    "why ",
    "how ",
    "do ",
    "does ",
    "did ",
    "can ",
)


def _is_question(text: str) -> bool:
    stripped = text.strip()
    if stripped.endswith("?"):
        return True
    return stripped.lower().startswith(_QUESTION_PREFIXES)


def _normalize_statements(statements: list[str]) -> list[str]:
    """Drop questions and normalize whitespace for fingerprinting."""
    normalized: list[str] = []
    for statement in statements:
        if _is_question(statement):
            continue
        text = re.sub(r"\s+", " ", statement.strip().lower())
        if text:
            normalized.append(text)
    return normalized


def fingerprint_workflow(user_statements: list[str]) -> str:
    """Hash normalized user statements into a stable workflow fingerprint."""
    normalized = _normalize_statements(user_statements)
    if not normalized:
        return ""
    payload = "\n".join(normalized)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:12]


@dataclass
class WorkflowEntry:
    fingerprint: str
    summary: str = ""
    success_count: int = 0
    last_marked: str = ""
    user_statements: list[str] = field(default_factory=list)
    crystallized_as: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "summary": self.summary,
            "success_count": self.success_count,
            "last_marked": self.last_marked,
            "user_statements": self.user_statements,
            "crystallized_as": self.crystallized_as,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowEntry:
        return cls(
            fingerprint=str(data.get("fingerprint", "")),
            summary=str(data.get("summary", "")),
            success_count=int(data.get("success_count", 0)),
            last_marked=str(data.get("last_marked", "")),
            user_statements=list(data.get("user_statements") or []),
            crystallized_as=str(data.get("crystallized_as", "")),
        )


@dataclass
class WorkflowMarkResult:
    fingerprint: str
    success_count: int
    threshold: int
    threshold_reached: bool
    already_crystallized: bool
    summary: str
    user_statements: list[str]
    message: str = ""


class WorkflowObserver:
    """Persists workflow success markers and detects crystallization thresholds."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.log_path = resolve_path(config.skills.workflow_log_path)
        self._last_marked_fingerprint: str = ""
        self._last_marked_turn: int = -1

    def ensure_log(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            self._save_log({})

    def _load_log(self) -> dict[str, dict[str, Any]]:
        self.ensure_log()
        try:
            with open(self.log_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError):
            return {}
        workflows = data.get("workflows", {})
        return workflows if isinstance(workflows, dict) else {}

    def _save_log(self, workflows: dict[str, dict[str, Any]]) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w", encoding="utf-8") as handle:
            json.dump({"workflows": workflows}, handle, indent=2)

    def get_workflow(self, fingerprint: str) -> WorkflowEntry | None:
        if not fingerprint:
            return None
        data = self._load_log().get(fingerprint)
        if not data:
            return None
        return WorkflowEntry.from_dict(data)

    def list_workflows(self) -> list[WorkflowEntry]:
        return [
            WorkflowEntry.from_dict(entry)
            for entry in self._load_log().values()
        ]

    def mark_crystallized(self, fingerprint: str, skill_name: str) -> None:
        workflows = self._load_log()
        entry = workflows.get(fingerprint)
        if not entry:
            return
        entry["crystallized_as"] = skill_name
        workflows[fingerprint] = entry
        self._save_log(workflows)

    def mark_success(
        self,
        messages: list[dict[str, str]],
        note: str = "",
        turn_count: int = 0,
    ) -> WorkflowMarkResult:
        """Record a successful workflow from recent user messages."""
        window = self.config.skills.success_window_turns
        user_statements = collect_user_statements(messages, last_n_turns=window)
        fingerprint = fingerprint_workflow(user_statements)
        threshold = self.config.skills.min_successful_repeats

        if not fingerprint:
            return WorkflowMarkResult(
                fingerprint="",
                success_count=0,
                threshold=threshold,
                threshold_reached=False,
                already_crystallized=False,
                summary=note.strip(),
                user_statements=user_statements,
                message="No declarative user messages found to fingerprint.",
            )

        workflows = self._load_log()
        existing = workflows.get(fingerprint, {})
        entry = WorkflowEntry.from_dict({**existing, "fingerprint": fingerprint})

        if entry.crystallized_as:
            return WorkflowMarkResult(
                fingerprint=fingerprint,
                success_count=entry.success_count,
                threshold=threshold,
                threshold_reached=False,
                already_crystallized=True,
                summary=entry.summary or note.strip(),
                user_statements=entry.user_statements or user_statements,
                message=(
                    f"Workflow already crystallized as '{entry.crystallized_as}'. "
                    "No new suggestion needed."
                ),
            )

        # Idempotent within the same turn for the same fingerprint.
        if (
            self._last_marked_turn == turn_count
            and self._last_marked_fingerprint == fingerprint
        ):
            return WorkflowMarkResult(
                fingerprint=fingerprint,
                success_count=entry.success_count,
                threshold=threshold,
                threshold_reached=entry.success_count >= threshold,
                already_crystallized=False,
                summary=entry.summary or note.strip(),
                user_statements=entry.user_statements or user_statements,
                message=(
                    f"Workflow already marked this turn "
                    f"({entry.success_count}/{threshold} successes)."
                ),
            )

        entry.success_count += 1
        entry.last_marked = datetime.now(timezone.utc).isoformat(timespec="seconds")
        entry.user_statements = user_statements
        if note.strip():
            entry.summary = note.strip()
        elif not entry.summary and user_statements:
            entry.summary = user_statements[0][:80]

        workflows[fingerprint] = entry.to_dict()
        self._save_log(workflows)

        self._last_marked_fingerprint = fingerprint
        self._last_marked_turn = turn_count

        threshold_reached = entry.success_count >= threshold
        message = (
            f"Workflow success recorded ({entry.success_count}/{threshold})."
        )
        if threshold_reached:
            message += " Threshold reached — run /crystallize to draft a skill."

        return WorkflowMarkResult(
            fingerprint=fingerprint,
            success_count=entry.success_count,
            threshold=threshold,
            threshold_reached=threshold_reached,
            already_crystallized=False,
            summary=entry.summary,
            user_statements=entry.user_statements,
            message=message,
        )

    def get_best_workflow_for_crystallize(
        self,
        fingerprint: str | None = None,
    ) -> WorkflowEntry | None:
        """Return a workflow eligible for crystallization."""
        if fingerprint:
            entry = self.get_workflow(fingerprint)
            if entry and not entry.crystallized_as:
                return entry
            return None

        candidates = [
            entry
            for entry in self.list_workflows()
            if not entry.crystallized_as
            and entry.success_count >= self.config.skills.min_successful_repeats
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda e: (e.success_count, e.last_marked))
