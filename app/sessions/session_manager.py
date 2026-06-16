"""Session manager: save, load, and list persisted conversations."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import AppConfig, resolve_path
from app.memory.memory_reviewer import _strip_user_turn

CONVERSATION_ROLES = ("user", "assistant")


@dataclass
class SessionMeta:
    id: str
    title: str
    created_at: str
    updated_at: str
    summary: str = ""
    turn_count: int = 0
    message_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "summary": self.summary,
            "turn_count": self.turn_count,
            "message_count": self.message_count,
        }


@dataclass
class SavedSession:
    id: str
    title: str
    created_at: str
    updated_at: str
    summary: str
    turn_count: int
    messages: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "summary": self.summary,
            "turn_count": self.turn_count,
            "messages": self.messages,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SavedSession | None:
        if not isinstance(data, dict):
            return None
        session_id = str(data.get("id", "")).strip()
        if not session_id:
            return None
        messages = data.get("messages", [])
        if not isinstance(messages, list):
            messages = []
        cleaned: list[dict[str, str]] = []
        for item in messages:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip()
            if role not in CONVERSATION_ROLES:
                continue
            content = str(item.get("content", ""))
            cleaned.append({"role": role, "content": content})
        return cls(
            id=session_id,
            title=str(data.get("title", session_id)).strip() or session_id,
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            summary=str(data.get("summary", "")),
            turn_count=int(data.get("turn_count", 0) or 0),
            messages=cleaned,
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _slugify_title(text: str, max_len: int = 60) -> str:
    line = re.sub(r"\s+", " ", text.strip())
    if not line:
        return "Untitled session"
    if len(line) <= max_len:
        return line
    return line[: max_len - 3].rstrip() + "..."


def title_from_messages(messages: list[dict[str, str]]) -> str:
    """Derive a session title from the first user message."""
    for message in messages:
        if message.get("role") != "user":
            continue
        content = _strip_user_turn(str(message.get("content", "")))
        if content and not content.startswith("/"):
            return _slugify_title(content)
    return "Untitled session"


def filter_conversation_messages(
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Keep only user/assistant messages for persistence."""
    return [
        {"role": message["role"], "content": message["content"]}
        for message in messages
        if message.get("role") in CONVERSATION_ROLES
        and str(message.get("content", "")).strip()
    ]


class SessionManager:
    """Manages saved conversation JSON files on disk."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.store_dir = resolve_path(config.sessions.store_path)
        self.max_saved = config.sessions.max_saved_sessions
        self.ensure_dir()

    def ensure_dir(self) -> None:
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def _session_path(self, session_id: str) -> Path:
        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "", session_id)
        return self.store_dir / f"{safe_id}.json"

    def _load_file(self, path: Path) -> SavedSession | None:
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError):
            return None
        return SavedSession.from_dict(data)

    def _save_file(self, session: SavedSession) -> None:
        path = self._session_path(session.id)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(session.to_dict(), handle, indent=2)

    def _prune_old_sessions(self) -> None:
        if self.max_saved <= 0:
            return
        sessions = self.list_sessions()
        if len(sessions) <= self.max_saved:
            return
        for meta in sessions[self.max_saved :]:
            self.delete_session(meta.id)

    def list_sessions(self) -> list[SessionMeta]:
        if not self.store_dir.exists():
            return []
        metas: list[SessionMeta] = []
        for path in self.store_dir.glob("*.json"):
            session = self._load_file(path)
            if session is None:
                continue
            metas.append(
                SessionMeta(
                    id=session.id,
                    title=session.title,
                    created_at=session.created_at,
                    updated_at=session.updated_at,
                    summary=session.summary,
                    turn_count=session.turn_count,
                    message_count=len(session.messages),
                )
            )
        metas.sort(key=lambda item: item.updated_at, reverse=True)
        return metas

    def get_session(self, session_id: str) -> SavedSession | None:
        needle = session_id.strip().lower()
        for meta in self.list_sessions():
            if meta.id.lower() == needle or meta.id.lower().startswith(needle):
                return self._load_file(self._session_path(meta.id))
        return None

    def save_session(
        self,
        messages: list[dict[str, str]],
        *,
        title: str = "",
        summary: str = "",
        turn_count: int = 0,
        session_id: str | None = None,
    ) -> SavedSession | None:
        conversation = filter_conversation_messages(messages)
        if not conversation:
            return None

        now = _now_iso()
        resolved_title = title.strip() or title_from_messages(conversation)
        if session_id:
            existing = self.get_session(session_id)
            if existing:
                session = SavedSession(
                    id=existing.id,
                    title=resolved_title,
                    created_at=existing.created_at,
                    updated_at=now,
                    summary=summary if summary else existing.summary,
                    turn_count=turn_count,
                    messages=conversation,
                )
                self._save_file(session)
                return session

        new_id = session_id or f"{datetime.now(timezone.utc).strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"
        session = SavedSession(
            id=new_id,
            title=resolved_title,
            created_at=now,
            updated_at=now,
            summary=summary,
            turn_count=turn_count,
            messages=conversation,
        )
        self._save_file(session)
        self._prune_old_sessions()
        return session

    def update_summary(self, session_id: str, summary: str) -> bool:
        session = self.get_session(session_id)
        if session is None:
            return False
        session.summary = summary.strip()
        session.updated_at = _now_iso()
        self._save_file(session)
        return True

    def delete_session(self, session_id: str) -> bool:
        session = self.get_session(session_id)
        if session is None:
            return False
        path = self._session_path(session.id)
        if path.exists():
            path.unlink()
            return True
        return False

    def format_session_list(self) -> str:
        sessions = self.list_sessions()
        if not sessions:
            return "No saved sessions. Use /session-save to save the current chat."
        lines = ["Saved sessions:", "=" * 40]
        for meta in sessions:
            summary_preview = meta.summary[:80] + "..." if len(meta.summary) > 80 else meta.summary
            lines.append(f"[{meta.id}] {meta.title}")
            lines.append(
                f"  Updated: {meta.updated_at} | Turns: {meta.turn_count} | "
                f"Messages: {meta.message_count}"
            )
            if summary_preview:
                lines.append(f"  Summary: {summary_preview}")
            lines.append("")
        lines.append("Load: /session-load <id>")
        return "\n".join(lines).rstrip()
