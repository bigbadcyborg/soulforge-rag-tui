"""Skill Manager: CRUD operations and loading for reusable workflow skills."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from app.core.config import AppConfig, resolve_path

REQUIRED_SECTIONS = ("## Trigger", "## Procedure", "## Validation")


@dataclass
class SkillMetadata:
    name: str
    description: str
    tags: list[str] = field(default_factory=list)
    status: str = "active"
    success_count: int = 0
    last_used: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "status": self.status,
            "success_count": self.success_count,
            "last_used": self.last_used,
        }


def _has_skill_sections(content: str) -> bool:
    lower = content.lower()
    return all(section.lower() in lower for section in REQUIRED_SECTIONS)


def _wrap_skill_body(name: str, description: str, content: str, tags: list[str]) -> str:
    """Ensure manual content uses SKILL-001 sections."""
    body = content.strip()
    if _has_skill_sections(body):
        return body

    return (
        f"## Trigger\n\n{description or 'User needs this workflow.'}\n\n"
        f"## Procedure\n\n{body}\n\n"
        f"## Validation\n\nUser confirms the workflow completed successfully.\n"
    )


def _build_skill_file(
    name: str,
    description: str,
    body: str,
    tags: list[str] | None = None,
    success_count: int = 0,
    last_used: str = "",
) -> str:
    tags = tags or []
    used = last_used or date.today().isoformat()
    return (
        f"---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        f"tags: {tags}\n"
        f"status: active\n"
        f"successCount: {success_count}\n"
        f"lastUsed: {used}\n"
        f"---\n\n"
        f"{body.strip()}\n"
    )


class SkillManager:
    """Manages skill files in active/archived folders and their registry."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.active_path = resolve_path(config.skills.active_path)
        self.archived_path = resolve_path(config.skills.archived_path)
        self.registry_path = resolve_path(config.skills.registry_path)
        self.ensure_dirs()

    def ensure_dirs(self) -> None:
        """Create skills directories if missing."""
        self.active_path.mkdir(parents=True, exist_ok=True)
        self.archived_path.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self._save_registry({})

    def _load_registry(self) -> dict[str, dict[str, Any]]:
        if not self.registry_path.exists():
            return {}
        try:
            with open(self.registry_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_registry(self, registry: dict[str, dict[str, Any]]) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w", encoding="utf-8") as handle:
            json.dump(registry, handle, indent=2)

    def list_skills(self, status: str = "active") -> list[dict[str, Any]]:
        registry = self._load_registry()
        return [meta for meta in registry.values() if meta.get("status") == status]

    def skill_exists(self, name: str) -> bool:
        return name in self._load_registry()

    def list_skill_names(self) -> list[str]:
        return list(self._load_registry().keys())

    def get_skill_content(self, name: str) -> str | None:
        """Read the markdown content of a skill."""
        registry = self._load_registry()
        meta = registry.get(name)
        if not meta:
            return None

        status = meta.get("status", "active")
        base_path = self.active_path if status == "active" else self.archived_path
        file_path = base_path / f"{name}.md"

        if not file_path.exists():
            return None

        return file_path.read_text(encoding="utf-8")

    def create_skill(
        self,
        name: str,
        description: str,
        content: str,
        tags: list[str] | None = None,
    ) -> bool:
        """Create a new skill file and record it in the registry."""
        registry = self._load_registry()
        if name in registry:
            return False

        tags = tags or []
        body = _wrap_skill_body(name, description, content, tags)
        full_content = _build_skill_file(name, description, body, tags)

        file_path = self.active_path / f"{name}.md"
        file_path.write_text(full_content, encoding="utf-8")

        registry[name] = SkillMetadata(
            name=name,
            description=description,
            tags=tags,
        ).to_dict()
        self._save_registry(registry)
        return True

    def create_skill_from_suggestion(
        self,
        name: str,
        description: str,
        proposed_content: str,
        success_count: int = 0,
        tags: list[str] | None = None,
    ) -> bool:
        """Save an approved crystallized skill suggestion."""
        registry = self._load_registry()
        if name in registry:
            return False

        tags = tags or []
        file_path = self.active_path / f"{name}.md"
        file_path.write_text(proposed_content.strip() + "\n", encoding="utf-8")

        registry[name] = SkillMetadata(
            name=name,
            description=description,
            tags=tags,
            success_count=success_count,
            last_used=date.today().isoformat(),
        ).to_dict()
        self._save_registry(registry)
        return True

    def record_skill_use(self, name: str) -> bool:
        """Update last_used for a skill."""
        registry = self._load_registry()
        meta = registry.get(name)
        if not meta:
            return False
        meta["last_used"] = date.today().isoformat()
        self._save_registry(registry)
        return True

    def archive_skill(self, name: str) -> bool:
        """Move a skill from active to archived."""
        registry = self._load_registry()
        meta = registry.get(name)
        if not meta or meta.get("status") == "archived":
            return False

        old_path = self.active_path / f"{name}.md"
        new_path = self.archived_path / f"{name}.md"

        if old_path.exists():
            new_path.parent.mkdir(parents=True, exist_ok=True)
            old_path.rename(new_path)

        meta["status"] = "archived"
        self._save_registry(registry)
        return True

    def delete_skill(self, name: str) -> bool:
        """Remove a skill from disk and registry."""
        registry = self._load_registry()
        meta = registry.get(name)
        if not meta:
            return False

        status = meta.get("status", "active")
        base_path = self.active_path if status == "active" else self.archived_path
        file_path = base_path / f"{name}.md"

        if file_path.exists():
            file_path.unlink()

        del registry[name]
        self._save_registry(registry)
        return True

    @staticmethod
    def extract_frontmatter_field(content: str, field_name: str) -> str:
        """Read a simple frontmatter field from skill markdown."""
        match = re.search(
            rf"^{field_name}:\s*(.+)$",
            content,
            flags=re.MULTILINE,
        )
        return match.group(1).strip() if match else ""
