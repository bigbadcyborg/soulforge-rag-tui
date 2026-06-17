"""Reusable Textual widgets for the SoulForge TUI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from rich.text import Text
from textual import events
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Label, RadioButton, RadioSet, Static, TextArea, Input

from app.core.compute_backend import ComputeBackend, UNKNOWN
from app.core.feature_state import FEATURE_KEYS
from app.memory.memory_manager import SECTION_FILENAMES
from app.memory.memory_reviewer import MemorySuggestion
from app.skills.skill_crystallizer import SkillSuggestion
from app.skills.curator import CuratorFinding, format_finding_view
from app.tasks.task_manager import COLUMN_LABELS, COLUMNS, Task
from app.tasks.task_suggester import TaskSuggestion, format_suggestion_view
from app.sessions.session_manager import SessionMeta
from app.rag.retriever import RetrievedChunk, Retriever
from app.tutorial.content import TUTORIAL_STEPS
from app.tools.models import PendingToolCall

ROLE_LABELS = {
    "user": "You",
    "assistant": "Bot",
    "system": "System",
}

ROLE_STYLES = {
    "user": "bold cyan",
    "assistant": "bold green",
    "system": "bold yellow",
}


class ChatMessage(Static):
    """A single chat bubble; supports incremental token appends for streaming."""

    def __init__(self, role: str, text: str = "") -> None:
        self.role = role
        self._text = text
        super().__init__(self._build(), classes=f"role-{role}")

    def append(self, chunk: str) -> None:
        self._text += chunk
        self.update(self._build())

    def set_text(self, text: str) -> None:
        self._text = text
        self.update(self._build())

    def _build(self) -> Text:
        label = ROLE_LABELS.get(self.role, self.role)
        style = ROLE_STYLES.get(self.role, "bold white")
        renderable = Text()
        renderable.append(f"{label}\n", style=style)
        renderable.append(self._text or "")
        return renderable


class PromptInput(TextArea):
    """Multiline chat prompt with paste-friendly editing and submit shortcuts."""

    @dataclass
    class Submitted(Message):
        """Posted when the user submits the prompt."""

        prompt: PromptInput
        value: str

        @property
        def control(self) -> PromptInput:
            return self.prompt

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("show_line_numbers", False)
        kwargs.setdefault("soft_wrap", True)
        super().__init__(*args, **kwargs)

    @property
    def value(self) -> str:
        return self.text

    @value.setter
    def value(self, text: str) -> None:
        self.text = text

    @staticmethod
    def _key_modifiers(key: str) -> set[str]:
        parts = key.lower().split("+")
        if len(parts) <= 1:
            return set()
        return set(parts[:-1])

    @staticmethod
    def _base_key(key: str) -> str:
        return key.lower().split("+")[-1]

    async def _on_key(self, event: events.Key) -> None:
        if self.read_only or self.disabled:
            return

        base_key = self._base_key(event.key)
        if base_key != "enter":
            await super()._on_key(event)
            return

        modifiers = self._key_modifiers(event.key)
        if modifiers & {"ctrl", "meta"}:
            await super()._on_key(event)
            return

        event.stop()
        event.prevent_default()
        self._submit()

    def _submit(self) -> None:
        self.post_message(self.Submitted(self, self.text))


class StatusBar(Horizontal):
    """Bottom status bar: model/features/state on the left, GPU/CPU badge on the right."""

    def __init__(self) -> None:
        super().__init__()
        self._model = "—"
        self._features = "—"
        self._state = "Starting"
        self._compute = UNKNOWN

    def compose(self):
        yield Static(id="status-left")
        yield Static(id="status-compute")

    def on_mount(self) -> None:
        self._refresh_left()
        self._refresh_compute()

    def set_model(self, model: str) -> None:
        self._model = model
        self._refresh_left()

    def set_features(self, features: str) -> None:
        self._features = features
        self._refresh_left()

    def set_state(self, state: str) -> None:
        self._state = state
        self._refresh_left()

    def set_compute(self, backend: ComputeBackend) -> None:
        self._compute = backend
        self._refresh_compute()

    def _build_left(self) -> Text:
        text = Text()
        text.append(" model: ", style="dim")
        text.append(self._model, style="bold")
        text.append("  │  features: ", style="dim")
        text.append(self._features, style="bold")
        text.append("  │  ", style="dim")
        text.append(self._state, style="bold magenta")
        return text

    def _refresh_left(self) -> None:
        left = self.query_one("#status-left", Static)
        left.update(self._build_left())

    def _refresh_compute(self) -> None:
        badge = self.query_one("#status-compute", Static)
        badge.remove_class("mode-gpu", "mode-cpu", "mode-unknown")
        badge.add_class(f"mode-{self._compute.mode}")
        text = Text()
        style = {
            "gpu": "bold green",
            "cpu": "bold yellow",
            "unknown": "dim",
        }.get(self._compute.mode, "bold white")
        text.append(self._compute.label, style=style)
        badge.update(text)


class RagSelectionModal(ModalScreen):
    """Modal for selecting RAG documents."""

    def __init__(self, available_sources: list[str]) -> None:
        super().__init__()
        self.available_sources = available_sources
        # Map sanitized IDs back to original source names
        self.id_to_source = {
            self._sanitize_id(source): source for source in available_sources
        }
        self.selected_sources: list[str] = []

    @staticmethod
    def _sanitize_id(source: str) -> str:
        """Convert source name to valid Textual ID (alphanumeric, underscore, hyphen only)."""
        # Replace invalid characters with underscores
        sanitized = "".join(c if c.isalnum() or c in "-_" else "_" for c in source)
        return f"doc_{sanitized}"

    def compose(self):
        """Compose the modal UI."""
        with Vertical(id="rag-modal-container"):
            yield Label("Select documents to use for RAG:")
            yield Checkbox("All documents", id="all-checkbox")
            for source in self.available_sources:
                sanitized_id = self._sanitize_id(source)
                yield Checkbox(f"  {source}", id=sanitized_id)
            with Container(id="button-container"):
                yield Button("OK", id="ok-button", variant="primary")
                yield Button("Cancel", id="cancel-button")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "ok-button":
            # Collect selected documents
            all_checkbox = self.query_one("#all-checkbox", Checkbox)
            if all_checkbox.value:
                # All documents selected
                self.dismiss(self.available_sources)
            else:
                selected = []
                for sanitized_id, source in self.id_to_source.items():
                    checkbox = self.query_one(f"#{sanitized_id}", Checkbox)
                    if checkbox.value:
                        selected.append(source)
                self.dismiss(selected if selected else None)
        elif event.button.id == "cancel-button":
            self.dismiss(None)

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Handle checkbox changes."""
        if event.checkbox.id == "all-checkbox":
            # Toggle all other checkboxes based on "All" state
            for sanitized_id in self.id_to_source.keys():
                checkbox = self.query_one(f"#{sanitized_id}", Checkbox)
                checkbox.value = event.checkbox.value


MODEL_ADD_SENTINEL = "__model_add__"


class ModelSelectionModal(ModalScreen):
    """Modal for choosing a chat model to switch to."""

    def __init__(self, models: list[str], current: str) -> None:
        super().__init__()
        self.models = models
        self.current = current

    def compose(self):
        with Vertical(id="model-modal-container"):
            yield Label("Select a chat model:")
            if self.models:
                with VerticalScroll(id="model-list-scroll"):
                    with RadioSet(id="model-radio-set"):
                        for name in self.models:
                            label = f"{name} (current)" if name == self.current else name
                            yield RadioButton(label, value=name)
            else:
                yield Static(
                    "No chat models in ./models/. Use Add model to import a .gguf file.",
                    id="model-list-empty",
                )
            with Container(id="button-container"):
                yield Button("Add model...", id="add-model-button")
                yield Button("Switch", id="switch-model-button", variant="primary")
                yield Button("Cancel", id="cancel-model-button")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-model-button":
            self.dismiss(None)
            return
        if event.button.id == "add-model-button":
            self.dismiss(MODEL_ADD_SENTINEL)
            return
        if event.button.id == "switch-model-button":
            if not self.models:
                self.dismiss(None)
                return
            radio_set = self.query_one("#model-radio-set", RadioSet)
            pressed = radio_set.pressed_button
            selected = pressed.value if pressed is not None else self.current
            self.dismiss(selected)


class AddModelModal(ModalScreen):
    """Modal for importing a .gguf file into ./models/."""

    def compose(self):
        with Vertical(id="add-model-modal-container"):
            yield Label("Import a .gguf file into ./models/:")
            yield Input(placeholder="/path/to/model.gguf", id="add-model-path")
            yield Checkbox("Switch to this model after import", id="switch-after-checkbox")
            with Container(id="button-container"):
                yield Button("Import", id="import-model-button", variant="primary")
                yield Button("Cancel", id="cancel-add-model-button")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-add-model-button":
            self.dismiss(None)
            return
        if event.button.id == "import-model-button":
            source = self.query_one("#add-model-path", Input).value.strip()
            if not source:
                self.dismiss(None)
                return
            switch_after = self.query_one("#switch-after-checkbox", Checkbox).value
            self.dismiss((source, switch_after))


FEATURE_TOGGLE_LABELS: dict[str, str] = {
    "soul": "SOUL (persona)",
    "rag": "RAG",
    "memory": "Memory",
    "skills": "Skills",
    "curator": "Curator",
    "kanban": "Kanban",
    "tools": "Tools",
    "show_sources": "Show sources",
    "streaming": "Streaming",
}

FEATURE_TOGGLE_DESCRIPTIONS: dict[str, str] = {
    "soul": "Inject SOUL.md persona, tone, and behavior into the system prompt.",
    "rag": "Retrieve relevant chunks from indexed docs and ground replies in them.",
    "memory": "Load user.md, memory.md, and session.md into the system prompt.",
    "skills": "Inject reusable workflow skills into prompts. (coming soon)",
    "curator": "Review and maintain skills and memory quality. (coming soon)",
    "kanban": "Track tasks on a local Kanban board with /tasks and /task-suggest.",
    "tools": "Let the assistant propose local file, shell, and action tools (approval required for risky ops).",
    "show_sources": "List retrieved document sources after each reply.",
    "streaming": "Stream model tokens into the chat as they are generated.",
}


class FeatureToggleModal(ModalScreen):
    """Modal for toggling runtime feature flags."""

    def __init__(self, current_state: dict[str, bool]) -> None:
        super().__init__()
        self.current_state = current_state

    def compose(self):
        with Vertical(id="feature-modal-container"):
            yield Label("Feature toggles (changes auto-save to config.yaml):")
            for key in FEATURE_KEYS:
                checkbox_id = f"feature-{key}"
                label = FEATURE_TOGGLE_LABELS.get(key, key)
                description = FEATURE_TOGGLE_DESCRIPTIONS.get(key, "")
                with Horizontal(classes="feature-row"):
                    yield Checkbox(label, id=checkbox_id, value=self.current_state.get(key, False))
                    yield Static(description, classes="feature-desc")
            with Container(id="button-container"):
                yield Button("OK", id="ok-button", variant="primary")
                yield Button("Cancel", id="cancel-button")

    def on_mount(self) -> None:
        for key in FEATURE_KEYS:
            checkbox = self.query_one(f"#feature-{key}", Checkbox)
            checkbox.value = self.current_state.get(key, False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ok-button":
            selected = {
                key: self.query_one(f"#feature-{key}", Checkbox).value
                for key in FEATURE_KEYS
            }
            self.dismiss(selected)
        elif event.button.id == "cancel-button":
            self.dismiss(None)


class SourcesModal(ModalScreen):
    """Modal for inspecting retrieved chunks from the last question."""

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        super().__init__()
        self.chunks = chunks

    def compose(self):
        with Vertical(id="sources-modal-container"):
            yield Label("Retrieved sources (last question):")
            with VerticalScroll(id="sources-scroll"):
                text = Retriever.format_sources_detail(self.chunks)
                yield Static(text, id="sources-content")
            with Container(id="button-container"):
                yield Button("Close", id="close-button", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close-button":
            self.dismiss()


class MemoryViewerModal(ModalScreen):
    """Modal for viewing user.md, memory.md, and session.md."""

    def __init__(self, content: str) -> None:
        super().__init__()
        self.content = content

    def compose(self):
        with Vertical(id="memory-modal-container"):
            yield Label("Persistent memory:")
            with VerticalScroll(id="memory-scroll"):
                yield Static(self.content, id="memory-content")
            with Container(id="button-container"):
                yield Button("Close", id="close-button", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close-button":
            self.dismiss()


class DiagnosticsModal(ModalScreen):
    """Scrollable modal for /diagnostics and /config output."""

    def __init__(self, content: str, *, title: str = "Diagnostics") -> None:
        super().__init__()
        self.content = content
        self.title_text = title

    def compose(self):
        with Vertical(id="diagnostics-modal-container"):
            yield Label(self.title_text)
            with VerticalScroll(id="diagnostics-scroll"):
                yield Static(self.content, id="diagnostics-content")
            with Container(id="button-container"):
                yield Button("Close", id="close-button", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close-button":
            self.dismiss()


class MemoryEditModal(ModalScreen):
    """Modal for editing a single memory section."""

    def __init__(self, section: str, initial_text: str, max_chars: int) -> None:
        super().__init__()
        self.section = section
        self.initial_text = initial_text
        self.max_chars = max_chars

    def compose(self):
        filename = SECTION_FILENAMES[self.section]
        with Vertical(id="memory-edit-container"):
            yield Label(f"Edit {filename}:")
            yield TextArea(self.initial_text, id="memory-textarea")
            yield Label(self._char_count_label(len(self.initial_text)), id="memory-char-count")
            with Container(id="button-container"):
                yield Button("Save", id="save-button", variant="primary")
                yield Button("Cancel", id="cancel-button")

    def _char_count_label(self, count: int) -> str:
        return f"Characters: {count} / {self.max_chars}"

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id != "memory-textarea":
            return
        label = self.query_one("#memory-char-count", Label)
        label.update(self._char_count_label(len(event.text_area.text)))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save-button":
            text = self.query_one("#memory-textarea", TextArea).text
            self.dismiss((self.section, text))
        elif event.button.id == "cancel-button":
            self.dismiss(None)


class MemoryReviewModal(ModalScreen):
    """Modal for reviewing and approving a pending memory suggestion."""

    def __init__(self, suggestion: MemorySuggestion, max_chars: int) -> None:
        super().__init__()
        self.suggestion = suggestion
        self.max_chars = max_chars

    def compose(self):
        filename = SECTION_FILENAMES.get(self.suggestion.section, f"{self.suggestion.section}.md")
        count = len(self.suggestion.proposed_content)
        preview = (
            f"Rationale:\n{self.suggestion.rationale}\n\n"
            f"Proposed {filename} ({count}/{self.max_chars} chars):\n"
            f"{self.suggestion.proposed_content or '(empty)'}"
        )
        with Vertical(id="memory-review-container"):
            yield Label(f"Memory review (turn {self.suggestion.turn_count}):")
            with VerticalScroll(id="memory-review-scroll"):
                yield Static(preview, id="memory-review-content")
            with Container(id="button-container"):
                yield Button("Accept", id="accept-button", variant="primary")
                yield Button("Edit", id="edit-button")
                yield Button("Reject", id="reject-button")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "accept-button":
            self.dismiss("accept")
        elif event.button.id == "edit-button":
            self.dismiss("edit")
        elif event.button.id == "reject-button":
            self.dismiss("reject")


class SkillCrystallizeModal(ModalScreen):
    """Modal for reviewing and approving a pending skill crystallization."""

    def __init__(self, suggestion: SkillSuggestion) -> None:
        super().__init__()
        self.suggestion = suggestion

    def compose(self):
        count = len(self.suggestion.proposed_content)
        preview = (
            f"Rationale:\n{self.suggestion.rationale}\n\n"
            f"Proposed {self.suggestion.name}.md ({count} chars):\n"
            f"{self.suggestion.proposed_content or '(empty)'}"
        )
        with Vertical(id="skill-crystallize-container"):
            yield Label(
                f"Skill crystallization ({self.suggestion.success_count} successes):"
            )
            with VerticalScroll(id="skill-crystallize-scroll"):
                yield Static(preview, id="skill-crystallize-content")
            with Container(id="button-container"):
                yield Button("Accept", id="accept-button", variant="primary")
                yield Button("Edit", id="edit-button")
                yield Button("Reject", id="reject-button")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "accept-button":
            self.dismiss("accept")
        elif event.button.id == "edit-button":
            self.dismiss("edit")
        elif event.button.id == "reject-button":
            self.dismiss("reject")


class SkillCrystallizeEditModal(ModalScreen):
    """Edit a pending skill suggestion before saving."""

    def __init__(self, suggestion: SkillSuggestion) -> None:
        super().__init__()
        self.suggestion = suggestion

    def compose(self):
        with Vertical(id="skill-crystallize-edit-container"):
            yield Label(f"Edit skill: {self.suggestion.name}")
            yield Label("Full markdown content:")
            yield TextArea(
                self.suggestion.proposed_content,
                id="skill-crystallize-textarea",
            )
            with Container(id="button-container"):
                yield Button("Save", id="save-button", variant="primary")
                yield Button("Cancel", id="cancel-button")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-button":
            self.dismiss(None)
        elif event.button.id == "save-button":
            text = self.query_one("#skill-crystallize-textarea", TextArea).text
            self.dismiss(text.strip())


class CuratorReviewModal(ModalScreen):
    """Modal for reviewing and approving curator findings one at a time."""

    def __init__(
        self,
        finding: CuratorFinding,
        index: int,
        total: int,
    ) -> None:
        super().__init__()
        self.finding = finding
        self.index = index
        self.total = total

    def compose(self):
        preview = format_finding_view(self.finding, self.index, self.total)
        with Vertical(id="curator-review-container"):
            yield Label(f"Curator review ({self.index + 1}/{self.total}):")
            with VerticalScroll(id="curator-review-scroll"):
                yield Static(preview, id="curator-review-content")
            with Container(id="button-container"):
                yield Button("Approve", id="approve-button", variant="primary")
                yield Button("Ignore", id="ignore-button")
                yield Button("Close", id="close-button")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close-button":
            self.dismiss(None)
        elif event.button.id == "approve-button":
            self.dismiss(("approve", self.finding.finding_id))
        elif event.button.id == "ignore-button":
            self.dismiss(("ignore", self.finding.finding_id))


class SkillViewerModal(ModalScreen):
    """Modal for listing and viewing skills."""

    def __init__(
        self,
        skills: list[dict[str, Any]],
        archived_skills: list[dict[str, Any]] | None = None,
        pending_message: str = "",
    ) -> None:
        super().__init__()
        self.skills = skills
        self.archived_skills = archived_skills or []
        self.pending_message = pending_message

    def compose(self):
        with Vertical(id="skill-viewer-container"):
            yield Label("Active Skills:")
            if self.pending_message:
                yield Label(self.pending_message, classes="dim")
            with VerticalScroll(id="skill-list-scroll"):
                if not self.skills:
                    yield Label("(no active skills)", classes="dim")
                for skill in self.skills:
                    name = skill.get("name", "unnamed")
                    desc = skill.get("description", "")
                    yield Button(
                        f"{name}: {desc}",
                        id=f"skill_btn_{name}",
                        classes="skill-list-item",
                    )
            if self.archived_skills:
                yield Label("Archived Skills:")
                for skill in self.archived_skills:
                    name = skill.get("name", "unnamed")
                    desc = skill.get("description", "")
                    yield Button(
                        f"{name}: {desc} (archived)",
                        id=f"skill_btn_{name}",
                        classes="skill-list-item dim",
                    )
            with Container(id="button-container"):
                yield Button("New Skill", id="new-skill-button", variant="success")
                yield Button("Close", id="close-button", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close-button":
            self.dismiss(None)
        elif event.button.id == "new-skill-button":
            self.dismiss("new")
        elif event.button.id and event.button.id.startswith("skill_btn_"):
            skill_name = event.button.id.replace("skill_btn_", "")
            self.dismiss(("view", skill_name))


class SkillDetailModal(ModalScreen):
    """Modal for viewing skill markdown content."""

    def __init__(self, name: str, content: str, archived: bool = False) -> None:
        super().__init__()
        self.skill_name = name
        self.content = content
        self.archived = archived

    def compose(self):
        label = f"Skill: {self.skill_name}"
        if self.archived:
            label += " (archived)"
        with Vertical(id="skill-detail-container"):
            yield Label(label)
            with VerticalScroll(id="skill-content-scroll"):
                yield Static(self.content, id="skill-content")
            with Container(id="button-container"):
                if self.archived:
                    yield Button("Restore", id="restore-button", variant="success")
                else:
                    yield Button("Archive", id="archive-button", variant="error")
                yield Button("Close", id="close-button", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close-button":
            self.dismiss(None)
        elif event.button.id == "archive-button":
            self.dismiss(("archive", self.skill_name))
        elif event.button.id == "restore-button":
            self.dismiss(("restore", self.skill_name))


class SkillCreateModal(ModalScreen):
    """Modal for creating a new skill manually."""

    def compose(self):
        with Vertical(id="skill-create-container"):
            yield Label("Create New Skill")
            yield Label("Name:")
            yield Input(placeholder="skill_name", id="skill-name-input")
            yield Label("Description:")
            yield Input(placeholder="What this skill does", id="skill-desc-input")
            yield Label("Procedure / Content (Markdown):")
            yield TextArea(id="skill-content-textarea")
            with Container(id="button-container"):
                yield Button("Create", id="create-button", variant="success")
                yield Button("Cancel", id="cancel-button")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-button":
            self.dismiss(None)
        elif event.button.id == "create-button":
            name = self.query_one("#skill-name-input", Input).value.strip()
            desc = self.query_one("#skill-desc-input", Input).value.strip()
            content = self.query_one("#skill-content-textarea", TextArea).text.strip()
            if name and content:
                self.dismiss({"name": name, "description": desc, "content": content})
            else:
                # Basic validation: could use a tooltip or status label here
                pass


class KanbanBoardModal(ModalScreen):
    """Modal for viewing the four-column Kanban board."""

    def __init__(
        self,
        board: dict[str, list[Task]],
        pending_count: int = 0,
    ) -> None:
        super().__init__()
        self.board = board
        self.pending_count = pending_count

    def compose(self):
        header = "Kanban Board"
        if self.pending_count:
            header += f" ({self.pending_count} suggestion(s) pending)"
        with Vertical(id="kanban-board-container"):
            yield Label(header)
            with Horizontal(id="kanban-columns"):
                for column in COLUMNS:
                    tasks = self.board.get(column, [])
                    with Vertical(classes="kanban-column"):
                        yield Label(f"{COLUMN_LABELS[column]} ({len(tasks)})")
                        with VerticalScroll(classes="kanban-column-scroll"):
                            if not tasks:
                                yield Static("(empty)", classes="kanban-empty")
                            for task in tasks:
                                yield Button(
                                    f"[{task.id}] {task.title}",
                                    id=f"task_btn_{task.id}",
                                    classes="kanban-task-item",
                                )
            with Container(id="button-container"):
                yield Button("New Task", id="new-task-button", variant="success")
                if self.pending_count:
                    yield Button(
                        "Review Suggestions",
                        id="review-suggestions-button",
                        variant="primary",
                    )
                yield Button("Close", id="close-button")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close-button":
            self.dismiss(None)
        elif event.button.id == "new-task-button":
            self.dismiss("new")
        elif event.button.id == "review-suggestions-button":
            self.dismiss("review")
        elif event.button.id and event.button.id.startswith("task_btn_"):
            task_id = event.button.id.replace("task_btn_", "", 1)
            self.dismiss(("view", task_id))


class TaskDetailModal(ModalScreen):
    """Modal for viewing and managing a single task."""

    def __init__(self, column: str, task: Task) -> None:
        super().__init__()
        self.column = column
        self.kanban_task = task

    def compose(self):
        column_label = COLUMN_LABELS.get(self.column, self.column)
        detail = (
            f"ID: {self.kanban_task.id}\n"
            f"Column: {column_label}\n"
            f"Created: {self.kanban_task.created_at}\n"
            f"Updated: {self.kanban_task.updated_at}\n\n"
            f"{self.kanban_task.description or '(no description)'}"
        )
        with Vertical(id="task-detail-container"):
            yield Label(f"Task: {self.kanban_task.title}")
            with VerticalScroll(id="task-detail-scroll"):
                yield Static(detail, id="task-detail-content")
                with Vertical(id="task-detail-button-container"):
                    yield Label("Move to:")
                    for column in COLUMNS:
                        if column == self.column:
                            continue
                        label = COLUMN_LABELS[column]
                        yield Button(
                            label,
                            id=f"move_{column}",
                            variant="primary",
                            classes="task-move-button",
                        )
                    with Horizontal(id="task-detail-footer-buttons"):
                        yield Button("Delete", id="delete-button", variant="error")
                        yield Button("Close", id="close-button")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close-button":
            self.dismiss(None)
        elif event.button.id == "delete-button":
            self.dismiss(("delete", self.kanban_task.id))
        elif event.button.id and event.button.id.startswith("move_"):
            target = event.button.id.replace("move_", "", 1)
            self.dismiss(("move", self.kanban_task.id, target))


class TaskCreateModal(ModalScreen):
    """Modal for creating a new task."""

    def compose(self):
        with Vertical(id="task-create-container"):
            yield Label("Create New Task")
            yield Label("Title:")
            yield Input(placeholder="Task title", id="task-title-input")
            yield Label("Description (optional):")
            yield TextArea(id="task-desc-textarea")
            with Container(id="button-container"):
                yield Button("Create", id="create-button", variant="success")
                yield Button("Cancel", id="cancel-button")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-button":
            self.dismiss(None)
        elif event.button.id == "create-button":
            title = self.query_one("#task-title-input", Input).value.strip()
            description = self.query_one("#task-desc-textarea", TextArea).text.strip()
            if title:
                self.dismiss({"title": title, "description": description})


class TaskSuggestionModal(ModalScreen):
    """Modal for reviewing and approving task suggestions one at a time."""

    def __init__(
        self,
        suggestion: TaskSuggestion,
        index: int,
        total: int,
    ) -> None:
        super().__init__()
        self.suggestion = suggestion
        self.index = index
        self.total = total

    def compose(self):
        preview = format_suggestion_view(self.suggestion, self.index, self.total)
        with Vertical(id="task-suggestion-container"):
            yield Label(f"Task suggestion ({self.index + 1}/{self.total}):")
            with VerticalScroll(id="task-suggestion-scroll"):
                yield Static(preview, id="task-suggestion-content")
            with Container(id="button-container"):
                yield Button("Approve", id="approve-button", variant="primary")
                yield Button("Ignore", id="ignore-button")
                yield Button("Close", id="close-button")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close-button":
            self.dismiss(None)
        elif event.button.id == "approve-button":
            self.dismiss(("approve", self.suggestion.suggestion_id))
        elif event.button.id == "ignore-button":
            self.dismiss(("ignore", self.suggestion.suggestion_id))


class ToolCallModal(ModalScreen):
    """Modal for reviewing and approving pending tool calls."""

    def __init__(
        self,
        pending: PendingToolCall,
        preview: str,
        index: int,
        total: int,
    ) -> None:
        super().__init__()
        self.pending = pending
        self.preview = preview
        self.index = index
        self.total = total

    def compose(self):
        with Vertical(id="tool-call-container"):
            yield Label(
                f"Tool call ({self.index + 1}/{self.total}) — "
                f"{self.pending.call.name} [{self.pending.risk.value}]"
            )
            with VerticalScroll(id="tool-call-scroll"):
                yield Static(self.preview, id="tool-call-content")
            with Container(id="button-container"):
                yield Button("Approve", id="approve-button", variant="primary")
                yield Button("Reject", id="reject-button")
                yield Button("Close", id="close-button")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close-button":
            self.dismiss(None)
        elif event.button.id == "approve-button":
            self.dismiss(("approve", self.pending.call_id))
        elif event.button.id == "reject-button":
            self.dismiss(("reject", self.pending.call_id))


class ToolsMenuModal(ModalScreen):
    """Main tools workshop: status, allowlist management, and actions."""

    def __init__(self, menu_data: dict) -> None:
        super().__init__()
        self.menu_data = menu_data

    def compose(self):
        header = "Tools Workshop"
        if self.menu_data.get("pending_count"):
            header += f" ({self.menu_data['pending_count']} pending)"
        with Vertical(id="tools-menu-container"):
            yield Label(header)
            with VerticalScroll(id="tools-menu-scroll"):
                yield Static(self.menu_data.get("catalog", ""), id="tools-menu-catalog")
                allowlist = self.menu_data.get("allowlist") or []
                if allowlist:
                    yield Label("Shell allowlist:")
                    for entry in allowlist:
                        safe_id = entry.replace(" ", "_").replace("/", "_")[:40]
                        with Horizontal(classes="allowlist-row"):
                            yield Static(f"  {entry}", classes="allowlist-entry")
                            yield Button(
                                "Remove",
                                id=f"remove_{safe_id}",
                                classes="allowlist-remove",
                            )
            with Container(id="button-container"):
                yield Button("Test Tool", id="test-tool-button", variant="primary")
                if self.menu_data.get("allow_shell"):
                    yield Button("Add Shell Command", id="add-shell-button")
                yield Button("View Log", id="view-log-button")
                yield Button("Close", id="close-button")

    def on_mount(self) -> None:
        self._allowlist_button_ids = {}
        for entry in self.menu_data.get("allowlist") or []:
            safe_id = entry.replace(" ", "_").replace("/", "_")[:40]
            self._allowlist_button_ids[f"remove_{safe_id}"] = entry

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "close-button":
            self.dismiss(None)
        elif bid == "test-tool-button":
            self.dismiss("test")
        elif bid == "add-shell-button":
            self.dismiss("add_shell")
        elif bid == "view-log-button":
            self.dismiss("log")
        elif bid.startswith("remove_"):
            command = self._allowlist_button_ids.get(bid, "")
            if command:
                self.dismiss(("remove", command))


class ToolTestModal(ModalScreen):
    """Pick a tool, provide JSON args, run a manual test."""

    def __init__(
        self,
        tool_defs: list[dict],
        run_test,
    ) -> None:
        super().__init__()
        self.tool_defs = tool_defs
        self.run_test = run_test
        self.selected_tool = tool_defs[0]["name"] if tool_defs else ""

    def compose(self):
        with Vertical(id="tool-test-container"):
            yield Label("Test Tool")
            with VerticalScroll(id="tool-test-body"):
                yield Label("Select tool:")
                with VerticalScroll(id="tool-test-picker"):
                    for tool_def in self.tool_defs:
                        name = tool_def["name"]
                        flag = "ok" if tool_def.get("available") else "off"
                        yield Button(
                            f"{name} [{tool_def.get('risk', '?')}] ({flag})",
                            id=f"pick_{name}",
                            classes="tool-pick-btn",
                        )
                yield Label("Args (JSON):")
                example = "{}"
                if self.tool_defs:
                    example = self.tool_defs[0].get("example_args", "{}")
                yield TextArea(example, id="tool-test-args")
                yield Label("", id="tool-test-result-label")
                with VerticalScroll(id="tool-test-result-scroll"):
                    yield Static("(run a test to see output)", id="tool-test-result")
            with Container(id="button-container"):
                yield Button("Run Test", id="run-test-button", variant="primary")
                yield Button("Close", id="close-button")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "close-button":
            self.dismiss(None)
        elif bid.startswith("pick_"):
            self.selected_tool = bid.replace("pick_", "", 1)
            for tool_def in self.tool_defs:
                if tool_def["name"] == self.selected_tool:
                    self.query_one("#tool-test-args", TextArea).text = tool_def.get(
                        "example_args", "{}"
                    )
                    break
        elif bid == "run-test-button":
            import json

            raw = self.query_one("#tool-test-args", TextArea).text.strip()
            try:
                args = json.loads(raw) if raw else {}
            except json.JSONDecodeError as error:
                self.query_one("#tool-test-result", Static).update(
                    f"Invalid JSON: {error}"
                )
                return
            if not isinstance(args, dict):
                self.query_one("#tool-test-result", Static).update(
                    "Args must be a JSON object."
                )
                return
            if not self.selected_tool:
                self.query_one("#tool-test-result", Static).update("Select a tool first.")
                return
            success, text = self.run_test(self.selected_tool, args)
            label = "OK" if success else "FAILED"
            self.query_one("#tool-test-result-label", Label).update(
                f"Result: {label}"
            )
            self.query_one("#tool-test-result", Static).update(text)


class AddShellCommandModal(ModalScreen):
    """Add a command prefix to tools.shellAllowlist."""

    def compose(self):
        with Vertical(id="add-shell-container"):
            yield Label("Add Shell Command")
            yield Label("Prefix must match start of command (e.g. git status):")
            yield Input(placeholder="git status", id="shell-command-input")
            with Container(id="button-container"):
                yield Button("Add", id="add-button", variant="primary")
                yield Button("Cancel", id="cancel-button")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-button":
            self.dismiss(None)
        elif event.button.id == "add-button":
            command = self.query_one("#shell-command-input", Input).value.strip()
            self.dismiss(command or None)


class SessionBrowserModal(ModalScreen):
    """Modal for browsing saved conversation sessions."""

    def __init__(self, sessions: list[SessionMeta]) -> None:
        super().__init__()
        self.sessions = sessions

    def compose(self):
        with Vertical(id="session-browser-container"):
            yield Label("Saved Sessions")
            with VerticalScroll(id="session-browser-scroll"):
                if not self.sessions:
                    yield Static("(no saved sessions)", classes="session-empty")
                for session in self.sessions:
                    label = f"[{session.id}] {session.title} ({session.message_count} msgs)"
                    yield Button(
                        label,
                        id=f"session_btn_{session.id}",
                        classes="session-list-item",
                    )
            with Container(id="button-container"):
                yield Button("Save Current", id="save-current-button", variant="success")
                yield Button("Close", id="close-button")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close-button":
            self.dismiss(None)
        elif event.button.id == "save-current-button":
            self.dismiss("save")
        elif event.button.id and event.button.id.startswith("session_btn_"):
            session_id = event.button.id.replace("session_btn_", "", 1)
            self.dismiss(("view", session_id))


class SessionDetailModal(ModalScreen):
    """Modal for viewing a saved session and loading or deleting it."""

    def __init__(self, session: SessionMeta) -> None:
        super().__init__()
        self.session_meta = session

    def compose(self):
        summary = self.session_meta.summary or "(no summary)"
        detail = (
            f"ID: {self.session_meta.id}\n"
            f"Created: {self.session_meta.created_at}\n"
            f"Updated: {self.session_meta.updated_at}\n"
            f"Turns: {self.session_meta.turn_count}\n"
            f"Messages: {self.session_meta.message_count}\n\n"
            f"Summary:\n{summary}"
        )
        with Vertical(id="session-detail-container"):
            yield Label(f"Session: {self.session_meta.title}")
            with VerticalScroll(id="session-detail-scroll"):
                yield Static(detail, id="session-detail-content")
            with Container(id="button-container"):
                yield Button("Load", id="load-button", variant="primary")
                yield Button("Delete", id="delete-button", variant="error")
                yield Button("Back", id="back-button")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-button":
            self.dismiss(None)
        elif event.button.id == "load-button":
            self.dismiss(("load", self.session_meta.id))
        elif event.button.id == "delete-button":
            self.dismiss(("delete", self.session_meta.id))


class SessionSaveModal(ModalScreen):
    """Modal for naming a session before save."""

    def __init__(self, default_title: str = "") -> None:
        super().__init__()
        self.default_title = default_title

    def compose(self):
        with Vertical(id="session-save-container"):
            yield Label("Save Session")
            yield Label("Title:")
            yield Input(
                value=self.default_title,
                placeholder="Session title",
                id="session-title-input",
            )
            with Container(id="button-container"):
                yield Button("Save", id="save-button", variant="success")
                yield Button("Cancel", id="cancel-button")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-button":
            self.dismiss(None)
        elif event.button.id == "save-button":
            title = self.query_one("#session-title-input", Input).value.strip()
            self.dismiss(title or self.default_title)


class TutorialFinished(Message):
    """Posted when the user finishes or skips the tutorial panel."""

    def __init__(self, result: str) -> None:
        self.result = result
        super().__init__()


class TutorialWizardPanel(Vertical):
    """Side-panel onboarding wizard; chat input stays usable alongside."""

    def __init__(self) -> None:
        super().__init__(id="tutorial-panel")
        self.step_index = 0
        self._checks: dict[str, bool] = {}

    def compose(self):
        yield Label("Tutorial — try commands in the chat as you go", id="tutorial-hint")
        yield Label("", id="tutorial-title")
        yield Label("", id="tutorial-progress")
        with VerticalScroll(id="tutorial-body-scroll"):
            yield Static("", id="tutorial-body")
        with Vertical(id="tutorial-checklist"):
            pass
        with Vertical(id="tutorial-button-stack"):
            with Container(classes="tutorial-button-row"):
                yield Button("Previous", id="tutorial-prev")
                yield Button("Next", id="tutorial-next", variant="primary")
            with Container(classes="tutorial-button-row"):
                yield Button("Skip", id="tutorial-skip")
                yield Button("Finish", id="tutorial-finish", variant="success")

    def show_panel(self) -> None:
        """Reveal the panel and refresh the current step."""
        self.add_class("-visible")
        self._refresh()

    def hide_panel(self) -> None:
        """Hide the panel without posting a completion message."""
        self.remove_class("-visible")

    def _complete(self) -> None:
        self.hide_panel()
        self.post_message(TutorialFinished("completed"))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "tutorial-prev" and self.step_index > 0:
            self.step_index -= 1
            self._refresh()
        elif button_id == "tutorial-next" and self.step_index < len(TUTORIAL_STEPS) - 1:
            self.step_index += 1
            self._refresh()
        elif button_id == "tutorial-skip":
            self._complete()
        elif button_id == "tutorial-finish":
            self._complete()

    def _check_mark(self, checked: bool) -> Text:
        mark = Text()
        if checked:
            mark.append("✓", style="bold bright_green")
        else:
            mark.append("○", style="dim")
        return mark

    def on_click(self, event) -> None:
        row = event.widget
        while row is not None and not row.has_class("tutorial-check-row"):
            row = row.parent
        if row is None or not row.id or not row.id.startswith("tutorial-check-"):
            return
        item_id = row.id.removeprefix("tutorial-check-")
        new_value = not self._checks.get(item_id, False)
        self._checks[item_id] = new_value
        mark = row.query_one(".tutorial-check-mark", Static)
        mark.update(self._check_mark(new_value))
        mark.set_class(new_value, "-checked")
        self._refresh_progress()

    def _refresh(self) -> None:
        step = TUTORIAL_STEPS[self.step_index]
        self.query_one("#tutorial-title", Label).update(step.title)
        self.query_one("#tutorial-body", Static).update(step.body)
        self._refresh_progress()
        self._refresh_checklist()
        self._refresh_nav_buttons()

    def _refresh_progress(self) -> None:
        total_checks = 0
        checked = 0
        for tutorial_step in TUTORIAL_STEPS:
            for item in tutorial_step.checklist:
                total_checks += 1
                if self._checks.get(item.id, False):
                    checked += 1
        progress = f"Step {self.step_index + 1}/{len(TUTORIAL_STEPS)}"
        if total_checks:
            progress += f"  •  Quest checks {checked}/{total_checks}"
        self.query_one("#tutorial-progress", Static).update(progress)

    def _refresh_checklist(self) -> None:
        step = TUTORIAL_STEPS[self.step_index]
        container = self.query_one("#tutorial-checklist", Vertical)
        for child in list(container.children):
            child.remove()
        if not step.checklist:
            container.mount(Static(""))
            return
        container.mount(Static("Quest checklist:"))
        for item in step.checklist:
            value = self._checks.get(item.id, False)
            row = Horizontal(
                classes="tutorial-check-row",
                id=f"tutorial-check-{item.id}",
            )
            container.mount(row)
            mark = Static(self._check_mark(value), classes="tutorial-check-mark")
            if value:
                mark.add_class("-checked")
            row.mount(mark)
            row.mount(Static(item.label, classes="tutorial-check-label"))

    def _refresh_nav_buttons(self) -> None:
        prev_btn = self.query_one("#tutorial-prev", Button)
        next_btn = self.query_one("#tutorial-next", Button)
        finish_btn = self.query_one("#tutorial-finish", Button)
        prev_btn.disabled = self.step_index == 0
        next_btn.disabled = self.step_index >= len(TUTORIAL_STEPS) - 1
        finish_btn.disabled = self.step_index < len(TUTORIAL_STEPS) - 1

