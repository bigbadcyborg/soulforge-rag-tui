"""Chat controller: the non-UI heart of the chatbot.

Owns the model runtime, prompt assembly, memory, and RAG retrieval, and exposes
a small surface that both the CLI loop and the Textual TUI drive. Keeping this
logic UI-agnostic means the same code path powers every front end.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterator

from app.core.compute_backend import ComputeBackend
from app.core.config import PROJECT_ROOT, AppConfig
from app.core.feature_state import FeatureStateManager
from app.core.model_runtime import ModelRuntime
from app.core.prompt_builder import PromptBuilder
from app.memory.memory_manager import MemoryManager, MemorySnapshot
from app.memory.memory_reviewer import (
    MemorySuggestion,
    format_suggestion_view,
    generate_suggestion,
    merge_and_compact,
)
from app.rag.ingest import IngestResult, ingest_documents
from app.rag.retriever import Retriever, RetrievedChunk, get_store_stats
from app.skills.skill_crystallizer import (
    SkillSuggestion,
    format_suggestion_view as format_skill_suggestion_view,
    generate_suggestion as generate_skill_suggestion,
    resolve_unique_name,
)
from app.skills.skill_manager import SkillManager
from app.skills.workflow_observer import WorkflowMarkResult, WorkflowObserver

SOUL_PATH = PROJECT_ROOT / "SOUL.md"


@dataclass
class TurnReviewResult:
    turn_count: int
    review_due: bool
    has_suggestion: bool
    message: str = ""


@dataclass
class SkillMarkResult:
    mark: WorkflowMarkResult
    has_suggestion: bool
    should_open_modal: bool
    message: str = ""


@dataclass
class SkillCrystallizeResult:
    has_suggestion: bool
    message: str = ""


def load_soul() -> str:
    """Load the persona from SOUL.md, or fall back to a neutral default."""
    if not SOUL_PATH.exists():
        return "You are a helpful local chatbot."
    return SOUL_PATH.read_text(encoding="utf-8", errors="ignore").strip()


class ChatController:
    """Coordinates models, prompts, memory, and retrieval for a session."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.runtime = ModelRuntime(config)
        self.prompt_builder = PromptBuilder(config)
        self.memory_manager = MemoryManager(config)
        self.skill_manager = SkillManager(config)
        self.workflow_observer = WorkflowObserver(config)
        self.features = FeatureStateManager(config, on_change=self._on_feature_change)

        self.turn_count: int = 0
        self.pending_suggestion: MemorySuggestion | None = None
        self.pending_skill_suggestion: SkillSuggestion | None = None
        self._pending_crystallize_fingerprint: str = ""

        # RAG is initialized lazily to avoid loading embedding model at startup
        self._retriever: Retriever | None = None
        self.selected_sources: list[str] | None = None  # None = all sources

        self.soul_text: str = ""
        self.memory: MemorySnapshot | None = None
        self.messages: list[dict[str, str]] = []
        self.last_retrieved_chunks: list[RetrievedChunk] = []
        self.loaded: bool = False

    def _on_feature_change(self, key: str, enabled: bool) -> None:
        if key in ("soul", "memory", "skills"):
            self._rebuild_system_prompt()
        elif key == "rag" and enabled and self.selected_sources is None:
            self.selected_sources = self.get_available_sources()

    def _initialize_retriever(self) -> Retriever | None:
        """Lazily initialize the retriever (only when RAG is first enabled)."""
        if self._retriever is not None:
            return self._retriever
        try:
            self._retriever = Retriever(self.config, self.runtime)
            self.runtime.load_embedding_model()
            return self._retriever
        except Exception as error:  # noqa: BLE001
            print(f"[rag] Failed to initialize retriever: {error}")
            return None

    @property
    def retriever(self) -> Retriever | None:
        """Return retriever if RAG is enabled, initializing if needed."""
        if self.features.is_enabled("rag"):
            return self._initialize_retriever()
        return None

    @property
    def rag_enabled(self) -> bool:
        """Whether RAG retrieval is active (compat for existing call sites)."""
        return self.features.is_enabled("rag")

    def load(self) -> None:
        """Load models and assemble the initial system prompt (blocking)."""
        self.memory_manager.ensure_files()
        self.soul_text = load_soul() if self.features.is_enabled("soul") else ""
        self.memory = (
            self.memory_manager.load() if self.features.is_enabled("memory") else None
        )

        self.runtime.load_chat_model()

        if self.features.is_enabled("rag") and self.selected_sources is None:
            self.selected_sources = self.get_available_sources()

        self.messages = [{"role": "system", "content": self._build_system_prompt()}]
        self.loaded = True

    def _build_system_prompt(self) -> str:
        skills: list[str] = []
        if self.features.is_enabled("skills"):
            # v1: Load all active skills for keyword matching or manual use.
            # In a later iteration, this will be more selective.
            active_skills = self.skill_manager.list_skills(status="active")
            for meta in active_skills:
                content = self.skill_manager.get_skill_content(meta["name"])
                if content:
                    skills.append(content)

        return self.prompt_builder.build_system_prompt(self.soul_text, self.memory, skills)

    def _rebuild_system_prompt(self) -> None:
        """Reload soul/memory content and refresh the system message."""
        self.soul_text = load_soul() if self.features.is_enabled("soul") else ""
        self.memory = (
            self.memory_manager.load() if self.features.is_enabled("memory") else None
        )
        if self.messages:
            self.messages[0] = {
                "role": "system",
                "content": self._build_system_prompt(),
            }

    def reload_soul(self) -> None:
        """Reload SOUL.md and rebuild the system prompt without restarting."""
        self._rebuild_system_prompt()

    def reload_memory(self) -> None:
        """Reload memory files and rebuild the system prompt without restarting."""
        self._rebuild_system_prompt()

    def get_memory_view(self) -> str:
        """Return formatted memory content with character counts."""
        self.memory_manager.ensure_files()
        snapshot = self.memory_manager.load()
        return self.memory_manager.format_view(
            snapshot,
            memory_enabled=self.features.is_enabled("memory"),
        )

    def save_memory(self, section: str, content: str) -> bool:
        """Save a memory section and reload the system prompt. Returns True if truncated."""
        _, truncated = self.memory_manager.save(section, content)
        self.reload_memory()
        return truncated

    def enable_memory(self) -> None:
        """Enable memory injection."""
        self.set_feature("memory", True)

    def disable_memory(self) -> None:
        """Disable memory injection."""
        self.set_feature("memory", False)

    def complete_turn(self) -> TurnReviewResult:
        """Increment turn counter and maybe generate a memory review suggestion."""
        if not self.features.is_enabled("memory"):
            return TurnReviewResult(self.turn_count, False, False)

        self.turn_count += 1
        return self._run_memory_review_if_due()

    def maybe_trigger_memory_review(self) -> bool:
        """Run review at the current turn count without incrementing."""
        result = self._run_memory_review_if_due()
        return result.has_suggestion and self.pending_suggestion is not None

    def _run_memory_review_if_due(self) -> TurnReviewResult:
        interval = self.config.memory.update_every_turns
        review_due = interval > 0 and self.turn_count % interval == 0

        if not review_due:
            return TurnReviewResult(self.turn_count, False, False)

        if self.pending_suggestion is not None:
            return TurnReviewResult(
                self.turn_count,
                True,
                True,
                "Memory review already pending. Run /memory-review to view.",
            )

        snapshot = self.memory_manager.load()
        suggestion, error = generate_suggestion(
            self.runtime,
            self.config,
            self.messages,
            snapshot,
            self.turn_count,
        )
        if error:
            return TurnReviewResult(
                self.turn_count,
                True,
                False,
                f"Memory review (turn {self.turn_count}) failed: {error}",
            )

        if suggestion is None:
            return TurnReviewResult(
                self.turn_count,
                True,
                False,
                f"Memory review (turn {self.turn_count}): no new facts to save.",
            )

        self.pending_suggestion = suggestion
        return TurnReviewResult(
            self.turn_count,
            True,
            True,
            f"Memory review ready (turn {self.turn_count}).",
        )

    def get_memory_review(self) -> str:
        """Return formatted pending suggestion, or a no-pending message."""
        if self.pending_suggestion is None:
            return "No pending memory suggestion."
        limit = self.memory_manager.limits()[self.pending_suggestion.section]
        return format_suggestion_view(self.pending_suggestion, limit)

    def accept_memory_suggestion(
        self,
        content: str | None = None,
    ) -> tuple[bool, bool]:
        """Save pending suggestion. Returns (saved, was_compacted)."""
        if self.pending_suggestion is None:
            raise ValueError("No pending memory suggestion.")

        section = self.pending_suggestion.section
        text = content if content is not None else self.pending_suggestion.proposed_content
        max_chars = self.memory_manager.limits()[section]
        final_text, was_compacted = merge_and_compact(
            self.runtime,
            section,
            "",
            text,
            max_chars,
        )
        _, truncated = self.memory_manager.save(section, final_text)
        self.reload_memory()
        self.pending_suggestion = None
        return True, was_compacted or truncated

    def reject_memory_suggestion(self) -> None:
        """Discard the pending memory suggestion without saving."""
        self.pending_suggestion = None

    def mark_workflow_success(self, note: str = "") -> SkillMarkResult:
        """Record a successful workflow from recent chat messages."""
        mark = self.workflow_observer.mark_success(
            self.messages,
            note=note,
            turn_count=self.turn_count,
        )
        if mark.already_crystallized:
            return SkillMarkResult(
                mark=mark,
                has_suggestion=False,
                should_open_modal=False,
                message=mark.message,
            )

        should_open_modal = False
        has_suggestion = self.pending_skill_suggestion is not None

        if mark.threshold_reached:
            self._pending_crystallize_fingerprint = mark.fingerprint
            if self.config.skills.auto_create:
                result = self.crystallize_workflow(mark.fingerprint)
                return SkillMarkResult(
                    mark=mark,
                    has_suggestion=result.has_suggestion,
                    should_open_modal=result.has_suggestion,
                    message=result.message or mark.message,
                )
            message = f"{mark.message} Run /crystallize to draft a skill."
            return SkillMarkResult(
                mark=mark,
                has_suggestion=False,
                should_open_modal=False,
                message=message,
            )

        return SkillMarkResult(
            mark=mark,
            has_suggestion=has_suggestion,
            should_open_modal=should_open_modal,
            message=mark.message,
        )

    def crystallize_workflow(
        self,
        fingerprint: str | None = None,
    ) -> SkillCrystallizeResult:
        """Generate a skill suggestion from a logged workflow."""
        target_fp = fingerprint or self._pending_crystallize_fingerprint
        workflow = self.workflow_observer.get_best_workflow_for_crystallize(target_fp)
        if workflow is None:
            return SkillCrystallizeResult(
                has_suggestion=False,
                message=(
                    "No workflow ready for crystallization. "
                    f"Mark success {self.config.skills.min_successful_repeats} times "
                    "with /success first."
                ),
            )

        if self.pending_skill_suggestion is not None:
            return SkillCrystallizeResult(
                has_suggestion=True,
                message="Skill suggestion already pending. Run /skill-accept or /skill-reject.",
            )

        existing_names = self.skill_manager.list_skill_names()
        suggestion, error = generate_skill_suggestion(
            self.runtime,
            self.config,
            self.messages,
            workflow,
            existing_names,
        )
        if error and suggestion is None:
            return SkillCrystallizeResult(
                has_suggestion=False,
                message=f"Skill crystallization failed: {error}",
            )

        if suggestion is None:
            return SkillCrystallizeResult(
                has_suggestion=False,
                message="Skill crystallization produced no suggestion.",
            )

        unique_name = resolve_unique_name(suggestion.name, set(existing_names))
        if unique_name != suggestion.name:
            suggestion.name = unique_name
            suggestion.proposed_content = re.sub(
                r"^name: .+$",
                f"name: {unique_name}",
                suggestion.proposed_content,
                count=1,
                flags=re.MULTILINE,
            )

        self.pending_skill_suggestion = suggestion
        self._pending_crystallize_fingerprint = workflow.fingerprint
        return SkillCrystallizeResult(
            has_suggestion=True,
            message=f"Skill suggestion ready for '{suggestion.name}'.",
        )

    def get_skill_review(self) -> str:
        if self.pending_skill_suggestion is None:
            return "No pending skill suggestion."
        return format_skill_suggestion_view(self.pending_skill_suggestion)

    def accept_skill_suggestion(self, content: str | None = None) -> bool:
        """Save pending skill suggestion. Requires skills feature enabled."""
        if self.pending_skill_suggestion is None:
            raise ValueError("No pending skill suggestion.")
        if not self.features.is_enabled("skills"):
            raise ValueError(
                "Skills feature is disabled. Run /features skills on before accepting."
            )

        suggestion = self.pending_skill_suggestion
        text = content if content is not None else suggestion.proposed_content
        saved = self.skill_manager.create_skill_from_suggestion(
            suggestion.name,
            suggestion.description,
            text,
            success_count=suggestion.success_count,
        )
        if not saved:
            raise ValueError(
                f"Could not save skill '{suggestion.name}' (it may already exist)."
            )

        if suggestion.fingerprint:
            self.workflow_observer.mark_crystallized(
                suggestion.fingerprint,
                suggestion.name,
            )

        self.pending_skill_suggestion = None
        self._pending_crystallize_fingerprint = ""
        self._rebuild_system_prompt()
        return True

    def reject_skill_suggestion(self) -> None:
        """Discard pending skill suggestion without saving."""
        self.pending_skill_suggestion = None

    def add_user_turn(self, user_input: str) -> list[RetrievedChunk]:
        """Retrieve context, append the user message, and return any sources."""
        if self.features.is_enabled("memory"):
            self.reload_memory()

        chunks: list[RetrievedChunk] = []
        context_text = ""
        if self.retriever is not None:
            chunks = self.retriever.retrieve(user_input)
            if self.selected_sources is not None and chunks:
                chunks = [c for c in chunks if c.source in self.selected_sources]
            context_text = Retriever.format_context(chunks)

        user_turn = self.prompt_builder.build_user_turn(
            user_input,
            context_text,
            use_rag=self.features.is_enabled("rag"),
            memory=self.memory,
            use_memory=self.features.is_enabled("memory"),
        )
        self.messages.append({"role": "user", "content": user_turn})
        self.last_retrieved_chunks = chunks
        return chunks

    def run_ingest(
        self,
        on_progress: Callable[[str, str, int, int], None] | None = None,
    ) -> IngestResult:
        """Index docs/ into ChromaDB and refresh the retriever cache."""
        result = ingest_documents(self.config, self.runtime, on_progress=on_progress)
        if self._retriever is not None:
            self._retriever.reset_collection()
        return result

    def enable_rag(self, sources: list[str] | None = None) -> None:
        """Enable RAG and optionally set document filters."""
        self.set_feature("rag", True)
        if sources is not None:
            self.selected_sources = sources
        elif self.selected_sources is None:
            self.selected_sources = self.get_available_sources()

    def disable_rag(self) -> None:
        """Disable RAG retrieval."""
        self.set_feature("rag", False)

    def toggle_rag(self) -> bool:
        """Toggle RAG on/off and return the new state."""
        return self.features.toggle("rag")

    def set_feature(self, key: str, enabled: bool) -> None:
        """Enable or disable a feature flag."""
        self.features.set_enabled(key, enabled)

    def apply_features(self, changes: dict[str, bool]) -> list[str]:
        """Apply multiple feature changes at once."""
        return self.features.apply_many(changes)

    def get_available_sources(self) -> list[str]:
        """Return available document sources from the vector store."""
        retriever = self._initialize_retriever()
        if retriever is None:
            return []
        return retriever.get_available_sources()

    def set_rag_sources(self, sources: list[str] | None) -> bool:
        """Set which document sources to use for RAG (None = all)."""
        if not self.features.is_enabled("rag"):
            return False
        self.selected_sources = sources
        return True

    def get_rag_stats(self) -> dict[str, int | list[str]]:
        """Return vector store statistics."""
        if self._retriever is not None:
            return self._retriever.get_stats()
        return get_store_stats(self.config)

    def get_rag_status(self) -> dict[str, bool | list[str] | None]:
        """Return current RAG state, enabled status, and selected sources."""
        return {
            "enabled": self.features.is_enabled("rag"),
            "selected_sources": self.selected_sources,
            "available_sources": self.get_available_sources(),
        }

    def stream_reply(self) -> Iterator[str]:
        """Yield reply tokens, appending the full assistant message at the end."""
        stream = self.runtime.create_chat_completion(self.messages, stream=True)
        parts: list[str] = []
        for token in self.runtime.iter_stream_text(stream):
            parts.append(token)
            yield token
        self.messages.append(
            {"role": "assistant", "content": "".join(parts).strip()}
        )

    def full_reply(self) -> str:
        """Generate a complete (non-streamed) reply and append it."""
        response = self.runtime.create_chat_completion(self.messages, stream=False)
        reply = response["choices"][0]["message"]["content"].strip()
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    @property
    def model_name(self) -> str:
        return self.config.model.chat_model.name

    @property
    def compute_backend(self) -> ComputeBackend:
        return self.runtime.compute_backend

    def active_features(self) -> list[str]:
        return self.features.active_features()

    def features_summary(self) -> str:
        return self.features.summary()
