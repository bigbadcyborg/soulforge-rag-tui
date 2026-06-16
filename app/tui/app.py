"""SoulForge Textual TUI (Iteration 2+).

A scrollable chat history, an input box, and a status bar. Model loading and
token generation run on worker threads so the UI stays responsive; widget
updates are marshalled back to the UI thread via ``call_from_thread``.
"""

from __future__ import annotations

from pathlib import Path

from app.utils.guards import format_startup_error
from textual import work
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Input

from app.core.chat_controller import ChatController
from app.core.commands import format_help_text
from app.core.compute_backend import UNKNOWN
from app.core.config import FEATURE_DISPLAY_NAMES
from app.main import bootstrap
from app.rag.retriever import RetrievedChunk, Retriever
from app.memory.memory_manager import SECTION_KEYS
from app.tui.widgets import (
    ChatMessage,
    DiagnosticsModal,
    FeatureToggleModal,
    MemoryEditModal,
    MemoryReviewModal,
    MemoryViewerModal,
    RagSelectionModal,
    SourcesModal,
    StatusBar,
    SkillViewerModal,
    SkillDetailModal,
    SkillCreateModal,
    SkillCrystallizeModal,
    SkillCrystallizeEditModal,
    CuratorReviewModal,
    KanbanBoardModal,
    TaskDetailModal,
    TaskCreateModal,
    TaskSuggestionModal,
    SessionBrowserModal,
    SessionDetailModal,
    SessionSaveModal,
)


class SoulForgeApp(App):
    """Terminal chat interface for the local GGUF model."""

    CSS_PATH = "styles.tcss"
    TITLE = "SoulForge TUI"

    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(
        self,
        controller: ChatController,
        startup_report=None,
    ) -> None:
        super().__init__()
        self.controller = controller
        self.startup_report = startup_report
        self.models_ready = False

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="chat-view")
        yield Input(placeholder="Loading model...", id="prompt", disabled=True)
        yield StatusBar()

    @property
    def chat_view(self) -> VerticalScroll:
        return self.query_one("#chat-view", VerticalScroll)

    @property
    def prompt(self) -> Input:
        return self.query_one("#prompt", Input)

    @property
    def status_bar(self) -> StatusBar:
        return self.query_one(StatusBar)

    def _refresh_features(self) -> None:
        self.status_bar.set_features(self.controller.features_summary())

    def on_mount(self) -> None:
        self.status_bar.set_model(self.controller.model_name)
        self._refresh_features()
        self.status_bar.set_state("Loading model...")
        self.status_bar.set_compute(UNKNOWN)
        self._write_message(
            "system",
            "Welcome to SoulForge TUI. Loading the model, please wait...\n"
            "Type /help for commands.",
        )
        if self.startup_report is not None:
            issues = self.startup_report.error_count + self.startup_report.warning_count
            if issues:
                self._write_message(
                    "system",
                    f"{issues} setup issue(s) detected — run /diagnostics for details.",
                )
        self._load_models()

    # --- workers -------------------------------------------------------------

    @work(thread=True, exclusive=True, group="load")
    def _load_models(self) -> None:
        try:
            self.controller.load()
        except Exception as error:  # noqa: BLE001 - surface any load failure
            self.call_from_thread(
                self._on_load_failed, format_startup_error(error)
            )
            return
        self.call_from_thread(self._on_models_loaded)

    @work(thread=True, exclusive=True, group="generation")
    def _generate(self, user_input: str) -> None:
        try:
            chunks = self.controller.add_user_turn(user_input)
            if self.controller.features.is_enabled("show_sources") and chunks:
                self.call_from_thread(self._write_sources, chunks)

            message: ChatMessage = self.call_from_thread(
                self._new_assistant_message
            )

            if self.controller.features.is_enabled("streaming"):
                for token in self.controller.stream_reply():
                    self.call_from_thread(message.append, token)
                    self.call_from_thread(self._scroll_to_end)
            else:
                reply = self.controller.full_reply()
                self.call_from_thread(message.set_text, reply)
                self.call_from_thread(self._scroll_to_end)

            self.call_from_thread(self.status_bar.set_state, "Reviewing memory...")
            review = self.controller.complete_turn()
            if review.message:
                self.call_from_thread(self._write_message, "system", review.message)
            if review.has_suggestion and self.controller.pending_suggestion is not None:
                self.call_from_thread(self._open_memory_review_modal)
        except Exception as error:  # noqa: BLE001 - surface generation failure
            self.call_from_thread(
                self._write_message, "system", f"Generation error: {error}"
            )
        finally:
            self.call_from_thread(self._generation_done)

    @work(thread=True, exclusive=True, group="ingest")
    def _run_ingest(self) -> None:
        def on_progress(name: str, method: str, current: int, total: int) -> None:
            self.call_from_thread(
                self.status_bar.set_state,
                f"Ingest {current}/{total}: {name} ({method})",
            )

        try:
            result = self.controller.run_ingest(on_progress=on_progress)
            lines = [result.summary()]
            for note in result.errors:
                lines.append(f"Note: {note}")
            for skipped in result.skipped:
                lines.append(f"Skipped: {skipped}")
            if not self.controller.rag_enabled:
                lines.append("Tip: run /rag on to enable retrieval.")
            self.call_from_thread(self._write_message, "system", "\n".join(lines))
        except Exception as error:  # noqa: BLE001
            self.call_from_thread(
                self._write_message, "system", f"Ingest failed: {error}"
            )
        finally:
            self.call_from_thread(self._ingest_done)

    # --- worker callbacks (UI thread) ---------------------------------------

    def _on_models_loaded(self) -> None:
        self.models_ready = True
        self.status_bar.set_state("Ready")
        self.status_bar.set_compute(self.controller.compute_backend)
        self.prompt.disabled = False
        self.prompt.placeholder = "Type a message, or /help"
        self.prompt.focus()
        self._write_message("system", "Model loaded. Ready to chat.")

    def _on_load_failed(self, error: str) -> None:
        self.status_bar.set_state("Load failed")
        self._write_message(
            "system",
            f"{error}\n\n"
            "Try /diagnostics and /config, then restart or press Ctrl+Q to quit.",
        )

    def _generation_done(self) -> None:
        self.status_bar.set_state("Ready")
        self.prompt.disabled = False
        self.prompt.focus()
        self._scroll_to_end()

    def _ingest_done(self) -> None:
        self.status_bar.set_state("Ready")
        self.prompt.disabled = False
        self.prompt.focus()
        self._refresh_features()

    # --- helpers (UI thread) -------------------------------------------------

    def _write_message(self, role: str, text: str) -> ChatMessage:
        message = ChatMessage(role, text)
        self.chat_view.mount(message)
        self._scroll_to_end()
        return message

    def _new_assistant_message(self) -> ChatMessage:
        return self._write_message("assistant", "")

    def _write_sources(self, chunks: list[RetrievedChunk]) -> None:
        lines = ["Sources:"]
        for chunk in chunks:
            lines.append(f"  - {chunk.source} (chunk {chunk.chunk_index})")
        self._write_message("system", "\n".join(lines))

    def _scroll_to_end(self) -> None:
        self.chat_view.scroll_end(animate=False)

    def _rebuild_chat_view(self) -> None:
        """Rebuild visible chat from controller.messages after session load."""
        self.chat_view.remove_children()
        for message in self.controller.get_conversation_messages():
            role = str(message.get("role", "system"))
            text = self.controller.message_for_display(message)
            if text.strip():
                self._write_message(role, text)

    def _status_text(self) -> str:
        stats = self.controller.get_rag_stats()

        rag_info = ""
        if stats.get("chunk_count", 0) or stats.get("sources"):
            rag_info = (
                f"\nRAG index: {stats.get('chunk_count', 0)} chunk(s), "
                f"{len(stats.get('sources', []))} source(s)"
            )

        return (
            f"Model: {self.controller.model_name}\n"
            f"Active features: {self.controller.features_summary()}\n"
            f"Compute: {self.controller.compute_backend.label} "
            f"({self.controller.compute_backend.detail})\n"
            f"Memory turns: {self.controller.turn_count} "
            f"(review every {self.controller.config.memory.update_every_turns})\n"
            f"State: {'ready' if self.models_ready else 'loading'}"
            f"{rag_info}"
        )

    def _handle_rag_command(self, args: str) -> None:
        """Handle /rag command: toggle RAG and optionally select documents."""
        arg = args.strip().lower()

        if arg == "on":
            self.controller.enable_rag()
            available = self.controller.get_available_sources()
            if not available:
                self._write_message(
                    "system",
                    "RAG enabled, but no documents found.\nRun /ingest to index docs/.",
                )
            else:
                self._write_message(
                    "system",
                    f"RAG enabled. Using all {len(available)} document(s).",
                )
            self._refresh_features()
            return

        if arg == "off":
            self.controller.disable_rag()
            self._write_message("system", "RAG disabled.")
            self._refresh_features()
            return

        if not args:
            available = self.controller.get_available_sources()
            if not available:
                if not self.controller.rag_enabled:
                    self.controller.toggle_rag()
                self._write_message(
                    "system",
                    "No documents found in the vector store.\nRun /ingest to index documents.",
                )
            else:
                modal = RagSelectionModal(available)
                self.app.push_screen(modal, self._handle_rag_modal_result)
            return

        if arg == "all":
            self.controller.enable_rag()
            available = self.controller.get_available_sources()
            if not available:
                self._write_message("system", "No documents found in the vector store.")
            else:
                self.controller.set_rag_sources(available)
                self._write_message(
                    "system",
                    f"RAG enabled using all {len(available)} document(s):\n  "
                    + "\n  ".join(available),
                )
            self._refresh_features()
            return

        requested = [doc.strip() for doc in args.split(",")]
        available = self.controller.get_available_sources()
        valid_docs = [doc for doc in requested if doc in available]
        invalid_docs = [doc for doc in requested if doc not in available]

        if not valid_docs:
            self._write_message(
                "system",
                f"None of the requested documents found.\n"
                f"Available: {', '.join(available) if available else 'none'}",
            )
            return

        self.controller.enable_rag(valid_docs)
        msg = (
            f"RAG enabled using {len(valid_docs)} document(s):\n  "
            + "\n  ".join(valid_docs)
        )
        if invalid_docs:
            msg += f"\n\nNot found: {', '.join(invalid_docs)}\nAvailable: {', '.join(available)}"
        self._write_message("system", msg)
        self._refresh_features()

    def _handle_rag_modal_result(self, selected: list[str] | None) -> None:
        """Handle the result from the RAG selection modal."""
        if selected is None:
            self._write_message("system", "RAG selection cancelled.")
            return

        self.controller.enable_rag(selected)

        if len(selected) == len(self.controller.get_available_sources()):
            msg = "RAG enabled using all documents."
        else:
            msg = f"RAG enabled using {len(selected)} document(s):\n  " + "\n  ".join(
                selected
            )
        self._write_message("system", msg)
        self._refresh_features()

    def _handle_features_command(self, args: str) -> None:
        """Handle /features command: open modal or list flags."""
        if args.lower() == "list":
            self._write_message(
                "system",
                "Feature flags:\n" + self.controller.features.format_list(),
            )
            return

        if args:
            parts = args.split()
            if len(parts) == 2 and parts[1].lower() in ("on", "off"):
                key, state = parts[0], parts[1].lower() == "on"
                try:
                    self.controller.set_feature(key, state)
                    resolved = self.controller.features._resolve_key(key)
                    label = FEATURE_DISPLAY_NAMES[resolved]
                except KeyError as error:
                    self._write_message("system", str(error))
                    return
                self._write_message(
                    "system",
                    f"Feature '{label}' set to {'on' if state else 'off'}.",
                )
                self._refresh_features()
                return

            self._write_message(
                "system",
                "Usage: /features | /features list | /features <name> on|off",
            )
            return

        modal = FeatureToggleModal(self.controller.features.as_dict())
        self.app.push_screen(modal, self._handle_features_modal_result)

    def _handle_features_modal_result(self, selected: dict[str, bool] | None) -> None:
        """Apply feature toggles chosen in the modal."""
        if selected is None:
            self._write_message("system", "Feature toggle cancelled.")
            return

        changed = self.controller.apply_features(selected)
        if not changed:
            self._write_message("system", "No feature changes.")
            return

        labels = [FEATURE_DISPLAY_NAMES.get(key, key) for key in changed]
        self._write_message(
            "system",
            "Updated features: " + ", ".join(labels) + "\n(saved to config.yaml)",
        )
        self._refresh_features()

    def _handle_ingest_command(self) -> None:
        if not self.models_ready:
            self._write_message("system", "Model is still loading, please wait.")
            return
        self.status_bar.set_state("Ingesting...")
        self.prompt.disabled = True
        self._run_ingest()

    def _handle_sources_command(self) -> None:
        chunks = self.controller.last_retrieved_chunks
        modal = SourcesModal(chunks)
        self.app.push_screen(modal)

    def _handle_memory_command(self) -> None:
        self.controller.reload_memory()
        modal = MemoryViewerModal(self.controller.get_memory_view())
        self.app.push_screen(modal)

    def _handle_memory_edit_command(self, args: str) -> None:
        section = args.strip().lower().split(maxsplit=1)[0] if args.strip() else ""
        if not section:
            self._write_message(
                "system",
                "Usage: /memory-edit <user|memory|session>\n"
                "  user    — stable user facts\n"
                "  memory  — durable project memory\n"
                "  session — short-term session notes",
            )
            return
        if section not in SECTION_KEYS:
            self._write_message(
                "system",
                f"Unknown section '{section}'. Use: user, memory, or session.",
            )
            return

        limits = self.controller.memory_manager.limits()
        initial = self.controller.memory_manager.read_raw(section)
        modal = MemoryEditModal(section, initial, limits[section])
        self.app.push_screen(modal, self._handle_memory_edit_result)

    def _handle_memory_edit_result(self, result: tuple[str, str] | None) -> None:
        if result is None:
            self._write_message("system", "Memory edit cancelled.")
            return

        section, text = result
        try:
            truncated = self.controller.save_memory(section, text)
        except ValueError as error:
            self._write_message("system", str(error))
            return

        filename = f"{section}.md"
        message = f"Saved {filename}."
        if truncated:
            message += " Content was truncated to fit the character limit."
        self._write_message("system", message)

    def _handle_memory_on_command(self) -> None:
        self.controller.enable_memory()
        self._refresh_features()
        self._write_message("system", "Memory injection enabled.")

    def _handle_memory_off_command(self) -> None:
        self.controller.disable_memory()
        self._refresh_features()
        self._write_message("system", "Memory injection disabled.")

    def _open_memory_review_modal(self) -> None:
        suggestion = self.controller.pending_suggestion
        if suggestion is None:
            self._write_message("system", "No pending memory suggestion.")
            return
        limit = self.controller.memory_manager.limits()[suggestion.section]
        modal = MemoryReviewModal(suggestion, limit)
        self.app.push_screen(modal, self._handle_memory_review_result)

    def _handle_memory_review_result(self, action: str | None) -> None:
        if action is None:
            return
        if action == "reject":
            self.controller.reject_memory_suggestion()
            self._write_message("system", "Memory suggestion rejected.")
            return
        if action == "accept":
            self._accept_pending_memory_suggestion()
            return
        if action == "edit":
            suggestion = self.controller.pending_suggestion
            if suggestion is None:
                return
            limit = self.controller.memory_manager.limits()[suggestion.section]
            modal = MemoryEditModal(
                suggestion.section,
                suggestion.proposed_content,
                limit,
            )
            self.app.push_screen(modal, self._handle_memory_review_edit_result)

    def _handle_memory_review_edit_result(self, result: tuple[str, str] | None) -> None:
        if result is None:
            self._write_message("system", "Memory edit cancelled. Suggestion still pending.")
            return
        _section, text = result
        self._accept_pending_memory_suggestion(text)

    def _accept_pending_memory_suggestion(self, content: str | None = None) -> None:
        try:
            _, was_compacted = self.controller.accept_memory_suggestion(content)
        except ValueError as error:
            self._write_message("system", str(error))
            return

        message = "Memory suggestion saved."
        if was_compacted:
            message += " Content was compacted to fit the character limit."
        self._write_message("system", message)

    def _handle_memory_review_command(self) -> None:
        if self.controller.pending_suggestion is None:
            self._write_message("system", "No pending memory suggestion.")
            return
        self._open_memory_review_modal()

    def _handle_memory_accept_command(self) -> None:
        if self.controller.pending_suggestion is None:
            self._write_message("system", "No pending memory suggestion.")
            return
        self._accept_pending_memory_suggestion()

    def _handle_skills_command(self) -> None:
        active_skills = self.controller.skill_manager.list_skills(status="active")
        archived_skills = self.controller.skill_manager.list_skills(status="archived")
        pending_message = ""
        if self.controller.pending_skill_suggestion is not None:
            name = self.controller.pending_skill_suggestion.name
            pending_message = f"Suggested skill pending: {name} (run /crystallize or /skill-accept)"
        if not active_skills and archived_skills:
            pending_message = (
                (pending_message + " " if pending_message else "")
                + f"No active skills ({len(archived_skills)} archived)."
            )
        modal = SkillViewerModal(active_skills, archived_skills, pending_message)
        self.app.push_screen(modal, self._handle_skills_modal_result)

    def _open_skill_crystallize_modal(self) -> None:
        suggestion = self.controller.pending_skill_suggestion
        if suggestion is None:
            self._write_message("system", "No pending skill suggestion.")
            return
        modal = SkillCrystallizeModal(suggestion)
        self.app.push_screen(modal, self._handle_skill_crystallize_result)

    def _handle_skill_crystallize_result(self, action: str | None) -> None:
        if action is None:
            return
        if action == "reject":
            self.controller.reject_skill_suggestion()
            self._write_message("system", "Skill suggestion rejected.")
            return
        if action == "accept":
            self._accept_pending_skill_suggestion()
            return
        if action == "edit":
            suggestion = self.controller.pending_skill_suggestion
            if suggestion is None:
                return
            modal = SkillCrystallizeEditModal(suggestion)
            self.app.push_screen(modal, self._handle_skill_crystallize_edit_result)

    def _handle_skill_crystallize_edit_result(self, content: str | None) -> None:
        if content is None:
            self._write_message("system", "Skill edit cancelled. Suggestion still pending.")
            return
        self._accept_pending_skill_suggestion(content)

    def _accept_pending_skill_suggestion(self, content: str | None = None) -> None:
        name = (
            self.controller.pending_skill_suggestion.name
            if self.controller.pending_skill_suggestion is not None
            else "skill"
        )
        try:
            self.controller.accept_skill_suggestion(content)
        except ValueError as error:
            self._write_message("system", str(error))
            return
        self._write_message("system", f"Skill '{name}' saved.")
        self._refresh_features()

    def _handle_success_command(self, args: str) -> None:
        result = self.controller.mark_workflow_success(args.strip())
        if result.message:
            self._write_message("system", result.message)
        if result.should_open_modal and self.controller.pending_skill_suggestion is not None:
            self._open_skill_crystallize_modal()

    def _handle_crystallize_command(self, args: str) -> None:
        fingerprint = args.strip() or None
        result = self.controller.crystallize_workflow(fingerprint)
        if result.message:
            self._write_message("system", result.message)
        if result.has_suggestion and self.controller.pending_skill_suggestion is not None:
            self._open_skill_crystallize_modal()

    def _handle_skill_accept_command(self) -> None:
        if self.controller.pending_skill_suggestion is None:
            self._write_message("system", "No pending skill suggestion.")
            return
        self._accept_pending_skill_suggestion()

    def _handle_skill_reject_command(self) -> None:
        if self.controller.pending_skill_suggestion is None:
            self._write_message("system", "No pending skill suggestion.")
            return
        self.controller.reject_skill_suggestion()
        self._write_message("system", "Skill suggestion rejected.")

    def _open_curator_review_modal(self) -> None:
        findings = self.controller._visible_curator_findings()
        if not findings:
            self._write_message(
                "system",
                "No pending curator findings. Run /curator-review first.",
            )
            return
        modal = CuratorReviewModal(findings[0], 0, len(findings))
        self.app.push_screen(modal, self._handle_curator_review_result)

    def _handle_curator_review_result(self, result: tuple[str, str] | None) -> None:
        if result is None:
            return
        action, finding_id = result
        if action == "approve":
            outcome = self.controller.accept_curator_finding(finding_id)
            self._write_message("system", outcome.message)
            if outcome.success:
                self._refresh_features()
        elif action == "ignore":
            self.controller.dismiss_curator_finding(finding_id)
            self._write_message("system", "Curator finding ignored.")

        remaining = self.controller._visible_curator_findings()
        if remaining:
            self._open_curator_review_modal()

    def _handle_curator_command(self) -> None:
        self._open_curator_review_modal()

    def _handle_curator_review_command(self) -> None:
        self.status_bar.set_state("Running curator review...")
        result = self.controller.run_curator_review()
        if result.message:
            self._write_message("system", result.message)
        self.status_bar.set_state("Ready")
        if result.has_findings:
            self._open_curator_review_modal()

    def _handle_curator_archive_command(self, args: str) -> None:
        outcome = self.controller.archive_skill_direct(args)
        self._write_message("system", outcome.message)
        if outcome.success:
            self._refresh_features()

    def _handle_curator_compact_command(self, args: str) -> None:
        self.status_bar.set_state("Compacting skill...")
        result = self.controller.compact_skill_direct(args)
        if result.message:
            self._write_message("system", result.message)
        self.status_bar.set_state("Ready")
        if result.has_findings:
            self._open_curator_review_modal()

    def _handle_skill_restore_command(self, args: str) -> None:
        outcome = self.controller.restore_skill_direct(args)
        self._write_message("system", outcome.message)
        if outcome.success:
            self._refresh_features()

    def _handle_tasks_command(self) -> None:
        if not self.controller.features.is_enabled("kanban"):
            self._write_message("system", "Kanban is disabled. Run /features kanban on.")
            return
        board = self.controller.task_manager.list_board()
        pending = self.controller._visible_task_suggestions()
        modal = KanbanBoardModal(board, pending_count=len(pending))
        self.app.push_screen(modal, self._handle_tasks_modal_result)

    def _handle_tasks_modal_result(self, result) -> None:
        if result == "new":
            self.app.push_screen(TaskCreateModal(), self._handle_task_create_result)
        elif result == "review":
            self._open_task_suggestion_modal()
        elif isinstance(result, tuple) and result[0] == "view":
            task_id = result[1]
            located = self.controller.task_manager.get_task(task_id)
            if located:
                column, task = located
                modal = TaskDetailModal(column, task)
                self.app.push_screen(modal, self._handle_task_detail_result)
        elif result is None:
            return
        else:
            self._handle_tasks_command()

    def _handle_task_detail_result(self, result) -> None:
        if result is None:
            self._handle_tasks_command()
            return
        if isinstance(result, tuple) and result[0] == "move":
            _, task_id, column = result
            outcome = self.controller.move_task_direct(task_id, column)
            self._write_message("system", outcome.message)
            if outcome.success:
                self._refresh_features()
        elif isinstance(result, tuple) and result[0] == "delete":
            _, task_id = result
            outcome = self.controller.delete_task_direct(task_id)
            self._write_message("system", outcome.message)
            if outcome.success:
                self._refresh_features()
        self._handle_tasks_command()

    def _handle_task_create_result(self, result: dict[str, str] | None) -> None:
        if result:
            outcome = self.controller.create_task_direct(
                result["title"],
                result.get("description", ""),
            )
            self._write_message("system", outcome.message)
            if outcome.success:
                self._refresh_features()
        self._handle_tasks_command()

    def _handle_task_new_modal_result(self, result: dict[str, str] | None) -> None:
        if result:
            outcome = self.controller.create_task_direct(
                result["title"],
                result.get("description", ""),
            )
            self._write_message("system", outcome.message)
            if outcome.success:
                self._refresh_features()

    def _handle_task_new_command(self, args: str) -> None:
        if not self.controller.features.is_enabled("kanban"):
            self._write_message("system", "Kanban is disabled. Run /features kanban on.")
            return
        if args.strip():
            outcome = self.controller.create_task_direct(args.strip())
            self._write_message("system", outcome.message)
            if outcome.success:
                self._refresh_features()
        else:
            self.app.push_screen(TaskCreateModal(), self._handle_task_new_modal_result)

    def _handle_task_move_command(self, args: str) -> None:
        parts = args.split(maxsplit=1)
        if len(parts) < 2:
            self._write_message(
                "system",
                "Usage: /task-move <id> <column>  (backlog, in_progress, blocked, done)",
            )
            return
        outcome = self.controller.move_task_direct(parts[0], parts[1])
        self._write_message("system", outcome.message)
        if outcome.success:
            self._refresh_features()

    def _handle_task_done_command(self, args: str) -> None:
        if not args.strip():
            self._write_message("system", "Usage: /task-done <id>")
            return
        outcome = self.controller.move_task_direct(args.strip(), "done")
        self._write_message("system", outcome.message)
        if outcome.success:
            self._refresh_features()

    def _handle_task_delete_command(self, args: str) -> None:
        if not args.strip():
            self._write_message("system", "Usage: /task-delete <id>")
            return
        outcome = self.controller.delete_task_direct(args.strip())
        self._write_message("system", outcome.message)
        if outcome.success:
            self._refresh_features()

    def _open_task_suggestion_modal(self) -> None:
        suggestions = self.controller._visible_task_suggestions()
        if not suggestions:
            self._write_message(
                "system",
                "No pending task suggestions. Run /task-suggest first.",
            )
            return
        modal = TaskSuggestionModal(suggestions[0], 0, len(suggestions))
        self.app.push_screen(modal, self._handle_task_suggestion_result)

    def _handle_task_suggestion_result(self, result: tuple[str, str] | None) -> None:
        if result is None:
            return
        action, suggestion_id = result
        if action == "approve":
            outcome = self.controller.accept_task_suggestion(suggestion_id)
            self._write_message("system", outcome.message)
            if outcome.success:
                self._refresh_features()
        elif action == "ignore":
            self.controller.dismiss_task_suggestion(suggestion_id)
            self._write_message("system", "Task suggestion ignored.")

        remaining = self.controller._visible_task_suggestions()
        if remaining:
            self._open_task_suggestion_modal()

    def _handle_task_suggest_command(self) -> None:
        if not self.controller.features.is_enabled("kanban"):
            self._write_message("system", "Kanban is disabled. Run /features kanban on.")
            return
        self.status_bar.set_state("Analyzing conversation for task updates...")
        result = self.controller.run_task_suggest()
        if result.message:
            self._write_message("system", result.message)
        self.status_bar.set_state("Ready")
        if result.has_suggestions:
            self._open_task_suggestion_modal()

    def _handle_task_accept_command(self, args: str) -> None:
        suggestion_id = args.strip()
        if not suggestion_id:
            self._write_message("system", "Usage: /task-accept <suggestion_id>")
            self._write_message("system", self.controller.get_task_suggestions_review())
            return
        outcome = self.controller.accept_task_suggestion(suggestion_id)
        self._write_message("system", outcome.message)
        if outcome.success:
            self._refresh_features()

    def _handle_task_reject_command(self, args: str) -> None:
        suggestion_id = args.strip()
        if not suggestion_id:
            self._write_message("system", "Usage: /task-reject <suggestion_id>")
            return
        self.controller.dismiss_task_suggestion(suggestion_id)
        self._write_message("system", "Task suggestion rejected.")

    def _default_session_title(self) -> str:
        from app.sessions.session_manager import title_from_messages

        return title_from_messages(self.controller.get_conversation_messages())

    def _handle_session_list_command(self) -> None:
        sessions = self.controller.session_manager.list_sessions()
        modal = SessionBrowserModal(sessions)
        self.app.push_screen(modal, self._handle_session_browser_result)

    def _handle_session_browser_result(self, result) -> None:
        if result == "save":
            self._open_session_save_modal()
        elif isinstance(result, tuple) and result[0] == "view":
            session_id = result[1]
            sessions = self.controller.session_manager.list_sessions()
            meta = next((item for item in sessions if item.id == session_id), None)
            if meta:
                modal = SessionDetailModal(meta)
                self.app.push_screen(modal, self._handle_session_detail_result)
            else:
                self._handle_session_list_command()
        elif result is None:
            return
        else:
            self._handle_session_list_command()

    def _handle_session_detail_result(self, result) -> None:
        if result is None:
            self._handle_session_list_command()
            return
        if isinstance(result, tuple) and result[0] == "load":
            outcome = self.controller.load_session_direct(result[1])
            self._write_message("system", outcome.message)
            if outcome.success:
                self._rebuild_chat_view()
                self._refresh_features()
            return
        if isinstance(result, tuple) and result[0] == "delete":
            outcome = self.controller.delete_session_direct(result[1])
            self._write_message("system", outcome.message)
        self._handle_session_list_command()

    def _open_session_save_modal(self) -> None:
        default = self._default_session_title()
        modal = SessionSaveModal(default)
        self.app.push_screen(modal, self._handle_session_save_result)

    def _handle_session_save_result(self, title: str | None) -> None:
        if title is None:
            self._handle_session_list_command()
            return
        outcome = self.controller.save_session_direct(title)
        self._write_message("system", outcome.message)
        self._handle_session_list_command()

    def _handle_session_save_command(self, args: str) -> None:
        if args.strip():
            outcome = self.controller.save_session_direct(args.strip())
            self._write_message("system", outcome.message)
        else:
            self._open_session_save_modal_from_command()

    def _open_session_save_modal_from_command(self) -> None:
        default = self._default_session_title()
        modal = SessionSaveModal(default)
        self.app.push_screen(modal, self._handle_session_save_command_result)

    def _handle_session_save_command_result(self, title: str | None) -> None:
        if title is None:
            return
        outcome = self.controller.save_session_direct(title)
        self._write_message("system", outcome.message)

    def _handle_session_load_command(self, args: str) -> None:
        if not args.strip():
            self._write_message("system", "Usage: /session-load <id>")
            return
        outcome = self.controller.load_session_direct(args.strip())
        self._write_message("system", outcome.message)
        if outcome.success:
            self._rebuild_chat_view()
            self._refresh_features()

    def _handle_session_summary_command(self) -> None:
        self.status_bar.set_state("Generating session summary...")
        result = self.controller.run_session_summary()
        self._write_message("system", result.message)
        self.status_bar.set_state("Ready")
        if result.success:
            self._refresh_features()

    def _handle_skills_modal_result(self, result: Any) -> None:
        if result == "new":
            self.app.push_screen(SkillCreateModal(), self._handle_skill_create_result)
        elif isinstance(result, tuple) and result[0] == "view":
            skill_name = result[1]
            content = self.controller.skill_manager.get_skill_content(skill_name)
            if content:
                is_archived = not self.controller.skill_manager.is_active(skill_name)
                modal = SkillDetailModal(skill_name, content, archived=is_archived)
                self.app.push_screen(modal, self._handle_skills_modal_result)
        elif isinstance(result, tuple) and result[0] == "archive":
            skill_name = result[1]
            if self.controller.skill_manager.archive_skill(skill_name):
                self._write_message("system", f"Skill '{skill_name}' archived.")
                if self.controller.features.is_enabled("skills"):
                    self.controller.reload_soul()
            else:
                self._write_message("system", f"Failed to archive skill '{skill_name}'.")
        elif isinstance(result, tuple) and result[0] == "restore":
            skill_name = result[1]
            outcome = self.controller.restore_skill_direct(skill_name)
            self._write_message("system", outcome.message)
            if outcome.success:
                self._refresh_features()

    def _handle_skill_create_result(self, result: dict[str, str] | None) -> None:
        if result:
            if self.controller.skill_manager.create_skill(
                result["name"], result["description"], result["content"]
            ):
                self._write_message("system", f"Skill '{result['name']}' created.")
                if self.controller.features.is_enabled("skills"):
                    self.controller.reload_soul()
            else:
                self._write_message("system", f"Failed to create skill '{result['name']}' (it may already exist).")

    def _handle_memory_reject_command(self) -> None:
        if self.controller.pending_suggestion is None:
            self._write_message("system", "No pending memory suggestion.")
            return
        self.controller.reject_memory_suggestion()
        self._write_message("system", "Memory suggestion rejected.")

    # --- input handling ------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        self.prompt.value = ""
        if not text:
            return

        if text.startswith("/"):
            self._handle_command(text)
            return

        if not self.models_ready:
            self._write_message("system", "Model is still loading, please wait.")
            return

        self._write_message("user", text)
        self.status_bar.set_state("Generating...")
        self.prompt.disabled = True
        self._generate(text)

    def _handle_command(self, text: str) -> None:
        parts = text.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if command in ("/exit", "/quit"):
            self.exit()
        elif command == "/help":
            help_text = format_help_text(args, self.controller.config)
            if not args.strip():
                help_text += "\n\nCtrl+Q also quits."
            self._write_message("system", help_text)
        elif command == "/status":
            self._write_message("system", self._status_text())
        elif command == "/health":
            self._write_message("system", self.controller.run_health_check())
        elif command == "/diagnostics":
            self.push_screen(DiagnosticsModal(self.controller.run_diagnostics()))
        elif command == "/config":
            self.push_screen(DiagnosticsModal(self.controller.get_config_view(), title="Configuration"))
        elif command == "/ingest":
            self._handle_ingest_command()
        elif command == "/sources":
            self._handle_sources_command()
        elif command == "/rag":
            self._handle_rag_command(args)
        elif command == "/features":
            self._handle_features_command(args)
        elif command == "/reload-soul":
            if not self.models_ready:
                self._write_message("system", "Model is still loading, please wait.")
                return
            self.controller.reload_soul()
            self._write_message("system", "SOUL.md reloaded.")
        elif command == "/memory":
            self._handle_memory_command()
        elif command == "/memory-edit":
            self._handle_memory_edit_command(args)
        elif command == "/memory-on":
            self._handle_memory_on_command()
        elif command == "/memory-off":
            self._handle_memory_off_command()
        elif command == "/memory-review":
            self._handle_memory_review_command()
        elif command == "/memory-accept":
            self._handle_memory_accept_command()
        elif command == "/memory-reject":
            self._handle_memory_reject_command()
        elif command == "/skills":
            self._handle_skills_command()
        elif command == "/success":
            self._handle_success_command(args)
        elif command == "/crystallize":
            self._handle_crystallize_command(args)
        elif command == "/skill-accept":
            self._handle_skill_accept_command()
        elif command == "/skill-reject":
            self._handle_skill_reject_command()
        elif command == "/skill-restore":
            self._handle_skill_restore_command(args)
        elif command == "/curator":
            self._handle_curator_command()
        elif command == "/curator-review":
            self._handle_curator_review_command()
        elif command == "/curator-archive":
            self._handle_curator_archive_command(args)
        elif command == "/curator-compact":
            self._handle_curator_compact_command(args)
        elif command == "/tasks":
            self._handle_tasks_command()
        elif command == "/task-new":
            self._handle_task_new_command(args)
        elif command == "/task-move":
            self._handle_task_move_command(args)
        elif command == "/task-done":
            self._handle_task_done_command(args)
        elif command == "/task-delete":
            self._handle_task_delete_command(args)
        elif command == "/task-suggest":
            self._handle_task_suggest_command()
        elif command == "/task-accept":
            self._handle_task_accept_command(args)
        elif command == "/task-reject":
            self._handle_task_reject_command(args)
        elif command == "/session-list":
            self._handle_session_list_command()
        elif command == "/session-save":
            self._handle_session_save_command(args)
        elif command == "/session-load":
            self._handle_session_load_command(args)
        elif command == "/session-summary":
            self._handle_session_summary_command()
        else:
            self._write_message(
                "system", f"Unknown command: {command}. Type /help."
            )


def run_tui(config_path: str | Path | None = None) -> None:
    config, startup_report = bootstrap(config_path)
    controller = ChatController(config)
    SoulForgeApp(controller, startup_report=startup_report).run()
