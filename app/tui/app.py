"""SoulForge Textual TUI (Iteration 2).

A scrollable chat history, an input box, and a status bar. Model loading and
token generation run on worker threads so the UI stays responsive; widget
updates are marshalled back to the UI thread via ``call_from_thread``.
"""

from __future__ import annotations

from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Input

from app.core.chat_controller import ChatController
from app.core.config import load_config
from app.rag.retriever import RetrievedChunk
from app.tui.widgets import ChatMessage, RagSelectionModal, StatusBar

HELP_TEXT = """Available commands:
  /help          Show this help
  /status        Show model and active features
  /rag [all|doc1,doc2,...]
                 Toggle RAG and select documents (e.g., /rag all or /rag doc1.txt,doc2.txt)
  /reload-soul   Reload SOUL.md and rebuild the system prompt
  /exit          Quit the app

Type anything else to chat. Ctrl+Q also quits."""


class SoulForgeApp(App):
    """Terminal chat interface for the local GGUF model."""

    CSS_PATH = "styles.tcss"
    TITLE = "SoulForge TUI"

    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, controller: ChatController) -> None:
        super().__init__()
        self.controller = controller
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

    def on_mount(self) -> None:
        self.status_bar.set_model(self.controller.model_name)
        self.status_bar.set_features(self.controller.features_summary())
        self.status_bar.set_state("Loading model...")
        self._write_message(
            "system",
            "Welcome to SoulForge TUI. Loading the model, please wait...\n"
            "Type /help for commands.",
        )
        self._load_models()

    # --- workers -------------------------------------------------------------

    @work(thread=True, exclusive=True, group="load")
    def _load_models(self) -> None:
        try:
            self.controller.load()
        except Exception as error:  # noqa: BLE001 - surface any load failure
            self.call_from_thread(self._on_load_failed, str(error))
            return
        self.call_from_thread(self._on_models_loaded)

    @work(thread=True, exclusive=True, group="generation")
    def _generate(self, user_input: str) -> None:
        try:
            chunks = self.controller.add_user_turn(user_input)
            if self.controller.config.features.show_sources and chunks:
                self.call_from_thread(self._write_sources, chunks)

            message: ChatMessage = self.call_from_thread(
                self._new_assistant_message
            )

            if self.controller.config.features.streaming:
                for token in self.controller.stream_reply():
                    self.call_from_thread(message.append, token)
                    self.call_from_thread(self._scroll_to_end)
            else:
                reply = self.controller.full_reply()
                self.call_from_thread(message.set_text, reply)
                self.call_from_thread(self._scroll_to_end)
        except Exception as error:  # noqa: BLE001 - surface generation failure
            self.call_from_thread(
                self._write_message, "system", f"Generation error: {error}"
            )
        finally:
            self.call_from_thread(self._generation_done)

    # --- worker callbacks (UI thread) ---------------------------------------

    def _on_models_loaded(self) -> None:
        self.models_ready = True
        self.status_bar.set_state("Ready")
        self.prompt.disabled = False
        self.prompt.placeholder = "Type a message, or /help"
        self.prompt.focus()
        self._write_message("system", "Model loaded. Ready to chat.")

    def _on_load_failed(self, error: str) -> None:
        self.status_bar.set_state("Load failed")
        self._write_message(
            "system",
            f"Failed to load model: {error}\n"
            "Fix config.yaml and restart, or press Ctrl+Q to quit.",
        )

    def _generation_done(self) -> None:
        self.status_bar.set_state("Ready")
        self.prompt.disabled = False
        self.prompt.focus()
        self._scroll_to_end()

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

    def _status_text(self) -> str:
        return (
            f"Model: {self.controller.model_name}\n"
            f"Active features: {self.controller.features_summary()}\n"
            f"State: {'ready' if self.models_ready else 'loading'}"
        )

    def _handle_rag_command(self, args: str) -> None:
        """Handle /rag command: toggle RAG and optionally select documents."""
        if not args:
            # Show interactive modal for document selection
            available = self.controller.get_available_sources()
            if not available:
                if not self.controller.rag_enabled:
                    self.controller.toggle_rag()
                self._write_message(
                    "system",
                    "No documents found in the vector store.\nRun ingestDocs.py to index documents."
                )
            else:
                # Push the modal and handle result asynchronously
                modal = RagSelectionModal(available)
                self.app.push_screen(modal, self._handle_rag_modal_result)
        elif args.lower() == "all":
            # Enable RAG with all documents
            if not self.controller.rag_enabled:
                self.controller.toggle_rag()
            available = self.controller.get_available_sources()
            if not available:
                self._write_message("system", "No documents found in the vector store.")
            else:
                self.controller.set_rag_sources(available)
                self._write_message(
                    "system",
                    f"RAG enabled using all {len(available)} document(s):\n  " + "\n  ".join(available)
                )
        else:
            # Enable RAG with specific documents
            requested = [doc.strip() for doc in args.split(",")]
            available = self.controller.get_available_sources()
            valid_docs = [doc for doc in requested if doc in available]
            invalid_docs = [doc for doc in requested if doc not in available]
            
            if not valid_docs:
                self._write_message(
                    "system",
                    f"None of the requested documents found.\nAvailable: {', '.join(available) if available else 'none'}"
                )
                return
            
            if not self.controller.rag_enabled:
                self.controller.toggle_rag()
            self.controller.set_rag_sources(valid_docs)
            msg = f"RAG enabled using {len(valid_docs)} document(s):\n  " + "\n  ".join(valid_docs)
            if invalid_docs:
                msg += f"\n\nNot found: {', '.join(invalid_docs)}\nAvailable: {', '.join(available)}"
            self._write_message("system", msg)

    def _handle_rag_modal_result(self, selected: list[str] | None) -> None:
        """Handle the result from the RAG selection modal."""
        if selected is None:
            # User cancelled
            self._write_message("system", "RAG selection cancelled.")
            return

        # Enable RAG and set selected sources
        if not self.controller.rag_enabled:
            self.controller.toggle_rag()
        
        self.controller.set_rag_sources(selected)
        
        if len(selected) == len(self.controller.get_available_sources()):
            msg = f"RAG enabled using all documents."
        else:
            msg = f"RAG enabled using {len(selected)} document(s):\n  " + "\n  ".join(selected)
        self._write_message("system", msg)

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
            self._write_message("system", HELP_TEXT)
        elif command == "/status":
            self._write_message("system", self._status_text())
        elif command == "/rag":
            self._handle_rag_command(args)
        elif command == "/reload-soul":
            if not self.models_ready:
                self._write_message("system", "Model is still loading, please wait.")
                return
            self.controller.reload_soul()
            self._write_message("system", "SOUL.md reloaded.")
        else:
            self._write_message(
                "system", f"Unknown command: {command}. Type /help."
            )


def run_tui(config_path: str | Path | None = None) -> None:
    config = load_config(config_path)
    controller = ChatController(config)
    SoulForgeApp(controller).run()
