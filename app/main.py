"""SoulForge entry point.

By default this launches the Textual TUI (Iteration 2). Pass ``--cli`` to use
the plain terminal loop, which is handy for headless environments or quick
debugging. Both front ends drive the same :class:`ChatController`.

    python -m app.main          # TUI
    python -m app.main --cli    # plain terminal loop
"""

from __future__ import annotations

import argparse

from app.core.chat_controller import ChatController
from app.core.config import load_config

EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit"}

CLI_HELP = """Available commands:
  /help          Show this help
  /status        Show model and active features
  /rag [all|doc1,doc2,...]
                 Toggle RAG and select documents (e.g., /rag all or /rag doc1.txt,doc2.txt)
  /reload-soul   Reload SOUL.md and rebuild the system prompt
  /exit or /quit Exit the chatbot

Type anything else to chat."""


def _handle_cli_command(controller: ChatController, cmd: str) -> bool:
    """Handle CLI commands. Return True if should continue, False if should exit."""
    parts = cmd.split(maxsplit=1)
    command = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    if command in ("/exit", "/quit", "exit", "quit"):
        return False
    elif command == "/help":
        print(CLI_HELP)
    elif command == "/status":
        print(f"Model: {controller.model_name}")
        print(f"Active features: {controller.features_summary()}")
    elif command == "/rag":
        if not args:
            # Toggle RAG without arguments
            enabled = controller.toggle_rag()
            if enabled:
                available = controller.get_available_sources()
                if not available:
                    print("RAG enabled, but no documents found in the vector store.\nRun ingestDocs.py to index documents.")
                else:
                    controller.set_rag_sources(available)
                    print(f"RAG enabled. Using all {len(available)} document(s):")
                    for doc in available:
                        print(f"  - {doc}")
            else:
                print("RAG disabled.")
        elif args.lower() == "all":
            # Enable RAG with all documents
            if not controller.rag_enabled:
                controller.toggle_rag()
            available = controller.get_available_sources()
            if not available:
                print("No documents found in the vector store.")
            else:
                controller.set_rag_sources(available)
                print(f"RAG enabled using all {len(available)} document(s):")
                for doc in available:
                    print(f"  - {doc}")
        else:
            # Enable RAG with specific documents
            requested = [doc.strip() for doc in args.split(",")]
            available = controller.get_available_sources()
            valid_docs = [doc for doc in requested if doc in available]
            invalid_docs = [doc for doc in requested if doc not in available]
            
            if not valid_docs:
                print(f"None of the requested documents found.\nAvailable: {', '.join(available) if available else 'none'}")
                return True
            
            if not controller.rag_enabled:
                controller.toggle_rag()
            controller.set_rag_sources(valid_docs)
            print(f"RAG enabled using {len(valid_docs)} document(s):")
            for doc in valid_docs:
                print(f"  - {doc}")
            if invalid_docs:
                print(f"\nNot found: {', '.join(invalid_docs)}")
    elif command == "/reload-soul":
        controller.reload_soul()
        print("SOUL.md reloaded.")
    else:
        print(f"Unknown command: {command}. Type /help for available commands.")

    return True


def run_cli() -> None:
    """Original plain terminal loop, kept as a lightweight fallback."""
    config = load_config()
    controller = ChatController(config)
    controller.load()

    print(f"\nModel: {controller.model_name}")
    print(f"Active features: {controller.features_summary()}")
    print("Local RAG chatbot started. Type '/help' for commands or 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            if not _handle_cli_command(controller, user_input):
                break
            continue

        if user_input.lower() in EXIT_COMMANDS:
            break

        chunks = controller.add_user_turn(user_input)
        if config.features.show_sources and chunks:
            print("\nSources:")
            for chunk in chunks:
                print(f"  - {chunk.source} (chunk {chunk.chunk_index})")

        if config.features.streaming:
            print("\nBot: ", end="", flush=True)
            for token in controller.stream_reply():
                print(token, end="", flush=True)
            print()
        else:
            print(f"\nBot: {controller.full_reply()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SoulForge local chatbot")
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run the plain terminal loop instead of the TUI.",
    )
    args = parser.parse_args()

    try:
        if args.cli:
            run_cli()
        else:
            from app.tui.app import run_tui

            run_tui()
    except FileNotFoundError as error:
        print(f"\nStartup error: {error}")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
