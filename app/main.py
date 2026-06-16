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
from app.core.commands import format_help_text
from app.core.config import FEATURE_DISPLAY_NAMES, load_config
from app.memory.memory_manager import SECTION_FILENAMES, SECTION_KEYS
from app.rag.retriever import Retriever

EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit"}


def _handle_features_cli(controller: ChatController, args: str) -> None:
    if not args or args.lower() == "list":
        print("Feature flags:")
        print(controller.features.format_list())
        return

    parts = args.split()
    if len(parts) == 2 and parts[1].lower() in ("on", "off"):
        key, state = parts[0], parts[1].lower() == "on"
        try:
            controller.set_feature(key, state)
            resolved = controller.features._resolve_key(key)
            label = FEATURE_DISPLAY_NAMES[resolved]
        except KeyError as error:
            print(error)
            return
        print(f"Feature '{label}' set to {'on' if state else 'off'}. (saved to config.yaml)")
        return

    print("Usage: /features | /features list | /features <name> on|off")


def _handle_rag_cli(controller: ChatController, args: str) -> None:
    arg = args.strip().lower()

    if arg == "on":
        controller.enable_rag()
        available = controller.get_available_sources()
        if not available:
            print("RAG enabled, but no documents found.\nRun /ingest to index docs/.")
        else:
            print(f"RAG enabled. Using all {len(available)} document(s).")
        return

    if arg == "off":
        controller.disable_rag()
        print("RAG disabled.")
        return

    if not args:
        enabled = controller.toggle_rag()
        if enabled:
            available = controller.get_available_sources()
            if not available:
                print(
                    "RAG enabled, but no documents found in the vector store.\n"
                    "Run /ingest to index documents."
                )
            else:
                controller.set_rag_sources(available)
                print(f"RAG enabled. Using all {len(available)} document(s):")
                for doc in available:
                    print(f"  - {doc}")
        else:
            print("RAG disabled.")
        return

    if arg == "all":
        controller.enable_rag()
        available = controller.get_available_sources()
        if not available:
            print("No documents found in the vector store.")
        else:
            controller.set_rag_sources(available)
            print(f"RAG enabled using all {len(available)} document(s):")
            for doc in available:
                print(f"  - {doc}")
        return

    requested = [doc.strip() for doc in args.split(",")]
    available = controller.get_available_sources()
    valid_docs = [doc for doc in requested if doc in available]
    invalid_docs = [doc for doc in requested if doc not in available]

    if not valid_docs:
        print(
            f"None of the requested documents found.\n"
            f"Available: {', '.join(available) if available else 'none'}"
        )
        return

    controller.enable_rag(valid_docs)
    print(f"RAG enabled using {len(valid_docs)} document(s):")
    for doc in valid_docs:
        print(f"  - {doc}")
    if invalid_docs:
        print(f"\nNot found: {', '.join(invalid_docs)}")


def _handle_ingest_cli(controller: ChatController) -> None:
    def on_progress(name: str, method: str, current: int, total: int) -> None:
        print(f"[{current}/{total}] {name} ({method})")

    result = controller.run_ingest(on_progress=on_progress)
    print(result.summary())
    for note in result.errors:
        print(f"Note: {note}")
    for skipped in result.skipped:
        print(f"Skipped: {skipped}")
    if not controller.rag_enabled:
        print("Tip: run /rag on to enable retrieval.")


def _handle_sources_cli(controller: ChatController) -> None:
    print(Retriever.format_sources_detail(controller.last_retrieved_chunks))


def _handle_memory_cli(controller: ChatController) -> None:
    controller.reload_memory()
    print(controller.get_memory_view())


def _handle_memory_edit_cli(controller: ChatController, args: str) -> None:
    if not args.strip():
        mem = controller.config.memory
        print("Memory files:")
        print(f"  user:    {mem.user_path}")
        print(f"  memory:  {mem.memory_path}")
        print(f"  session: {mem.session_path}")
        print("\nUsage: /memory-edit <user|memory|session> [content]")
        return

    parts = args.split(maxsplit=1)
    section = parts[0].lower()
    if section not in SECTION_KEYS:
        print(f"Invalid section '{section}'. Use: user, memory, or session.")
        return

    if len(parts) == 1:
        mem = controller.config.memory
        paths = {
            "user": mem.user_path,
            "memory": mem.memory_path,
            "session": mem.session_path,
        }
        print(f"Path: {paths[section]}")
        print()
        print(controller.memory_manager.read_raw(section))
        print("\nEdit in an external editor, then run /memory to reload.")
        return

    try:
        truncated = controller.save_memory(section, parts[1])
    except ValueError as error:
        print(error)
        return

    filename = SECTION_FILENAMES[section]
    print(f"Saved {filename}.")
    if truncated:
        print("Warning: content was truncated to fit the character limit.")


def _handle_memory_review_cli(controller: ChatController) -> None:
    print(controller.get_memory_review())


def _handle_memory_accept_cli(controller: ChatController) -> None:
    if controller.pending_suggestion is None:
        print("No pending memory suggestion.")
        return
    try:
        _, was_compacted = controller.accept_memory_suggestion()
    except ValueError as error:
        print(error)
        return
    print("Memory suggestion saved.")
    if was_compacted:
        print("Note: content was compacted to fit the character limit.")


def _handle_memory_reject_cli(controller: ChatController) -> None:
    if controller.pending_suggestion is None:
        print("No pending memory suggestion.")
        return
    controller.reject_memory_suggestion()
    print("Memory suggestion rejected.")


def _handle_success_cli(controller: ChatController, args: str) -> None:
    result = controller.mark_workflow_success(args.strip())
    if result.message:
        print(result.message)
    if result.should_open_modal and controller.pending_skill_suggestion is not None:
        print(controller.get_skill_review())
        print("Run /skill-accept or /skill-reject.")


def _handle_crystallize_cli(controller: ChatController, args: str) -> None:
    fingerprint = args.strip() or None
    result = controller.crystallize_workflow(fingerprint)
    if result.message:
        print(result.message)
    if result.has_suggestion and controller.pending_skill_suggestion is not None:
        print()
        print(controller.get_skill_review())
        print("Run /skill-accept or /skill-reject.")


def _handle_skill_accept_cli(controller: ChatController) -> None:
    if controller.pending_skill_suggestion is None:
        print("No pending skill suggestion.")
        return
    try:
        controller.accept_skill_suggestion()
    except ValueError as error:
        print(error)
        return
    print("Skill suggestion saved.")


def _handle_skill_reject_cli(controller: ChatController) -> None:
    if controller.pending_skill_suggestion is None:
        print("No pending skill suggestion.")
        return
    controller.reject_skill_suggestion()
    print("Skill suggestion rejected.")


def _handle_skill_restore_cli(controller: ChatController, args: str) -> None:
    outcome = controller.restore_skill_direct(args)
    print(outcome.message)


def _handle_skills_cli(controller: ChatController) -> None:
    active = controller.skill_manager.list_skills(status="active")
    archived = controller.skill_manager.list_skills(status="archived")
    print("Active skills:")
    if not active:
        print("  (none)")
    for skill in active:
        print(f"  - {skill.get('name')}: {skill.get('description', '')}")
    print("Archived skills:")
    if not archived:
        print("  (none)")
    for skill in archived:
        print(f"  - {skill.get('name')}: {skill.get('description', '')}")
    if controller.pending_skill_suggestion is not None:
        print(
            f"\nPending suggestion: {controller.pending_skill_suggestion.name} "
            "(run /skill-accept or /skill-reject)"
        )


def _handle_curator_review_cli(controller: ChatController) -> None:
    result = controller.run_curator_review()
    if result.message:
        print(result.message)
    if result.has_findings:
        print()
        print(controller.get_curator_review())
        print("\nApprove: /curator-accept <finding_id>")
        print("Ignore:  /curator-ignore <finding_id>")


def _handle_curator_accept_cli(controller: ChatController, args: str) -> None:
    finding_id = args.strip()
    if not finding_id:
        print("Usage: /curator-accept <finding_id>")
        print(controller.get_curator_review())
        return
    outcome = controller.accept_curator_finding(finding_id)
    print(outcome.message)


def _handle_curator_ignore_cli(controller: ChatController, args: str) -> None:
    finding_id = args.strip()
    if not finding_id:
        print("Usage: /curator-ignore <finding_id>")
        return
    controller.dismiss_curator_finding(finding_id)
    print("Curator finding ignored.")


def _handle_curator_archive_cli(controller: ChatController, args: str) -> None:
    outcome = controller.archive_skill_direct(args)
    print(outcome.message)


def _handle_curator_compact_cli(controller: ChatController, args: str) -> None:
    result = controller.compact_skill_direct(args)
    if result.message:
        print(result.message)
    if result.has_findings and result.findings:
        finding = result.findings[0]
        print(f"\nFinding ID: {finding.finding_id}")
        print(f"Run /curator-accept {finding.finding_id} to save compacted content.")


def _handle_curator_cli(controller: ChatController) -> None:
    visible = controller._visible_curator_findings()
    if not visible:
        print("No pending curator findings. Run /curator-review first.")
        return
    print(controller.get_curator_review())


def _handle_tasks_cli(controller: ChatController) -> None:
    print(controller.get_board_view())
    pending = controller._visible_task_suggestions()
    if pending:
        print()
        print(controller.get_task_suggestions_review())
        print("\nApprove: /task-accept <suggestion_id>")
        print("Reject:  /task-reject <suggestion_id>")


def _handle_task_new_cli(controller: ChatController, args: str) -> None:
    if not args.strip():
        print("Usage: /task-new <title>  (TUI supports modal without title)")
        return
    outcome = controller.create_task_direct(args.strip())
    print(outcome.message)


def _handle_task_move_cli(controller: ChatController, args: str) -> None:
    parts = args.split(maxsplit=1)
    if len(parts) < 2:
        print("Usage: /task-move <id> <column>")
        return
    outcome = controller.move_task_direct(parts[0], parts[1])
    print(outcome.message)


def _handle_task_done_cli(controller: ChatController, args: str) -> None:
    if not args.strip():
        print("Usage: /task-done <id>")
        return
    outcome = controller.move_task_direct(args.strip(), "done")
    print(outcome.message)


def _handle_task_delete_cli(controller: ChatController, args: str) -> None:
    if not args.strip():
        print("Usage: /task-delete <id>")
        return
    outcome = controller.delete_task_direct(args.strip())
    print(outcome.message)


def _handle_task_suggest_cli(controller: ChatController) -> None:
    result = controller.run_task_suggest()
    if result.message:
        print(result.message)
    if result.has_suggestions:
        print()
        print(controller.get_task_suggestions_review())
        print("\nApprove: /task-accept <suggestion_id>")
        print("Reject:  /task-reject <suggestion_id>")


def _handle_task_accept_cli(controller: ChatController, args: str) -> None:
    suggestion_id = args.strip()
    if not suggestion_id:
        print("Usage: /task-accept <suggestion_id>")
        print(controller.get_task_suggestions_review())
        return
    outcome = controller.accept_task_suggestion(suggestion_id)
    print(outcome.message)


def _handle_task_reject_cli(controller: ChatController, args: str) -> None:
    suggestion_id = args.strip()
    if not suggestion_id:
        print("Usage: /task-reject <suggestion_id>")
        return
    controller.dismiss_task_suggestion(suggestion_id)
    print("Task suggestion rejected.")


def _handle_cli_command(controller: ChatController, cmd: str) -> bool:
    """Handle CLI commands. Return True if should continue, False if should exit."""
    parts = cmd.split(maxsplit=1)
    command = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    if command in ("/exit", "/quit", "exit", "quit"):
        return False
    elif command == "/help":
        print(format_help_text(args, controller.config))
    elif command == "/status":
        backend = controller.compute_backend
        print(f"Model: {controller.model_name}")
        print(f"Active features: {controller.features_summary()}")
        print(f"Compute: {backend.label} ({backend.detail})")
        print(
            f"Memory turns: {controller.turn_count} "
            f"(review every {controller.config.memory.update_every_turns})"
        )
        print(
            f"Skill crystallization threshold: "
            f"{controller.config.skills.min_successful_repeats} successes"
        )
    elif command == "/features":
        _handle_features_cli(controller, args)
    elif command == "/ingest":
        _handle_ingest_cli(controller)
    elif command == "/sources":
        _handle_sources_cli(controller)
    elif command == "/rag":
        _handle_rag_cli(controller, args)
    elif command == "/reload-soul":
        controller.reload_soul()
        print("SOUL.md reloaded.")
    elif command == "/memory":
        _handle_memory_cli(controller)
    elif command == "/memory-edit":
        _handle_memory_edit_cli(controller, args)
    elif command == "/memory-on":
        controller.enable_memory()
        print("Memory injection enabled.")
    elif command == "/memory-off":
        controller.disable_memory()
        print("Memory injection disabled.")
    elif command == "/memory-review":
        _handle_memory_review_cli(controller)
    elif command == "/memory-accept":
        _handle_memory_accept_cli(controller)
    elif command == "/memory-reject":
        _handle_memory_reject_cli(controller)
    elif command == "/skills":
        _handle_skills_cli(controller)
    elif command == "/success":
        _handle_success_cli(controller, args)
    elif command == "/crystallize":
        _handle_crystallize_cli(controller, args)
    elif command == "/skill-accept":
        _handle_skill_accept_cli(controller)
    elif command == "/skill-reject":
        _handle_skill_reject_cli(controller)
    elif command == "/skill-restore":
        _handle_skill_restore_cli(controller, args)
    elif command == "/curator":
        _handle_curator_cli(controller)
    elif command == "/curator-review":
        _handle_curator_review_cli(controller)
    elif command == "/curator-archive":
        _handle_curator_archive_cli(controller, args)
    elif command == "/curator-compact":
        _handle_curator_compact_cli(controller, args)
    elif command == "/curator-accept":
        _handle_curator_accept_cli(controller, args)
    elif command == "/curator-ignore":
        _handle_curator_ignore_cli(controller, args)
    elif command == "/tasks":
        _handle_tasks_cli(controller)
    elif command == "/task-new":
        _handle_task_new_cli(controller, args)
    elif command == "/task-move":
        _handle_task_move_cli(controller, args)
    elif command == "/task-done":
        _handle_task_done_cli(controller, args)
    elif command == "/task-delete":
        _handle_task_delete_cli(controller, args)
    elif command == "/task-suggest":
        _handle_task_suggest_cli(controller)
    elif command == "/task-accept":
        _handle_task_accept_cli(controller, args)
    elif command == "/task-reject":
        _handle_task_reject_cli(controller, args)
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
        if controller.features.is_enabled("show_sources") and chunks:
            print("\nSources:")
            for chunk in chunks:
                print(f"  - {chunk.source} (chunk {chunk.chunk_index})")

        if controller.features.is_enabled("streaming"):
            print("\nBot: ", end="", flush=True)
            for token in controller.stream_reply():
                print(token, end="", flush=True)
            print()
        else:
            print(f"\nBot: {controller.full_reply()}")

        review = controller.complete_turn()
        if review.message:
            print(f"\n[System] {review.message}")
            if review.has_suggestion and controller.pending_suggestion is not None:
                print("Run /memory-review, /memory-accept, or /memory-reject.")


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
