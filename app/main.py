"""SoulForge entry point.

By default this launches the Textual TUI (Iteration 2). Pass ``--cli`` to use
the plain terminal loop, which is handy for headless environments or quick
debugging. Both front ends drive the same :class:`ChatController`.

    python -m app.main          # TUI
    python -m app.main --cli    # plain terminal loop
"""

from __future__ import annotations

import argparse
import json

from app.core.chat_controller import ChatController
from app.core.commands import format_help_text
from app.core.config import FEATURE_DISPLAY_NAMES, load_config
from app.core.diagnostics import format_diagnostics_view, run_startup_diagnostics
from app.memory.memory_manager import SECTION_FILENAMES, SECTION_KEYS
from app.rag.retriever import Retriever
from app.utils.guards import format_startup_error
from app.utils.logging import get_logger, setup_logging

EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit"}


def _log_startup_report(report) -> None:
    """Write startup diagnostics summary to the log file."""
    logger = get_logger("startup")
    if report.has_errors:
        logger.error(
            "Startup diagnostics: %d error(s), %d warning(s)",
            report.error_count,
            report.warning_count,
        )
    elif report.has_warnings:
        logger.warning(
            "Startup diagnostics: %d warning(s)",
            report.warning_count,
        )
    else:
        logger.info("Startup diagnostics: all checks passed")
    logger.debug(format_diagnostics_view(report))


def _print_startup_issues(report) -> None:
    """Print startup warnings and errors to stdout before model load."""
    if not report.has_errors and not report.has_warnings:
        return
    for check in report.checks:
        if check.status in ("error", "warn"):
            tag = "ERROR" if check.status == "error" else "WARN"
            print(f"[startup {tag}] {check.name}: {check.message}")
            if check.remediation:
                print(f"  → {check.remediation}")
    print()


def bootstrap(config_path=None):
    """Load config, init logging, and run startup diagnostics."""
    config = load_config(config_path)
    setup_logging(config)
    report = run_startup_diagnostics(config)
    _log_startup_report(report)
    return config, report


def _handle_tools_cli(controller: ChatController, args: str) -> None:
    if not controller.features.is_enabled("tools"):
        print("Tools are disabled. Run /features tools on first.")
        return
    if not args.strip():
        print(controller.get_tools_status())
        return
    parts = args.split(maxsplit=1)
    sub = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""

    if sub == "test":
        name_parts = rest.split(maxsplit=1)
        if not name_parts:
            print("Usage: /tools test <name> '<json args>'")
            return
        name = name_parts[0]
        raw_json = name_parts[1] if len(name_parts) > 1 else "{}"
        try:
            tool_args = json.loads(raw_json)
        except json.JSONDecodeError as error:
            print(f"Invalid JSON: {error}")
            return
        if not isinstance(tool_args, dict):
            print("Args must be a JSON object.")
            return
        result = controller.run_tool_test(name, tool_args)
        status = "OK" if result.success else "FAILED"
        print(f"[{status}] {result.summary(4000)}")
        return

    if sub == "add-shell":
        if not rest.strip():
            print("Usage: /tools add-shell <command prefix>")
            return
        print(controller.add_shell_allowlist_entry(rest.strip()))
        return

    if sub == "allowlist":
        allowlist = controller.config.tools.shell_allowlist
        if not allowlist:
            print("shellAllowlist is empty.")
        else:
            print("shellAllowlist:")
            for entry in allowlist:
                print(f"  - {entry}")
        return

    print(f"Unknown /tools subcommand: {sub}")
    print("Usage: /tools | /tools test <name> '<json>' | /tools add-shell <cmd> | /tools allowlist")


def _handle_model_cli(controller: ChatController, args: str) -> None:
    stripped = args.strip()
    if not stripped:
        print(f"Current model: {controller.model_name}")
        return

    if stripped.lower() == "list":
        print(controller.format_model_list())
        return

    parts = stripped.split()
    if parts[0].lower() == "add":
        if len(parts) < 2:
            print("Usage: /model add <path> [switch]")
            return
        switch_after = len(parts) >= 3 and parts[-1].lower() == "switch"
        source = " ".join(parts[1:-1] if switch_after else parts[1:])
        try:
            print("Importing model...")
            
            def on_progress(copied: int, total: int) -> None:
                pct = int((copied / total) * 100) if total else 0
                gb_copied = copied / (1024**3)
                gb_total = total / (1024**3)
                print(f"\rImporting: {pct}% ({gb_copied:.1f} GB / {gb_total:.1f} GB)", end="", flush=True)

            message = controller.import_chat_model(
                source, 
                switch_after=switch_after,
                on_progress=on_progress,
            )
            print()  # newline after progress
        except (OSError, ValueError, FileNotFoundError) as error:
            print(f"\nModel import failed: {error}")
            return
        print(message)
        return

    try:
        print("Switching model...")
        name = controller.switch_chat_model(stripped)
    except (OSError, ValueError, FileNotFoundError) as error:
        print(f"Model switch failed: {error}")
        return
    print(f"Switched to model: {name}")


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


def _handle_memory_clear_cli(controller: ChatController) -> None:
    controller.clear_all_memory()
    print("Cleared user.md, memory.md, and session.md.")


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


def _handle_session_list_cli(controller: ChatController) -> None:
    print(controller.list_sessions_view())


def _handle_session_save_cli(controller: ChatController, args: str) -> None:
    if not args.strip():
        print("Usage: /session-save <title>")
        return
    outcome = controller.save_session_direct(args.strip())
    print(outcome.message)


def _handle_session_load_cli(controller: ChatController, args: str) -> None:
    if not args.strip():
        print("Usage: /session-load <id>")
        print(controller.list_sessions_view())
        return
    outcome = controller.load_session_direct(args.strip())
    print(outcome.message)


def _handle_session_summary_cli(controller: ChatController) -> None:
    result = controller.run_session_summary()
    print(result.message)
    if result.success and result.summary:
        print("\n--- Summary ---")
        print(result.summary)


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
    elif command == "/health":
        print(controller.run_health_check())
    elif command == "/diagnostics":
        print(controller.run_diagnostics())
    elif command == "/config":
        print(controller.get_config_view())
    elif command == "/tutorial":
        print("The tutorial wizard is available in TUI mode.")
        print("Run: python -m app.main")
    elif command == "/features":
        _handle_features_cli(controller, args)
    elif command == "/model":
        _handle_model_cli(controller, args)
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
    elif command == "/memory-clear":
        _handle_memory_clear_cli(controller)
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
    elif command == "/session-list":
        _handle_session_list_cli(controller)
    elif command == "/session-save":
        _handle_session_save_cli(controller, args)
    elif command == "/session-load":
        _handle_session_load_cli(controller, args)
    elif command == "/session-summary":
        _handle_session_summary_cli(controller)
    elif command in ("/tools", "/tool"):
        _handle_tools_cli(controller, args)
    elif command == "/tools-log":
        print(controller.get_tool_log_view())
    elif command == "/tool-approve":
        call_id = args.strip()
        if not call_id:
            print("Usage: /tool-approve <call_id>")
            print(controller.get_tools_status())
        else:
            outcome = controller.approve_tool_call(call_id)
            print(outcome.message)
    elif command == "/tool-reject":
        call_id = args.strip()
        if not call_id:
            print("Usage: /tool-reject <call_id>")
        else:
            outcome = controller.reject_tool_call(call_id)
            print(outcome.message)
    else:
        print(f"Unknown command: {command}. Type /help for available commands.")

    return True


def run_cli() -> None:
    """Original plain terminal loop, kept as a lightweight fallback."""
    config, startup_report = bootstrap()
    _print_startup_issues(startup_report)
    controller = ChatController(config)
    try:
        controller.load()
    except Exception as error:  # noqa: BLE001
        print(format_startup_error(error))
        raise SystemExit(1) from error

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
            parts: list[str] = []
            for token in controller.stream_reply():
                print(token, end="", flush=True)
                parts.append(token)
            print()
            raw_reply = getattr(controller, "_pending_raw_reply", "") or "".join(parts)
        else:
            raw_reply = controller.full_reply()
            print(f"\nBot: {raw_reply}")

        tool_turn = controller.finalize_assistant_reply(raw_reply)
        if tool_turn.parse_error:
            print(f"\n[System] Tool parse warning: {tool_turn.parse_error}")
        for result in tool_turn.auto_results:
            status = "ok" if result.success else "failed"
            print(
                f"\n[System] Tool result: {result.name} ({status}) — "
                f"{result.summary()}"
            )
        for pending in tool_turn.pending:
            print(
                f"\n[System] Tool proposed: {pending.call.name} — "
                f"approval required (id {pending.call_id}). "
                "Run /tool-approve or /tool-reject."
            )

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
        print(format_startup_error(error))
        raise SystemExit(1) from error
    except ImportError as error:
        print(format_startup_error(error))
        raise SystemExit(1) from error
    except Exception as error:  # noqa: BLE001
        print(format_startup_error(error))
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
