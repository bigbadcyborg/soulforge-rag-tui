"""Shared command help registry for TUI and CLI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.core.config import AppConfig


@dataclass(frozen=True)
class CommandHelp:
    category: str
    usage: str
    description: str


COMMANDS: tuple[CommandHelp, ...] = (
    CommandHelp(
        "General",
        "/help [topic]",
        "Show commands; use /help crystallize for the full skill workflow guide",
    ),
    CommandHelp("General", "/status", "Show model name, active features, and app state"),
    CommandHelp("General", "/exit, /quit", "Exit the application"),
    CommandHelp(
        "Features",
        "/features",
        "Open the feature toggle menu (TUI); changes auto-save to config.yaml",
    ),
    CommandHelp(
        "Features",
        "/features list",
        "List all feature flags and their on/off state",
    ),
    CommandHelp(
        "Features",
        "/features <name> on|off",
        "Toggle one feature (e.g. /features rag on, /features soul off)",
    ),
    CommandHelp(
        "RAG",
        "/ingest",
        "Index files from docs/ into ChromaDB (text + PDF/OCR)",
    ),
    CommandHelp(
        "RAG",
        "/sources",
        "View retrieved chunks from the last question",
    ),
    CommandHelp(
        "RAG",
        "/rag",
        "Toggle RAG or open document selection modal (TUI)",
    ),
    CommandHelp("RAG", "/rag on", "Enable RAG retrieval"),
    CommandHelp("RAG", "/rag off", "Disable RAG retrieval"),
    CommandHelp("RAG", "/rag all", "Enable RAG using all indexed documents"),
    CommandHelp(
        "RAG",
        "/rag doc1,doc2",
        "Enable RAG filtered to specific document names",
    ),
    CommandHelp(
        "Persona",
        "/reload-soul",
        "Reload SOUL.md persona into the system prompt",
    ),
    CommandHelp(
        "Memory",
        "/memory",
        "View user.md, memory.md, and session.md with character counts",
    ),
    CommandHelp(
        "Memory",
        "/memory-edit [name]",
        "Edit user, memory, or session (TUI editor; CLI: optional inline content)",
    ),
    CommandHelp(
        "Memory",
        "/memory-on",
        "Enable memory injection (equivalent to /features memory on)",
    ),
    CommandHelp(
        "Memory",
        "/memory-off",
        "Disable memory injection (equivalent to /features memory off)",
    ),
    CommandHelp(
        "Memory",
        "/memory-review",
        "Show pending memory update suggestion",
    ),
    CommandHelp(
        "Memory",
        "/memory-accept",
        "Save the pending suggestion to user.md or memory.md",
    ),
    CommandHelp(
        "Memory",
        "/memory-reject",
        "Discard the pending memory suggestion",
    ),
    CommandHelp(
        "Skills",
        "/skills",
        "Browse active skills, create manually, or archive (TUI modal)",
    ),
    CommandHelp(
        "Skills",
        "/success [note]",
        "Mark the recent chat workflow as successful (counts toward crystallization)",
    ),
    CommandHelp(
        "Skills",
        "/crystallize [fingerprint]",
        "Draft a reusable skill from a workflow that reached the success threshold",
    ),
    CommandHelp(
        "Skills",
        "/skill-accept",
        "Save the pending skill draft to app/skills/active/ (requires skills on)",
    ),
    CommandHelp(
        "Skills",
        "/skill-reject",
        "Discard the pending skill draft without saving",
    ),
    CommandHelp(
        "Skills",
        "/skill-restore <name>",
        "Restore an archived skill to app/skills/active/",
    ),
    CommandHelp(
        "Curator",
        "/curator",
        "Open curator review modal for pending findings (TUI)",
    ),
    CommandHelp(
        "Curator",
        "/curator-review",
        "Scan active skills for stale, bloated, or duplicate entries",
    ),
    CommandHelp(
        "Curator",
        "/curator-archive <skill>",
        "Archive a skill immediately (moves to skills/archived/)",
    ),
    CommandHelp(
        "Curator",
        "/curator-compact <skill>",
        "Draft a shorter version of a bloated skill for approval",
    ),
    CommandHelp(
        "Curator",
        "/curator-accept <id>",
        "Apply a curator finding by ID (CLI; TUI uses Approve in modal)",
    ),
    CommandHelp(
        "Curator",
        "/curator-ignore <id>",
        "Dismiss a curator finding by ID (CLI)",
    ),
    CommandHelp(
        "Kanban",
        "/tasks",
        "Open Kanban board modal (TUI) or print board (CLI)",
    ),
    CommandHelp(
        "Kanban",
        "/task-new [title]",
        "Create a task in Backlog (modal if no title in TUI)",
    ),
    CommandHelp(
        "Kanban",
        "/task-move <id> <column>",
        "Move a task (backlog, in_progress, blocked, done)",
    ),
    CommandHelp(
        "Kanban",
        "/task-done <id>",
        "Move a task to Done",
    ),
    CommandHelp(
        "Kanban",
        "/task-delete <id>",
        "Delete a task from the board",
    ),
    CommandHelp(
        "Kanban",
        "/task-suggest",
        "Suggest task updates from recent conversation (requires approval)",
    ),
    CommandHelp(
        "Kanban",
        "/task-accept <id>",
        "Apply a pending task suggestion by ID (CLI; TUI uses Approve in modal)",
    ),
    CommandHelp(
        "Kanban",
        "/task-reject <id>",
        "Dismiss a pending task suggestion by ID (CLI)",
    ),
    CommandHelp(
        "Sessions",
        "/session-list",
        "List saved conversations (TUI opens browser modal)",
    ),
    CommandHelp(
        "Sessions",
        "/session-save [title]",
        "Save current chat to app/sessions/ (modal in TUI if no title)",
    ),
    CommandHelp(
        "Sessions",
        "/session-load <id>",
        "Restore a saved conversation and rebuild the chat view",
    ),
    CommandHelp(
        "Sessions",
        "/session-summary",
        "LLM summary of current chat written to session.md for prompt injection",
    ),
)


def _skill_settings(config: AppConfig | None) -> tuple[int, int, bool]:
    if config is None:
        return 3, 3, False
    skills = config.skills
    return (
        skills.min_successful_repeats,
        skills.success_window_turns,
        skills.auto_create,
    )


def _skill_crystallization_summary(config: AppConfig | None = None) -> str:
    min_repeats, window_turns, auto_create = _skill_settings(config)
    auto_line = (
        "At threshold, a review modal opens automatically after /success."
        if auto_create
        else "At threshold, run /crystallize to open the review modal."
    )
    return (
        "Skill crystallization (quick guide):\n"
        "  Turn a repeated successful chat workflow into a reusable skill file.\n"
        "  1. Enable skills: /features skills on\n"
        "  2. Chat through a workflow (similar user messages each time)\n"
        f"  3. After each bot reply, run /success [optional note] "
        f"({min_repeats} times total; once per turn)\n"
        f"  4. {auto_line}\n"
        "  5. Review Trigger / Procedure / Validation, then Accept or Edit\n"
        "  6. Saved skills live in app/skills/active/ and inject when skills are on\n"
        f"  Fingerprint uses your last {window_turns} user messages. "
        "For the full walkthrough: /help crystallize"
    )


def format_crystallize_help_text(config: AppConfig | None = None) -> str:
    """Detailed help for /crystallize and the full skill crystallization workflow."""
    min_repeats, window_turns, auto_create = _skill_settings(config)
    auto_detail = (
        "When autoCreate is true in config.yaml, the app runs the crystallizer "
        "and opens the review modal automatically on the threshold /success."
        if auto_create
        else "When autoCreate is false (default), the threshold /success only "
        "reminds you to run /crystallize — nothing is drafted until you do."
    )
    return "\n".join(
        [
            "Skill crystallization — full guide",
            "",
            "What it does:",
            "  Repeatedly marking the same workflow as successful lets SoulForge",
            "  suggest a reusable skill (Trigger, Procedure, Validation) from your",
            "  chat. Nothing is saved until you Accept or run /skill-accept.",
            "",
            "Prerequisites:",
            "  • /features skills on  — required to save; marking /success works either way",
            "  • A repeatable workflow in chat (same kind of steps each time)",
            "",
            "Settings (config.yaml → skills:):",
            f"  • minSuccessfulRepeats: {min_repeats}  — /success marks needed before crystallize",
            f"  • successWindowTurns: {window_turns}  — user messages used for fingerprinting",
            f"  • autoCreate: {str(auto_create).lower()}  — {auto_detail}",
            "  • workflowLogPath — tracks counts in app/skills/workflow_log.json",
            "",
            "Step-by-step:",
            "  1. Chat normally. Example workflow:",
            "       You: I need to rebuild llama-cpp-python with CUDA in WSL",
            "       Bot:  (reply)",
            "       You: activate venv, export CUDA paths, build with GGML_CUDA",
            "       Bot:  (reply)",
            "",
            f"  2. Run /success [note] after a bot reply — e.g. /success rebuild cuda",
            "     Optional note becomes a hint for the skill name and summary.",
            "",
            "  3. Repeat similar chat + /success until the counter reaches "
            f"{min_repeats}/{min_repeats}.",
            "     • Each /success counts once per turn (same turn twice does not double-count).",
            "     • Use similar user phrasing so the fingerprint matches.",
            f"     • Only your last {window_turns} user messages are fingerprinted.",
            "",
            f"  4. When count reaches {min_repeats}:",
            "     • autoCreate true  → review modal opens after /success",
            "     • autoCreate false → system says to run /crystallize",
            "",
            "  5. /crystallize [fingerprint]",
            "     Drafts skill markdown from the logged workflow (LLM + validation).",
            "     Omit fingerprint to use the best eligible workflow from the log.",
            "",
            "  6. Review modal (TUI) or printed draft (CLI):",
            "     • Accept      — save to app/skills/active/<name>.md",
            "     • Edit        — change markdown, then save",
            "     • Reject      — discard draft (/skill-reject)",
            "",
            "  7. Verify: /skills lists the new skill; registry.json and workflow_log.json update.",
            "",
            "Related commands:",
            "  /success [note]     Mark workflow success (increments counter)",
            "  /crystallize        Draft pending skill from threshold workflow",
            "  /skill-accept       Save pending draft without modal",
            "  /skill-reject       Discard pending draft",
            "  /skills             Browse, create, or archive skills",
            "",
            "Tips:",
            "  • Lower minSuccessfulRepeats temporarily (e.g. 2) for faster testing.",
            "  • After Accept, the same workflow will not suggest again (crystallized_as in log).",
            "  • Saving requires skills on; enable before Accept if you marked success earlier.",
            "",
            "Back to all commands: /help",
        ]
    )


def format_sessions_help_text() -> str:
    return "\n".join(
        [
            "Session persistence — quick guide",
            "",
            "Saved conversations vs session.md:",
            "  • /session-save stores full chat history in app/sessions/<id>.json",
            "  • /session-summary writes a compact summary to app/memory/session.md",
            "  • session.md is injected when memory is enabled (short-term context)",
            "",
            "Workflow:",
            "  1. Chat normally",
            "  2. /session-save My project chat",
            "  3. Quit and restart later",
            "  4. /session-list then /session-load <id>",
            "  5. /session-summary before ending to capture notes in session.md",
            "",
            "Related commands:",
            "  /session-list      Browse saved sessions",
            "  /session-save    Persist current conversation",
            "  /session-load    Resume a saved conversation",
            "  /session-summary Generate and inject session summary",
            "",
            "Back to all commands: /help",
        ]
    )


def format_help_text(topic: str = "", config: AppConfig | None = None) -> str:
    """Format command help for /help or /help <topic>."""
    topic_key = topic.strip().lower()
    if topic_key in ("crystallize", "crystallization", "skills"):
        return format_crystallize_help_text(config)
    if topic_key in ("sessions", "session"):
        return format_sessions_help_text()

    lines = ["Available commands:", ""]
    current_category = ""
    for cmd in COMMANDS:
        if cmd.category != current_category:
            current_category = cmd.category
            lines.append(f"{current_category}:")
        lines.append(f"  {cmd.usage}")
        lines.append(f"    {cmd.description}")
        lines.append("")

    lines.append(_skill_crystallization_summary(config))
    lines.append("")
    lines.append("Type anything else to chat.")
    return "\n".join(lines).rstrip()
