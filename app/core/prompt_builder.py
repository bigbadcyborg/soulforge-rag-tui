"""Prompt builder: assembles prompts in the order defined by plan section 10.

Assembly order (when every feature is enabled):
    1. Core system rules
    2. SOUL.md
    3. user.md
    4. memory.md
    5. session.md summary
    6. Relevant skills        (added in a later iteration)
    7. Retrieved RAG context
    8. Current user message

The builder respects feature toggles: disabled sections are simply skipped.
The system prompt carries rules + persona + memory; retrieved context is
attached to the user turn so the model treats it as grounding for the question.
"""

from __future__ import annotations

from app.core.config import AppConfig
from app.memory.memory_manager import MemorySnapshot

# Kept intentionally empty so the persona in SOUL.md remains the dominant voice.
# Future iterations can populate baseline operating rules here.
CORE_SYSTEM_RULES = ""

MEMORY_GROUNDING_RULES = (
    "IMPORTANT - LOCAL MEMORY (READ-ONLY):\n"
    "The # User, # Memory, and # Session sections below are the complete contents "
    "of local memory files (user.md, memory.md, session.md). They are already in "
    "your context. The person chatting with you IS the user described in user.md. "
    "When they say 'my', 'I', or 'me', apply facts from user.md to them.\n\n"
    "You CANNOT write to, update, or save these files during chat. Never claim you saved, "
    "recorded, added, or updated their profile or memory. Periodically the system may "
    "suggest updates for your approval via /memory-review; nothing is saved until you "
    "run /memory-accept. For manual edits, use /memory-edit user (or memory or session).\n\n"
    "Rules for memory-backed answers:\n"
    "- If the answer appears in these sections, use ONLY that text. Do not "
    "supplement with general knowledge, famous people, interviews, or articles.\n"
    "- Do not invent past conversations, dates, or sources. You have no record "
    "of earlier chats except what is written in these sections and the current "
    "conversation above.\n"
    "- Do not pretend to read, open, or inspect files.\n"
    "- If these sections do not contain the answer, say it is not in memory yet."
)

DEFAULT_SYSTEM_PROMPT = "You are a helpful local chatbot."

TASK_GROUNDING_RULES = (
    "IMPORTANT - TASK BOARD (READ-ONLY):\n"
    "The # Active Tasks section below is the current Kanban board. You cannot "
    "modify it during chat. To persist task changes, the user runs /task-suggest "
    "and approves suggestions in the review modal."
)


class PromptBuilder:
    """Builds system prompts and user turns from toggled feature inputs."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def build_system_prompt(
        self,
        soul_text: str = "",
        memory: MemorySnapshot | None = None,
        skills: list[str] | None = None,
        task_summary: str = "",
    ) -> str:
        features = self.config.features
        sections: list[str] = []

        if CORE_SYSTEM_RULES.strip():
            sections.append(CORE_SYSTEM_RULES.strip())

        if features.soul and soul_text.strip():
            sections.append(soul_text.strip())

        if features.memory and memory is not None:
            memory_sections: list[str] = []
            if memory.user.strip():
                memory_sections.append(
                    f"# User (user.md)\n{memory.user.strip()}"
                )
            if memory.memory.strip():
                memory_sections.append(
                    f"# Memory (memory.md)\n{memory.memory.strip()}"
                )
            if memory.session.strip():
                memory_sections.append(
                    f"# Session (session.md)\n{memory.session.strip()}"
                )
            if memory_sections:
                sections.append(MEMORY_GROUNDING_RULES)
                sections.extend(memory_sections)

        if features.skills and skills:
            sections.append("# Relevant Skills")
            sections.extend(skills)

        if features.kanban and task_summary.strip():
            sections.append(TASK_GROUNDING_RULES)
            sections.append(f"# Active Tasks\n{task_summary.strip()}")

        if not sections:
            return DEFAULT_SYSTEM_PROMPT

        return "\n\n".join(sections)

    @staticmethod
    def _format_active_memory(memory: MemorySnapshot) -> str:
        """Compact memory block for per-turn user message injection."""
        parts: list[str] = []
        if memory.user.strip():
            parts.append(f"[user.md]\n{memory.user.strip()}")
        if memory.memory.strip():
            parts.append(f"[memory.md]\n{memory.memory.strip()}")
        if memory.session.strip():
            parts.append(f"[session.md]\n{memory.session.strip()}")
        return "\n\n".join(parts)

    def build_user_turn(
        self,
        user_input: str,
        context_text: str = "",
        use_rag: bool = True,
        memory: MemorySnapshot | None = None,
        use_memory: bool = False,
    ) -> str:
        """Combine memory, retrieved RAG context, and the user's question."""
        sections: list[str] = []

        if use_memory and memory is not None:
            mem_text = self._format_active_memory(memory)
            if mem_text:
                sections.append(
                    "ACTIVE MEMORY (read-only; authoritative for questions about "
                    "the user; 'my'/'I'/'me' refer to the user in user.md):\n"
                    f"{mem_text}\n"
                    "You cannot save new facts to these files. Never claim you updated "
                    "memory. To persist new info, tell the user to run /memory-edit."
                )

        if use_rag and context_text.strip():
            sections.append(
                "AVAILABLE CONTEXT FROM DOCUMENTS:\n"
                f"{context_text}\n\n"
                "When responding, consider the above context. If it's relevant and helps answer the question, "
                "incorporate it into your response and mention the source. If the context doesn't contain relevant information, "
                "answer based on your knowledge and acknowledge the documents didn't cover that topic."
            )

        sections.append(f"USER MESSAGE:\n{user_input}")
        return "\n\n".join(sections)
