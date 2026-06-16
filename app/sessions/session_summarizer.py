"""Session summarizer: LLM-generated summary for session.md injection."""

from __future__ import annotations

from app.core.model_runtime import ModelRuntime
from app.memory.memory_reviewer import format_conversation_window

SUMMARY_COMPLETION_OVERRIDES = {
    "temperature": 0.2,
    "max_tokens": 1024,
    "stop": ["</s>"],
}


def _build_summary_prompt() -> str:
    return (
        "You are a session summarizer for a local chatbot. Review the conversation "
        "and write a concise summary for session.md.\n\n"
        "Format as short bullet points covering:\n"
        "- Main topics discussed\n"
        "- Decisions or conclusions reached\n"
        "- Open items or next steps\n\n"
        "Rules:\n"
        "- Use only facts from the conversation.\n"
        "- Do not invent people, tools, or events.\n"
        "- Keep it brief and scannable.\n"
        "- Reply with plain text only (no JSON, no markdown fences)."
    )


def generate_summary(
    runtime: ModelRuntime,
    messages: list[dict[str, str]],
) -> str:
    """Generate a plain-text session summary from conversation history."""
    window = format_conversation_window(messages, last_n_turns=20)
    llm_messages = [
        {"role": "system", "content": _build_summary_prompt()},
        {"role": "user", "content": f"CONVERSATION:\n{window}"},
    ]
    try:
        response = runtime.create_chat_completion(
            llm_messages,
            stream=False,
            **SUMMARY_COMPLETION_OVERRIDES,
        )
        raw = response["choices"][0]["message"]["content"]
    except Exception as error:  # noqa: BLE001
        print(f"[sessions] Summary generation failed: {error}")
        return ""

    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text
