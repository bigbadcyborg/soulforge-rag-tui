"""Friendly error formatting and safe I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def safe_json_load(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Load JSON from disk; return (data, error_message)."""
    if not path.exists():
        return None, f"File not found: {path}"
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as error:
        return None, f"Invalid JSON: {error}"
    except OSError as error:
        return None, str(error)
    if not isinstance(data, dict):
        return None, "Root value is not a JSON object"
    return data, None


def format_startup_error(exc: Exception) -> str:
    """Map common startup failures to actionable messages."""
    if isinstance(exc, FileNotFoundError):
        message = str(exc).strip()
        lines = [f"Startup error: {message}", ""]
        lower = message.lower()
        if "config" in lower and "yaml" in lower:
            lines.append("Remediation:")
            lines.append("  • Create config.yaml at the project root")
            lines.append("  • Copy from a working install or the repository template")
        elif "embedding" in lower or "embed" in lower:
            lines.append("Remediation:")
            lines.append("  • Set model.embeddingModelPath in config.yaml")
            lines.append("  • Or disable RAG: features.rag: false")
        elif "model" in lower or "gguf" in lower:
            lines.append("Remediation:")
            lines.append("  • Place your GGUF file under models/")
            lines.append("  • Update model.chatModelPath in config.yaml")
        else:
            lines.append("Remediation:")
            lines.append("  • Run /diagnostics after startup for a full check")
            lines.append("  • Run /config to review resolved paths")
        return "\n".join(lines)

    if isinstance(exc, ImportError):
        return (
            f"Startup error: {exc}\n\n"
            "Remediation:\n"
            "  • Activate the project virtual environment\n"
            "  • Install dependencies: pip install -r requirements.txt\n"
            "  • For GPU: install llama-cpp-python with CUDA support"
        )

    return (
        f"Startup error: {exc}\n\n"
        "Remediation:\n"
        "  • Run /diagnostics for a full system check\n"
        "  • Run /config to review configuration\n"
        "  • Check logs/soulforge.log for details"
    )
