"""Semantic validation for loaded AppConfig."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core.compute_backend import detect_compute_backend
from app.core.config import PROJECT_ROOT, AppConfig


@dataclass
class ConfigIssue:
    field: str
    severity: str  # "error" | "warning"
    message: str
    remediation: str = ""


def _dir_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        test_file = path / ".write_test"
        test_file.write_text("", encoding="utf-8")
        test_file.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def validate_config(config: AppConfig) -> list[ConfigIssue]:
    """Return semantic config issues without loading models."""
    issues: list[ConfigIssue] = []

    chat_path = config.model.chat_model
    if not chat_path.exists():
        issues.append(
            ConfigIssue(
                field="model.chatModelPath",
                severity="error",
                message=f"Chat model not found: {chat_path}",
                remediation="Place the GGUF file at that path or update config.yaml.",
            )
        )
    elif chat_path.stat().st_size == 0:
        issues.append(
            ConfigIssue(
                field="model.chatModelPath",
                severity="error",
                message=f"Chat model file is empty: {chat_path}",
                remediation="Replace with a valid GGUF download.",
            )
        )

    embed_path = config.model.embedding_model
    if config.features.rag:
        if not embed_path.exists():
            issues.append(
                ConfigIssue(
                    field="model.embeddingModelPath",
                    severity="error",
                    message=f"Embedding model not found: {embed_path}",
                    remediation="Add the embedding GGUF or set features.rag: false.",
                )
            )
    elif not embed_path.exists():
        issues.append(
            ConfigIssue(
                field="model.embeddingModelPath",
                severity="warning",
                message=f"Embedding model not found: {embed_path}",
                remediation="Required only when RAG is enabled.",
            )
        )

    if config.model.context_size <= 0:
        issues.append(
            ConfigIssue(
                field="model.contextSize",
                severity="error",
                message="contextSize must be positive.",
                remediation="Set model.contextSize to a value like 8192.",
            )
        )

    if config.rag.top_k <= 0:
        issues.append(
            ConfigIssue(
                field="rag.topK",
                severity="error",
                message="rag.topK must be positive.",
                remediation="Set rag.topK to at least 1.",
            )
        )

    if config.memory.update_every_turns <= 0:
        issues.append(
            ConfigIssue(
                field="memory.updateEveryTurns",
                severity="error",
                message="memory.updateEveryTurns must be positive.",
                remediation="Set memory.updateEveryTurns to at least 1.",
            )
        )

    for name, limit in (
        ("memory.maxUserChars", config.memory.max_user_chars),
        ("memory.maxMemoryChars", config.memory.max_memory_chars),
        ("memory.maxSessionChars", config.memory.max_session_chars),
    ):
        if limit <= 0:
            issues.append(
                ConfigIssue(
                    field=name,
                    severity="error",
                    message=f"{name} must be positive.",
                    remediation=f"Increase {name} in config.yaml.",
                )
            )

    backend = detect_compute_backend(config)
    if config.model.gpu_layers != 0 and backend.mode != "gpu":
        issues.append(
            ConfigIssue(
                field="model.gpuLayers",
                severity="warning",
                message=f"GPU offload requested but running on CPU: {backend.detail}",
                remediation="Rebuild llama-cpp-python with CUDA or set gpuLayers: 0.",
            )
        )

    soul_path = PROJECT_ROOT / "SOUL.md"
    if not soul_path.exists():
        issues.append(
            ConfigIssue(
                field="SOUL.md",
                severity="warning",
                message="SOUL.md not found at project root.",
                remediation="Create SOUL.md or disable features.soul.",
            )
        )

    if config.features.rag:
        docs_dir = config.rag.docs_dir
        if not docs_dir.exists():
            issues.append(
                ConfigIssue(
                    field="rag.docsPath",
                    severity="warning",
                    message=f"Docs directory not found: {docs_dir}",
                    remediation="Create docs/ and add files, then run /ingest.",
                )
            )
        elif not any(docs_dir.iterdir()):
            issues.append(
                ConfigIssue(
                    field="rag.docsPath",
                    severity="warning",
                    message=f"Docs directory is empty: {docs_dir}",
                    remediation="Add documents and run /ingest.",
                )
            )

    for label, path in (
        ("skills.activePath", config.skills.active_dir),
        ("skills.archivedPath", config.skills.archived_dir),
        ("sessions.storePath", config.sessions.store_dir),
        ("memory.userFile", config.memory.user_path.parent),
    ):
        if not _dir_writable(path):
            issues.append(
                ConfigIssue(
                    field=label,
                    severity="error",
                    message=f"Cannot write to directory: {path}",
                    remediation="Check permissions or update the path in config.yaml.",
                )
            )

    return issues
