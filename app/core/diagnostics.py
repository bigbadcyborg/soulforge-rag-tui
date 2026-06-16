"""Startup and runtime diagnostics for SoulForge."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from app.core.compute_backend import detect_compute_backend
from app.core.config import DEFAULT_CONFIG_PATH, AppConfig, features_to_yaml_dict
from app.core.config_validator import validate_config
from app.rag.retriever import get_store_stats
from app.utils.guards import safe_json_load


@dataclass
class DiagnosticCheck:
    name: str
    status: str  # "ok" | "warn" | "error"
    message: str
    remediation: str = ""


@dataclass
class DiagnosticReport:
    checks: list[DiagnosticCheck] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(check.status == "error" for check in self.checks)

    @property
    def has_warnings(self) -> bool:
        return any(check.status == "warn" for check in self.checks)

    @property
    def error_count(self) -> int:
        return sum(1 for check in self.checks if check.status == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for check in self.checks if check.status == "warn")

    def add(self, check: DiagnosticCheck) -> None:
        self.checks.append(check)


def _status_icon(status: str) -> str:
    return {"ok": "[OK]", "warn": "[WARN]", "error": "[FAIL]"}.get(status, "[?]")


def format_health_view(report: DiagnosticReport) -> str:
    """Short pass/warn/fail summary with top remediation hints."""
    if report.has_errors:
        overall = f"UNHEALTHY — {report.error_count} error(s)"
        if report.warning_count:
            overall += f", {report.warning_count} warning(s)"
    elif report.has_warnings:
        overall = f"OK with {report.warning_count} warning(s)"
    else:
        overall = "OK — all checks passed"

    lines = [f"Health: {overall}", ""]
    hints: list[str] = []
    for check in report.checks:
        if check.status in ("error", "warn"):
            lines.append(f"  {_status_icon(check.status)} {check.name}: {check.message}")
            if check.remediation and len(hints) < 3:
                hints.append(f"  • {check.remediation}")
    if hints:
        lines.append("")
        lines.append("Suggested fixes:")
        lines.extend(hints)
    lines.append("")
    lines.append("Run /diagnostics for the full report.")
    return "\n".join(lines)


def format_diagnostics_view(report: DiagnosticReport) -> str:
    """Full bullet list for /diagnostics."""
    lines = ["Diagnostics report:", ""]
    for check in report.checks:
        lines.append(f"{_status_icon(check.status)} {check.name}")
        lines.append(f"  {check.message}")
        if check.remediation:
            lines.append(f"  Fix: {check.remediation}")
        lines.append("")
    return "\n".join(lines).rstrip()


def format_config_view(config: AppConfig) -> str:
    """Resolved configuration paths, features, and key limits."""
    features = features_to_yaml_dict(config.features)
    feature_lines = [f"  {key}: {str(value).lower()}" for key, value in features.items()]

    return "\n".join(
        [
            "Configuration (resolved paths):",
            "",
            "Model:",
            f"  chatModelPath: {config.model.chat_model}",
            f"  embeddingModelPath: {config.model.embedding_model}",
            f"  contextSize: {config.model.context_size}",
            f"  gpuLayers: {config.model.gpu_layers}",
            f"  threads: {config.model.threads}",
            "",
            "Features:",
            *feature_lines,
            "",
            "RAG:",
            f"  dbPath: {config.rag.db_dir}",
            f"  docsPath: {config.rag.docs_dir}",
            f"  topK: {config.rag.top_k}",
            "",
            "Memory:",
            f"  userFile: {config.memory.user_path}",
            f"  memoryFile: {config.memory.memory_path}",
            f"  sessionFile: {config.memory.session_path}",
            f"  updateEveryTurns: {config.memory.update_every_turns}",
            "",
            "Skills:",
            f"  registryPath: {config.skills.registry_file}",
            f"  activePath: {config.skills.active_dir}",
            "",
            "Tasks:",
            f"  kanbanPath: {config.tasks.kanban_file}",
            "",
            "Sessions:",
            f"  storePath: {config.sessions.store_dir}",
            "",
            "Logging:",
            f"  logPath: {config.logging.log_path}",
            f"  level: {config.logging.level}",
        ]
    )


def _probe_nvidia_smi() -> str | None:
    """Return nvidia-smi summary line or None if unavailable."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()[0]
    except (OSError, subprocess.TimeoutExpired):
        return None
    return None


def _check_memory_file(path: Path, label: str, report: DiagnosticReport) -> None:
    if not path.exists():
        report.add(
            DiagnosticCheck(
                name=label,
                status="warn",
                message=f"File not found: {path}",
                remediation="Will be created on first use.",
            )
        )
        return
    try:
        path.read_text(encoding="utf-8")
        report.add(
            DiagnosticCheck(
                name=label,
                status="ok",
                message=f"Readable ({path})",
            )
        )
    except OSError as error:
        report.add(
            DiagnosticCheck(
                name=label,
                status="warn",
                message=f"Cannot read {path}: {error}",
                remediation="Check file permissions.",
            )
        )


def run_startup_diagnostics(
    config: AppConfig,
    *,
    loaded: bool = False,
    compute_detail: str | None = None,
) -> DiagnosticReport:
    """Run environment checks without loading the chat model unless loaded=True."""
    report = DiagnosticReport()

    if DEFAULT_CONFIG_PATH.exists():
        report.add(
            DiagnosticCheck(
                name="Config file",
                status="ok",
                message=f"Found {DEFAULT_CONFIG_PATH}",
            )
        )
    else:
        report.add(
            DiagnosticCheck(
                name="Config file",
                status="error",
                message=f"Config file not found: {DEFAULT_CONFIG_PATH}",
                remediation="Create config.yaml at the project root.",
            )
        )

    for issue in validate_config(config):
        report.add(
            DiagnosticCheck(
                name=f"Config: {issue.field}",
                status="error" if issue.severity == "error" else "warn",
                message=issue.message,
                remediation=issue.remediation,
            )
        )

    chat_path = config.model.chat_model
    if chat_path.exists() and chat_path.stat().st_size > 0:
        size_mb = chat_path.stat().st_size / (1024 * 1024)
        report.add(
            DiagnosticCheck(
                name="Chat model",
                status="ok",
                message=f"{chat_path.name} ({size_mb:.1f} MB)",
            )
        )
    elif not any(c.name == "Config: model.chatModelPath" for c in report.checks):
        report.add(
            DiagnosticCheck(
                name="Chat model",
                status="error",
                message=f"Missing or empty: {chat_path}",
                remediation="Download a GGUF and update model.chatModelPath.",
            )
        )

    if config.features.rag:
        embed_path = config.model.embedding_model
        if embed_path.exists() and embed_path.stat().st_size > 0:
            report.add(
                DiagnosticCheck(
                    name="Embedding model",
                    status="ok",
                    message=str(embed_path),
                )
            )
        elif not any(
            c.name == "Config: model.embeddingModelPath" for c in report.checks
        ):
            report.add(
                DiagnosticCheck(
                    name="Embedding model",
                    status="error",
                    message=f"Missing or empty: {embed_path}",
                    remediation="Add embedding GGUF or disable features.rag.",
                )
            )

    backend = detect_compute_backend(config)
    cuda_msg = backend.detail
    if config.model.gpu_layers != 0:
        smi = _probe_nvidia_smi()
        if smi:
            cuda_msg = f"{backend.detail}; GPU: {smi}"
        elif backend.mode == "gpu":
            cuda_msg = f"{backend.detail}; nvidia-smi not available"
    status = "ok" if backend.mode == "gpu" or config.model.gpu_layers == 0 else "warn"
    if backend.mode == "cpu" and config.model.gpu_layers != 0:
        status = "warn"
    report.add(
        DiagnosticCheck(
            name="Compute backend",
            status=status,
            message=f"{backend.label}: {cuda_msg}",
            remediation=(
                "Install CUDA-enabled llama-cpp-python or set gpuLayers: 0."
                if status == "warn"
                else ""
            ),
        )
    )

    if config.features.rag:
        stats = get_store_stats(config)
        chunk_count = int(stats.get("chunk_count", 0))
        sources = stats.get("sources", [])
        db_dir = config.rag.db_dir
        if chunk_count > 0:
            report.add(
                DiagnosticCheck(
                    name="RAG index",
                    status="ok",
                    message=(
                        f"{chunk_count} chunk(s), {len(sources)} source(s) "
                        f"at {db_dir}"
                    ),
                )
            )
        elif db_dir.exists():
            report.add(
                DiagnosticCheck(
                    name="RAG index",
                    status="warn",
                    message=f"ChromaDB exists but index is empty ({db_dir})",
                    remediation="Add docs and run /ingest.",
                )
            )
        else:
            report.add(
                DiagnosticCheck(
                    name="RAG index",
                    status="warn",
                    message=f"No vector store at {db_dir}",
                    remediation="Run /ingest after adding documents to docs/.",
                )
            )

    registry_path = config.skills.registry_file
    data, error = safe_json_load(registry_path)
    if error:
        report.add(
            DiagnosticCheck(
                name="Skills registry",
                status="warn",
                message=error,
                remediation=f"Fix or remove {registry_path}.",
            )
        )
    else:
        count = len(data.get("skills", [])) if data else 0
        report.add(
            DiagnosticCheck(
                name="Skills registry",
                status="ok",
                message=f"{count} skill(s) in registry",
            )
        )

    kanban_path = config.tasks.kanban_file
    kanban_data, kanban_error = safe_json_load(kanban_path)
    if kanban_error:
        report.add(
            DiagnosticCheck(
                name="Kanban board",
                status="warn",
                message=kanban_error,
                remediation=f"Fix or remove {kanban_path}.",
            )
        )
    else:
        tasks = kanban_data.get("tasks", []) if kanban_data else []
        report.add(
            DiagnosticCheck(
                name="Kanban board",
                status="ok",
                message=f"{len(tasks)} task(s) on board",
            )
        )

    _check_memory_file(config.memory.user_path, "Memory: user.md", report)
    _check_memory_file(config.memory.memory_path, "Memory: memory.md", report)
    _check_memory_file(config.memory.session_path, "Memory: session.md", report)

    sessions_dir = config.sessions.store_dir
    try:
        sessions_dir.mkdir(parents=True, exist_ok=True)
        test_file = sessions_dir / ".write_test"
        test_file.write_text("", encoding="utf-8")
        test_file.unlink(missing_ok=True)
        report.add(
            DiagnosticCheck(
                name="Sessions store",
                status="ok",
                message=f"Writable: {sessions_dir}",
            )
        )
    except OSError as error:
        report.add(
            DiagnosticCheck(
                name="Sessions store",
                status="error",
                message=f"Cannot write to {sessions_dir}: {error}",
                remediation="Check directory permissions.",
            )
        )

    if loaded:
        detail = compute_detail or "Chat model loaded"
        report.add(
            DiagnosticCheck(
                name="Model runtime",
                status="ok",
                message=detail,
            )
        )

    return report
