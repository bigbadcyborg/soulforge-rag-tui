"""Tests for iteration 12 stability layer (diagnostics, logging, guards)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

from app.core.chat_controller import ChatController
from app.core.config import AppConfig, load_config
from app.core.config_validator import validate_config
from app.core.diagnostics import (
    DiagnosticCheck,
    DiagnosticReport,
    format_config_view,
    format_health_view,
    run_startup_diagnostics,
)
from app.utils.guards import format_startup_error, safe_json_load
from app.utils.logging import setup_logging


def _write_config(tmp_path: Path, **overrides) -> Path:
    data = {
        "model": {
            "chatModelPath": str(tmp_path / "chat.gguf"),
            "embeddingModelPath": str(tmp_path / "embed.gguf"),
            "contextSize": 8192,
            "gpuLayers": 0,
        },
        "features": {"rag": False, "soul": False, "memory": False},
        "rag": {"dbPath": str(tmp_path / "chroma"), "docsPath": str(tmp_path / "docs")},
        "memory": {
            "userFile": str(tmp_path / "memory" / "user.md"),
            "memoryFile": str(tmp_path / "memory" / "memory.md"),
            "sessionFile": str(tmp_path / "memory" / "session.md"),
        },
        "skills": {
            "activePath": str(tmp_path / "skills" / "active"),
            "archivedPath": str(tmp_path / "skills" / "archived"),
            "registryPath": str(tmp_path / "skills" / "registry.json"),
        },
        "tasks": {"kanbanPath": str(tmp_path / "tasks" / "kanban.json")},
        "sessions": {"storePath": str(tmp_path / "sessions")},
        "logging": {"logPath": str(tmp_path / "test.log"), "level": "info"},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and key in data and isinstance(data[key], dict):
            data[key].update(value)
        else:
            data[key] = value
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return config_path


def _minimal_dirs(tmp_path: Path) -> None:
    for sub in (
        "skills/active",
        "skills/archived",
        "memory",
        "sessions",
        "tasks",
    ):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    (tmp_path / "tasks" / "kanban.json").write_text(
        json.dumps({"tasks": []}), encoding="utf-8"
    )
    (tmp_path / "skills" / "registry.json").write_text(
        json.dumps({"skills": []}), encoding="utf-8"
    )


def test_validate_missing_chat_model_is_error(tmp_path) -> None:
    _minimal_dirs(tmp_path)
    config_path = _write_config(tmp_path)
    config = load_config(config_path)
    issues = validate_config(config)
    assert any(
        i.severity == "error" and "chat" in i.field.lower() for i in issues
    )


def test_validate_rag_on_missing_embedding_is_error(tmp_path) -> None:
    _minimal_dirs(tmp_path)
    (tmp_path / "chat.gguf").write_bytes(b"x" * 100)
    config_path = _write_config(tmp_path, features={"rag": True})
    config = load_config(config_path)
    issues = validate_config(config)
    assert any(
        i.severity == "error" and "embedding" in i.field.lower() for i in issues
    )


def test_diagnostics_missing_gguf_reported(tmp_path) -> None:
    _minimal_dirs(tmp_path)
    config_path = _write_config(tmp_path)
    config = load_config(config_path)
    report = run_startup_diagnostics(config)
    chat_checks = [
        c
        for c in report.checks
        if "chat" in c.name.lower() and c.status == "error"
    ]
    assert chat_checks
    assert report.has_errors


def test_diagnostics_corrupt_registry_warns(tmp_path) -> None:
    _minimal_dirs(tmp_path)
    (tmp_path / "chat.gguf").write_bytes(b"x" * 100)
    (tmp_path / "skills" / "registry.json").write_text("{bad", encoding="utf-8")
    config_path = _write_config(tmp_path)
    config = load_config(config_path)
    report = run_startup_diagnostics(config)
    registry_checks = [c for c in report.checks if c.name == "Skills registry"]
    assert registry_checks
    assert registry_checks[0].status == "warn"


def test_format_health_view_shows_unhealthy_on_errors() -> None:
    report = DiagnosticReport(
        checks=[
            DiagnosticCheck(
                name="Chat model",
                status="error",
                message="Missing",
                remediation="Add GGUF",
            )
        ]
    )
    text = format_health_view(report)
    assert "UNHEALTHY" in text
    assert "Add GGUF" in text


def test_format_config_view_includes_resolved_paths(tmp_path) -> None:
    _minimal_dirs(tmp_path)
    (tmp_path / "chat.gguf").write_bytes(b"x" * 100)
    config_path = _write_config(tmp_path)
    config = load_config(config_path)
    text = format_config_view(config)
    assert "chat.gguf" in text
    assert "Features:" in text
    assert str(config.sessions.store_dir) in text or "sessions" in text


def test_format_startup_error_file_not_found() -> None:
    text = format_startup_error(
        FileNotFoundError("Chat model not found: ./models/missing.gguf")
    )
    assert "Remediation" in text
    assert "config.yaml" in text


def test_logging_writes_to_temp_path(tmp_path) -> None:
    _minimal_dirs(tmp_path)
    log_file = tmp_path / "custom.log"
    config_path = _write_config(
        tmp_path,
        logging={"logPath": str(log_file), "level": "info", "console": False},
    )
    config = load_config(config_path)
    # Reset logging singleton for test isolation
    import app.utils.logging as logging_module

    logging_module._CONFIGURED = False
    root = logging.getLogger("soulforge")
    root.handlers.clear()

    setup_logging(config)
    logger = logging.getLogger("soulforge.test")
    logger.info("test message")
    for handler in root.handlers:
        handler.flush()

    assert log_file.exists()
    assert "test message" in log_file.read_text(encoding="utf-8")


def test_safe_json_load_corrupt() -> None:
    data, error = safe_json_load(Path("nonexistent-file-xyz.json"))
    assert data is None
    assert error is not None


def test_controller_health_check_with_corrupt_registry(tmp_path) -> None:
    _minimal_dirs(tmp_path)
    (tmp_path / "skills" / "registry.json").write_text("{bad", encoding="utf-8")
    config_path = _write_config(tmp_path)
    config = load_config(config_path)
    controller = ChatController(config)
    result = controller.run_health_check()
    assert isinstance(result, str)
    assert "Skills registry" in result or "UNHEALTHY" in result or "WARN" in result


def test_controller_get_config_view(tmp_path) -> None:
    _minimal_dirs(tmp_path)
    config_path = _write_config(tmp_path)
    config = load_config(config_path)
    controller = ChatController(config)
    view = controller.get_config_view()
    assert "Configuration" in view


def test_controller_run_diagnostics_never_raises(tmp_path) -> None:
    _minimal_dirs(tmp_path)
    config_path = _write_config(tmp_path)
    config = load_config(config_path)
    controller = ChatController(config)
    with patch(
        "app.core.chat_controller.run_startup_diagnostics",
        side_effect=RuntimeError("boom"),
    ):
        result = controller.run_diagnostics()
    assert "boom" in result
