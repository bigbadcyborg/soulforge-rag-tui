"""Filesystem tool handlers."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from app.core.config import AppConfig
from app.tools.permissions import check_read_path, check_write_path


def read_file(config: AppConfig, args: dict) -> str:
    raw = str(args.get("path", "")).strip()
    if not raw:
        raise ValueError("read_file requires path")
    path = check_read_path(config, raw)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")
    max_bytes = config.tools.max_read_bytes
    size = path.stat().st_size
    if size > max_bytes:
        with path.open("rb") as handle:
            data = handle.read(max_bytes)
        return data.decode("utf-8", errors="replace") + f"\n\n[truncated at {max_bytes} bytes]"
    return path.read_text(encoding="utf-8", errors="replace")


def list_dir(config: AppConfig, args: dict) -> str:
    raw = str(args.get("path", ".")).strip() or "."
    path = check_read_path(config, raw)
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")
    if not path.is_dir():
        raise ValueError(f"Not a directory: {path}")
    entries = sorted(
        name for name in path.iterdir() if not name.name.startswith(".")
    )
    lines = [f"{item.name}/" if item.is_dir() else item.name for item in entries]
    return "\n".join(lines) if lines else "(empty directory)"


def write_file(config: AppConfig, args: dict) -> str:
    raw = str(args.get("path", "")).strip()
    content = str(args.get("content", ""))
    if not raw:
        raise ValueError("write_file requires path")
    path = check_write_path(config, raw)
    max_bytes = config.tools.max_read_bytes
    encoded = content.encode("utf-8")
    if len(encoded) > max_bytes:
        raise ValueError(f"Content exceeds max size ({max_bytes} bytes)")
    path.parent.mkdir(parents=True, exist_ok=True)
    directory = path.parent
    fd, temp_path = tempfile.mkstemp(dir=directory, prefix=".tool-write-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(temp_path, path)
    except Exception:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise
    return f"Wrote {len(content)} characters to {path}"
