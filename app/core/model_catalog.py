"""Chat model discovery and import helpers."""

from __future__ import annotations

import shutil
from typing import Callable
from pathlib import Path

from app.core.config import PROJECT_ROOT, AppConfig

MODELS_DIR = PROJECT_ROOT / "models"
GGUF_SUFFIX = ".gguf"


def path_for_config(path: Path) -> str:
    """Return a project-relative path string when possible."""
    resolved = path.resolve()
    try:
        rel = resolved.relative_to(PROJECT_ROOT.resolve())
        return str(rel).replace("\\", "/")
    except ValueError:
        return str(resolved)


def list_available_chat_models(config: AppConfig) -> list[Path]:
    """Return sorted ``*.gguf`` files in ``./models/``, excluding the embedding model."""
    if not MODELS_DIR.is_dir():
        return []

    embedding = config.model.embedding_model.resolve()
    models: list[Path] = []
    for path in sorted(MODELS_DIR.glob(f"*{GGUF_SUFFIX}")):
        if path.resolve() == embedding:
            continue
        models.append(path.resolve())
    return models


def validate_gguf_source(path: Path) -> Path:
    """Resolve *path* and require an existing ``.gguf`` file."""
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"Not a file: {resolved}")
    if resolved.suffix.lower() != GGUF_SUFFIX:
        raise ValueError("Only .gguf files can be imported as chat models.")
    return resolved


def unique_dest_path(filename: str) -> Path:
    """Return ``models/<filename>`` or a suffixed variant if the name collides."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / filename
    if not dest.exists():
        return dest.resolve()

    stem = dest.stem
    suffix = dest.suffix
    counter = 1
    while True:
        candidate = MODELS_DIR / f"{stem}-{counter}{suffix}"
        if not candidate.exists():
            return candidate.resolve()
        counter += 1


def import_chat_model(
    source: Path,
    *,
    dest_name: str | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Copy *source* into ``./models/`` and return the destination path."""
    resolved = validate_gguf_source(source)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if resolved.parent.resolve() == MODELS_DIR.resolve():
        raise ValueError(
            f"Model already in ./models/: {resolved.name}. "
            "Use /model <name> to switch to it."
        )

    filename = dest_name or resolved.name
    if not filename.lower().endswith(GGUF_SUFFIX):
        raise ValueError("Destination filename must end with .gguf.")

    dest = unique_dest_path(filename)

    total_size = resolved.stat().st_size
    copied = 0
    chunk_size = 1024 * 1024 * 8  # 8 MB chunks

    with resolved.open("rb") as fsrc, dest.open("wb") as fdst:
        while True:
            buf = fsrc.read(chunk_size)
            if not buf:
                break
            fdst.write(buf)
            copied += len(buf)
            if on_progress:
                on_progress(copied, total_size)

    shutil.copystat(resolved, dest)
    return dest.resolve()


def resolve_chat_model_selection(name: str, config: AppConfig) -> Path:
    """Resolve a model filename or unique partial match from ``./models/``."""
    query = name.strip()
    if not query:
        raise ValueError("Model name is required.")

    available = list_available_chat_models(config)

    exact = [path for path in available if path.name == query]
    if len(exact) == 1:
        return exact[0]

    candidate = Path(query)
    if candidate.is_file():
        return validate_gguf_source(candidate.expanduser().resolve())

    models_path = (MODELS_DIR / query).resolve()
    if models_path.is_file():
        return validate_gguf_source(models_path)

    if not query.lower().endswith(GGUF_SUFFIX):
        suffixed = f"{query}{GGUF_SUFFIX}"
        exact = [path for path in available if path.name == suffixed]
        if len(exact) == 1:
            return exact[0]
        models_path = (MODELS_DIR / suffixed).resolve()
        if models_path.is_file():
            return validate_gguf_source(models_path)

    partial = [path for path in available if query.lower() in path.name.lower()]
    if len(partial) == 1:
        return partial[0]
    if not partial:
        raise FileNotFoundError(
            f"No chat model matching '{name}'. Run /model list to see available models."
        )
    names = ", ".join(path.name for path in partial)
    raise ValueError(f"Ambiguous model name '{name}'. Matches: {names}")
