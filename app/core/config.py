"""Configuration loading for SoulForge TUI.

All tunable values live in ``config.yaml`` at the project root. Core logic
should depend only on the typed dataclasses produced here, never on hardcoded
paths or magic numbers.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def resolve_path(value: str | Path) -> Path:
    """Resolve a possibly-relative config path against the project root."""
    path = Path(value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


@dataclass
class ModelConfig:
    chat_model_path: str
    embedding_model_path: str
    context_size: int = 8192
    gpu_layers: int = -1
    threads: int = 8
    chat_format: str = "mistral-instruct"

    @property
    def chat_model(self) -> Path:
        return resolve_path(self.chat_model_path)

    @property
    def embedding_model(self) -> Path:
        return resolve_path(self.embedding_model_path)


@dataclass
class GenerationConfig:
    temperature: float = 0.75
    top_p: float = 0.95
    repeat_penalty: float = 1.1
    max_tokens: int = 700
    stop: list[str] = field(default_factory=lambda: ["</s>", "User:", "You:"])


@dataclass
class FeatureConfig:
    soul: bool = True
    rag: bool = True
    memory: bool = True
    skills: bool = False
    curator: bool = False
    kanban: bool = False
    streaming: bool = True
    show_sources: bool = True


# Maps FeatureConfig attribute names to config.yaml camelCase keys.
FEATURE_YAML_KEYS: dict[str, str] = {
    "soul": "soul",
    "rag": "rag",
    "memory": "memory",
    "skills": "skills",
    "curator": "curator",
    "kanban": "kanban",
    "streaming": "streaming",
    "show_sources": "showSources",
}

# Short labels used in the status bar and /features list.
FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "soul": "soul",
    "rag": "rag",
    "memory": "memory",
    "skills": "skills",
    "curator": "curator",
    "kanban": "kanban",
    "show_sources": "sources",
    "streaming": "streaming",
}


@dataclass
class RagConfig:
    db_path: str = "./chromaDb"
    docs_path: str = "./docs"
    collection_name: str = "localDocs"
    top_k: int = 5
    chunk_size: int = 1200
    chunk_overlap: int = 200
    pdf_ocr_enabled: bool = True
    pdf_ocr_lang: str = "eng"
    pdf_min_text_chars: int = 32

    @property
    def db_dir(self) -> Path:
        return resolve_path(self.db_path)

    @property
    def docs_dir(self) -> Path:
        return resolve_path(self.docs_path)


@dataclass
class MemoryConfig:
    user_file: str = "./app/memory/user.md"
    memory_file: str = "./app/memory/memory.md"
    session_file: str = "./app/memory/session.md"
    update_every_turns: int = 10
    max_user_chars: int = 3000
    max_memory_chars: int = 6000
    max_session_chars: int = 4000

    @property
    def user_path(self) -> Path:
        return resolve_path(self.user_file)

    @property
    def memory_path(self) -> Path:
        return resolve_path(self.memory_file)

    @property
    def session_path(self) -> Path:
        return resolve_path(self.session_file)


@dataclass
class SkillsConfig:
    active_path: str = "./app/skills/active"
    archived_path: str = "./app/skills/archived"
    registry_path: str = "./app/skills/registry.json"
    workflow_log_path: str = "./app/skills/workflow_log.json"
    auto_create: bool = False
    min_successful_repeats: int = 3
    success_window_turns: int = 3

    @property
    def active_dir(self) -> Path:
        return resolve_path(self.active_path)

    @property
    def archived_dir(self) -> Path:
        return resolve_path(self.archived_path)

    @property
    def registry_file(self) -> Path:
        return resolve_path(self.registry_path)

    @property
    def workflow_log_file(self) -> Path:
        return resolve_path(self.workflow_log_path)


@dataclass
class CuratorConfig:
    stale_days: int = 30
    bloat_max_chars: int = 1000


@dataclass
class TasksConfig:
    kanban_path: str = "./app/tasks/kanban.json"

    @property
    def kanban_file(self) -> Path:
        return resolve_path(self.kanban_path)


@dataclass
class SessionsConfig:
    store_path: str = "./app/sessions"
    max_saved_sessions: int = 50

    @property
    def store_dir(self) -> Path:
        return resolve_path(self.store_path)


@dataclass
class LoggingConfig:
    log_path: str = "./logs/soulforge.log"
    level: str = "info"
    console: bool = False


@dataclass
class AppConfig:
    model: ModelConfig
    generation: GenerationConfig
    features: FeatureConfig
    rag: RagConfig
    memory: MemoryConfig
    skills: SkillsConfig
    curator: CuratorConfig
    tasks: TasksConfig
    sessions: SessionsConfig
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    raw: dict[str, Any] = field(default_factory=dict)


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    return value if isinstance(value, dict) else {}


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load and validate ``config.yaml`` into typed dataclasses."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            "Copy or create config.yaml at the project root."
        )

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    model_section = _section(data, "model")
    model = ModelConfig(
        chat_model_path=model_section.get(
            "chatModelPath", "./models/NemoMix-Unleashed-12B-Q4_K_M.gguf"
        ),
        embedding_model_path=model_section.get(
            "embeddingModelPath", "./models/embedding-model.gguf"
        ),
        context_size=model_section.get("contextSize", 8192),
        gpu_layers=model_section.get("gpuLayers", -1),
        threads=model_section.get("threads", 8),
        chat_format=model_section.get("chatFormat", "mistral-instruct"),
    )

    gen_section = _section(data, "generation")
    generation = GenerationConfig(
        temperature=gen_section.get("temperature", 0.75),
        top_p=gen_section.get("topP", 0.95),
        repeat_penalty=gen_section.get("repeatPenalty", 1.1),
        max_tokens=gen_section.get("maxTokens", 700),
        stop=gen_section.get("stop", ["</s>", "User:", "You:"]),
    )

    feat_section = _section(data, "features")
    features = FeatureConfig(
        soul=feat_section.get("soul", True),
        rag=feat_section.get("rag", True),
        memory=feat_section.get("memory", True),
        skills=feat_section.get("skills", False),
        curator=feat_section.get("curator", False),
        kanban=feat_section.get("kanban", False),
        streaming=feat_section.get("streaming", True),
        show_sources=feat_section.get("showSources", True),
    )

    rag_section = _section(data, "rag")
    rag = RagConfig(
        db_path=rag_section.get("dbPath", "./chromaDb"),
        docs_path=rag_section.get("docsPath", "./docs"),
        collection_name=rag_section.get("collectionName", "localDocs"),
        top_k=rag_section.get("topK", 5),
        chunk_size=rag_section.get("chunkSize", 1200),
        chunk_overlap=rag_section.get("chunkOverlap", 200),
        pdf_ocr_enabled=rag_section.get("pdfOcrEnabled", True),
        pdf_ocr_lang=rag_section.get("pdfOcrLang", "eng"),
        pdf_min_text_chars=rag_section.get("pdfMinTextChars", 32),
    )

    mem_section = _section(data, "memory")
    memory = MemoryConfig(
        user_file=mem_section.get("userFile", "./app/memory/user.md"),
        memory_file=mem_section.get("memoryFile", "./app/memory/memory.md"),
        session_file=mem_section.get("sessionFile", "./app/memory/session.md"),
        update_every_turns=mem_section.get("updateEveryTurns", 10),
        max_user_chars=mem_section.get("maxUserChars", 3000),
        max_memory_chars=mem_section.get("maxMemoryChars", 6000),
        max_session_chars=mem_section.get("maxSessionChars", 4000),
    )

    skills_section = _section(data, "skills")
    skills = SkillsConfig(
        active_path=skills_section.get("activePath", "./app/skills/active"),
        archived_path=skills_section.get("archivedPath", "./app/skills/archived"),
        registry_path=skills_section.get("registryPath", "./app/skills/registry.json"),
        workflow_log_path=skills_section.get(
            "workflowLogPath", "./app/skills/workflow_log.json"
        ),
        auto_create=skills_section.get("autoCreate", False),
        min_successful_repeats=skills_section.get("minSuccessfulRepeats", 3),
        success_window_turns=skills_section.get("successWindowTurns", 3),
    )

    curator_section = _section(data, "curator")
    curator = CuratorConfig(
        stale_days=curator_section.get("staleDays", 30),
        bloat_max_chars=curator_section.get("bloatMaxChars", 1000),
    )

    tasks_section = _section(data, "tasks")
    tasks = TasksConfig(
        kanban_path=tasks_section.get("kanbanPath", "./app/tasks/kanban.json"),
    )

    sessions_section = _section(data, "sessions")
    sessions = SessionsConfig(
        store_path=sessions_section.get("storePath", "./app/sessions"),
        max_saved_sessions=sessions_section.get("maxSavedSessions", 50),
    )

    logging_section = _section(data, "logging")
    logging_cfg = LoggingConfig(
        log_path=logging_section.get("logPath", "./logs/soulforge.log"),
        level=logging_section.get("level", "info"),
        console=logging_section.get("console", False),
    )

    return AppConfig(
        model=model,
        generation=generation,
        features=features,
        rag=rag,
        memory=memory,
        skills=skills,
        curator=curator,
        tasks=tasks,
        sessions=sessions,
        logging=logging_cfg,
        raw=data,
    )


def features_to_yaml_dict(features: FeatureConfig) -> dict[str, bool]:
    """Convert a FeatureConfig to the camelCase dict written under ``features:``."""
    return {
        yaml_key: getattr(features, attr)
        for attr, yaml_key in FEATURE_YAML_KEYS.items()
    }


def save_features(
    config: AppConfig,
    path: str | Path | None = None,
) -> None:
    """Persist the current feature flags to ``config.yaml`` (atomic write)."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH

    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    else:
        data = dict(config.raw) if config.raw else {}

    data["features"] = features_to_yaml_dict(config.features)
    config.raw = data

    directory = config_path.parent
    directory.mkdir(parents=True, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(
        dir=directory,
        prefix=".config-",
        suffix=".yaml.tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, default_flow_style=False, sort_keys=False)
        os.replace(temp_path, config_path)
    except Exception:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise
