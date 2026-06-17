"""Model runtime: loads and serves the local GGUF chat and embedding models.

This module wraps ``llama_cpp.Llama`` so the rest of the app never touches the
backend directly. The heavy ``llama_cpp`` import is deferred until a model is
actually loaded, which keeps config/prompt logic importable in lightweight
environments (tests, tooling) without the native dependency.
"""

from __future__ import annotations

import gc
import threading
from typing import TYPE_CHECKING, Any, Iterator

from app.core.compute_backend import UNKNOWN, ComputeBackend, detect_compute_backend
from app.core.config import AppConfig

if TYPE_CHECKING:  # pragma: no cover - typing only
    from llama_cpp import Llama


class ModelRuntime:
    """Owns the chat model and the optional embedding model."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._chat: "Llama | None" = None
        self._embedder: "Llama | None" = None
        self._compute_backend: ComputeBackend = UNKNOWN
        self._lock = threading.RLock()

    @property
    def compute_backend(self) -> ComputeBackend:
        return self._compute_backend

    def load_chat_model(self) -> "Llama":
        """Load the GGUF chat model, raising a clear error if it is missing."""
        with self._lock:
            return self._load_chat_model_unlocked()

    def unload_chat_model(self) -> None:
        """Release the loaded chat model and reset compute backend detection."""
        with self._lock:
            self._chat = None
            self._compute_backend = UNKNOWN
        gc.collect()

    def reload_chat_model(self) -> "Llama":
        """Unload and load the chat model from the current config path."""
        with self._lock:
            self._chat = None
            self._compute_backend = UNKNOWN
        gc.collect()
        with self._lock:
            return self._load_chat_model_unlocked(force=True)

    def load_embedding_model(self) -> "Llama":
        """Load the GGUF embedding model, raising a clear error if missing."""
        with self._lock:
            return self._load_embedding_model_unlocked()

    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for a piece of text."""
        with self._lock:
            embedder = self._load_embedding_model_unlocked()
            result = embedder.create_embedding(text)
            return result["data"][0]["embedding"]

    def _completion_params(self, **overrides: Any) -> dict[str, Any]:
        gen = self.config.generation
        params: dict[str, Any] = {
            "temperature": gen.temperature,
            "top_p": gen.top_p,
            "repeat_penalty": gen.repeat_penalty,
            "max_tokens": gen.max_tokens,
            "stop": gen.stop,
        }
        params.update(overrides)
        return params

    def _load_chat_model_unlocked(self, *, force: bool = False) -> "Llama":
        if self._chat is not None and not force:
            return self._chat

        from llama_cpp import Llama

        model_path = self.config.model.chat_model
        if not model_path.exists():
            raise FileNotFoundError(
                f"Chat model not found: {model_path}. "
                "Check model.chatModelPath in config.yaml."
            )

        print("Loading chat model...")
        self._chat = Llama(
            model_path=str(model_path),
            n_ctx=self.config.model.context_size,
            n_gpu_layers=self.config.model.gpu_layers,
            n_threads=self.config.model.threads,
            chat_format=self.config.model.chat_format,
            verbose=False,
        )
        self._compute_backend = detect_compute_backend(self.config)
        return self._chat

    def _load_embedding_model_unlocked(self) -> "Llama":
        if self._embedder is not None:
            return self._embedder

        from llama_cpp import Llama

        model_path = self.config.model.embedding_model
        if not model_path.exists():
            raise FileNotFoundError(
                f"Embedding model not found: {model_path}. "
                "Check model.embeddingModelPath in config.yaml, "
                "or disable RAG in config.yaml (features.rag: false)."
            )

        print("Loading embedding model...")
        self._embedder = Llama(
            model_path=str(model_path),
            embedding=True,
            n_ctx=2048,
            n_gpu_layers=self.config.model.gpu_layers,
            verbose=False,
        )
        return self._embedder

    def create_chat_completion(
        self,
        messages: list[dict[str, str]],
        stream: bool = False,
        **overrides: Any,
    ) -> Any:
        """Run a chat completion using configured generation defaults.

        Any keyword in ``overrides`` takes precedence over the config values.
        Returns the raw llama_cpp response, or a streaming iterator when
        ``stream`` is True.
        """
        if stream:
            return self._stream_chat_completion(messages, **overrides)

        with self._lock:
            chat = self._load_chat_model_unlocked()
            return chat.create_chat_completion(
                messages=messages,
                stream=False,
                **self._completion_params(**overrides),
            )

    def _stream_chat_completion(
        self,
        messages: list[dict[str, str]],
        **overrides: Any,
    ) -> Iterator[dict[str, Any]]:
        """Stream tokens while holding the runtime lock for the full decode."""
        with self._lock:
            chat = self._load_chat_model_unlocked()
            stream = chat.create_chat_completion(
                messages=messages,
                stream=True,
                **self._completion_params(**overrides),
            )
            for chunk in stream:
                yield chunk

    @staticmethod
    def iter_stream_text(stream: Iterator[dict[str, Any]]) -> Iterator[str]:
        """Yield incremental text from a streaming chat completion."""
        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content")
            if content:
                yield content
