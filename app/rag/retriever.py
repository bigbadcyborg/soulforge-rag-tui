"""RAG retriever: queries the local ChromaDB vector store.

Fails gracefully when the database is missing or empty so the chatbot keeps
working even before any documents have been ingested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.core.config import AppConfig
from app.core.model_runtime import ModelRuntime

if TYPE_CHECKING:  # pragma: no cover - typing only
    from chromadb.api.models.Collection import Collection


@dataclass
class RetrievedChunk:
    source: str
    chunk_index: Any
    distance: float | None
    document: str


class Retriever:
    """Embeds queries and retrieves the most relevant document chunks."""

    def __init__(self, config: AppConfig, runtime: ModelRuntime) -> None:
        self.config = config
        self.runtime = runtime
        self._collection: "Collection | None" = None

    def _get_collection(self) -> "Collection | None":
        if self._collection is not None:
            return self._collection

        try:
            import chromadb

            client = chromadb.PersistentClient(path=str(self.config.rag.db_dir))
            self._collection = client.get_or_create_collection(
                name=self.config.rag.collection_name
            )
        except Exception as error:  # noqa: BLE001 - degrade gracefully
            print(f"[rag] Could not open vector store: {error}")
            return None

        return self._collection

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Return the top-K relevant chunks for a query, or [] on failure."""
        collection = self._get_collection()
        if collection is None:
            return []

        k = top_k if top_k is not None else self.config.rag.top_k

        try:
            query_embedding = self.runtime.embed(query)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as error:  # noqa: BLE001 - degrade gracefully
            print(f"[rag] Retrieval failed: {error}")
            return []

        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        chunks: list[RetrievedChunk] = []
        for index, document in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) else {}
            distance = distances[index] if index < len(distances) else None
            chunks.append(
                RetrievedChunk(
                    source=metadata.get("source", "unknown"),
                    chunk_index=metadata.get("chunkIndex", "unknown"),
                    distance=distance,
                    document=document,
                )
            )

        return chunks

    def get_available_sources(self) -> list[str]:
        """Return the unique set of document sources in the vector store."""
        collection = self._get_collection()
        if collection is None:
            return []

        try:
            # Get all documents with metadata to extract unique sources
            results = collection.get(include=["metadatas"])
            metadatas = results.get("metadatas") or []
            sources = set()
            for metadata in metadatas:
                if isinstance(metadata, dict):
                    source = metadata.get("source")
                    if source:
                        sources.add(source)
            return sorted(list(sources))
        except Exception as error:  # noqa: BLE001 - degrade gracefully
            print(f"[rag] Failed to get sources: {error}")
            return []

    @staticmethod
    def format_context(chunks: list[RetrievedChunk]) -> str:
        """Render retrieved chunks into a prompt-ready context block."""
        blocks = [
            f"[Source: {chunk.source}, chunk: {chunk.chunk_index}, "
            f"distance: {chunk.distance}]\n{chunk.document}"
            for chunk in chunks
        ]
        return "\n\n---\n\n".join(blocks)
