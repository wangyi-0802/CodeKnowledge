"""Vector store abstraction layer over ChromaDB."""
from __future__ import annotations
import uuid
from typing import Any
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions
from src.config.settings import get_settings
from src.utils.logger import get_logger
logger = get_logger(__name__)
class VectorStore:
    """Persistent vector store for code chunks using ChromaDB.
    Provides add, search, and delete operations with metadata filtering.
    """
    def __init__(self, collection_name: str = "code_knowledge", persist_dir: str | None = None):
        settings = get_settings()
        self.persist_dir = persist_dir or settings.vector_store_path
        self.collection_name = collection_name
        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        if settings.openai_api_key:
            self._ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=settings.openai_api_key,
                model_name=settings.embedding_model,
            )
        else:
            self._ef = None
        self._collection = self._get_or_create_collection()
        logger.info(
            "VectorStore ready at %s (collection: %s, count: %d)",
            self.persist_dir, collection_name, self._collection.count(),
        )
    def _get_or_create_collection(self) -> Any:
        try:
            return self._client.get_collection(
                name=self.collection_name,
                embedding_function=self._ef,
            )
        except ValueError:
            return self._client.create_collection(
                name=self.collection_name,
                embedding_function=self._ef,
            )
    def add_chunks(
        self,
        contents: list[str],
        metadata: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in contents]
        if metadata is None:
            metadata = [{} for _ in contents]
        self._collection.add(
            documents=contents,
            metadatas=metadata,
            ids=ids,
        )
        logger.debug("Added %d chunks to vector store", len(contents))
        return ids
    def search(
        self,
        query: str,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        results = self._collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
        )
        formatted = []
        for i in range(len(results["ids"][0])):
            formatted.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else 0.0,
            })
        return formatted
    def delete_chunks(self, ids: list[str]) -> None:
        self._collection.delete(ids=ids)
        logger.debug("Deleted %d chunks", len(ids))
    def count(self) -> int:
        return self._collection.count()
    def clear(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._get_or_create_collection()
        logger.info("Cleared collection: %s", self.collection_name)
