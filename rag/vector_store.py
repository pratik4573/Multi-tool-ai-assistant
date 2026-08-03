"""
Vector store wrapper.

Defaults to ChromaDB (persistent, on-disk); a FAISS backend is available
behind the same interface via config.VECTOR_STORE_BACKEND.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    """Common interface used by the retriever, regardless of backend."""

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        raise NotImplementedError

    def query(
        self,
        embedding: list[float],
        top_k: int,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    def delete_by_source(self, source: str) -> None:
        raise NotImplementedError

    def is_empty(self) -> bool:
        raise NotImplementedError

    def clear(self) -> None:
        """Remove all indexed data."""
        raise NotImplementedError


# =====================================================================
# ChromaDB Backend
# =====================================================================

class ChromaVectorStore(VectorStore):
    """Persistent ChromaDB collection."""

    def __init__(self) -> None:
        import chromadb

        self._client = chromadb.PersistentClient(
            path=str(config.CHROMA_DB_DIR)
        )

        self._collection = self._client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            "ChromaDB collection '%s' initialized.",
            config.CHROMA_COLLECTION_NAME,
        )

    def add(
        self,
        ids,
        embeddings,
        documents,
        metadatas,
    ) -> None:

        if not ids:
            return

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(
        self,
        embedding: list[float],
        top_k: int,
    ) -> list[dict[str, Any]]:

        # NOTE: with metadata={"hnsw:space": "cosine"}, Chroma's returned
        # "distance" is (1 - cosine_similarity), so similarity = 1 - dist
        # below is correct and matches FAISS's inner-product score range
        # (both end up higher-is-better).
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
        )

        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]

        matches = []

        for doc, meta, dist in zip(docs, metas, dists):
            similarity = 1.0 - float(dist)

            matches.append(
                {
                    "text": doc,
                    "metadata": meta,
                    "score": similarity,
                }
            )

        return matches

    def delete_by_source(self, source: str) -> None:
        self._collection.delete(where={"source": source})

    def is_empty(self) -> bool:
        return self._collection.count() == 0

    def clear(self) -> None:
        """
        Remove every document from the collection.

        Safer than deleting the database directory manually.
        """

        logger.info("Clearing Chroma collection...")

        try:
            self._client.delete_collection(config.CHROMA_COLLECTION_NAME)
        except Exception:
            # Collection may not exist.
            pass

        self._collection = self._client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info("Chroma collection cleared.")


# =====================================================================
# FAISS Backend
# =====================================================================

class FaissVectorStore(VectorStore):
    """
    Lightweight in-memory FAISS backend.

    Does not persist metadata like ChromaDB.
    """

    def __init__(self) -> None:
        import faiss
        import numpy as np

        self._faiss = faiss
        self._np = np

        self._index: Any = None
        self._texts: list[str] = []
        self._metadatas: list[dict[str, Any]] = []
        self._ids: list[str] = []

    def _ensure_index(self, dimension: int) -> None:
        if self._index is None:
            # NOTE: IndexFlatIP computes inner product. Since embedder.py
            # calls encode(..., normalize_embeddings=True) for both queries
            # and passages, every vector is unit-length, so inner product
            # here is mathematically equivalent to cosine similarity.
            # Higher score = more similar, matching ChromaVectorStore's
            # (1 - distance) convention above. This is correct as long as
            # embedder.py keeps normalizing — if that ever changes, this
            # index type must change too (e.g. to IndexFlatL2).
            self._index = self._faiss.IndexFlatIP(dimension)

    def add(
        self,
        ids,
        embeddings,
        documents,
        metadatas,
    ) -> None:

        if not ids:
            return

        vectors = self._np.asarray(
            embeddings,
            dtype="float32",
        )

        self._ensure_index(vectors.shape[1])

        self._index.add(vectors)

        self._ids.extend(ids)
        self._texts.extend(documents)
        self._metadatas.extend(metadatas)

    def query(
        self,
        embedding: list[float],
        top_k: int,
    ) -> list[dict[str, Any]]:

        if self._index is None or self._index.ntotal == 0:
            return []

        vector = self._np.asarray(
            [embedding],
            dtype="float32",
        )

        scores, indices = self._index.search(
            vector,
            min(top_k, self._index.ntotal),
        )

        matches = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            matches.append(
                {
                    "text": self._texts[idx],
                    "metadata": self._metadatas[idx],
                    "score": float(score),
                }
            )

        return matches

    def delete_by_source(self, source: str) -> None:
        logger.warning(
            "delete_by_source() is not supported for FAISS. "
            "Use clear() and rebuild the index."
        )

    def is_empty(self) -> bool:
        return self._index is None or self._index.ntotal == 0

    def clear(self) -> None:
        """Reset the in-memory index."""

        logger.info("Clearing FAISS index...")

        self._index = None
        self._texts.clear()
        self._metadatas.clear()
        self._ids.clear()

        logger.info("FAISS index cleared.")


# =====================================================================
# Factory
# =====================================================================

@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    """
    Return a singleton vector store.
    """

    backend = config.VECTOR_STORE_BACKEND.lower()

    if backend == "chroma":
        try:
            logger.info("Using ChromaDB vector store.")
            return ChromaVectorStore()
        except ModuleNotFoundError as exc:
            logger.warning(
                "ChromaDB backend requested but chromadb is not installed; "
                "falling back to FAISS. Install chromadb if you want persistent "
                "on-disk RAG. (%s)",
                exc,
            )
            return FaissVectorStore()
        except Exception as exc:
            logger.warning(
                "Failed to initialize ChromaDB backend; falling back to FAISS: %s",
                exc,
            )
            return FaissVectorStore()

    logger.info("Using FAISS vector store.")
    return FaissVectorStore()