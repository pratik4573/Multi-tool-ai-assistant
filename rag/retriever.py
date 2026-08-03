"""
High-level RAG API: ingest personal/uploaded documents and retrieve the most
relevant chunks for a query. This is what src/llm/orchestrator.py calls.
"""

from __future__ import annotations

from pathlib import Path

import config
from src.rag.chunker import chunk_text
from src.rag.document_loader import discover_personal_documents, load_document
from src.rag.embedder import get_shared_embedder
from src.rag.vector_store import get_vector_store
from src.utils.exceptions import RAGError
from src.utils.logger import get_logger
from src.utils.text_utils import new_id

logger = get_logger(__name__)


class Retriever:
    """Owns embedding + vector store access for personal knowledge retrieval."""

    def __init__(self) -> None:
        self._embedder = get_shared_embedder()
        self._store = get_vector_store()
        self._auto_index_personal_documents()

    # ------------------------------------------------------------------
    # Auto Index
    # ------------------------------------------------------------------

    def _auto_index_personal_documents(self) -> None:
        """Index personal documents once if vector DB is empty."""

        if not self._store.is_empty():
            return

        docs = discover_personal_documents()

        if not docs:
            logger.warning("No personal documents found.")
            return

        logger.info(
            "Auto-indexing %d personal document(s) for RAG.",
            len(docs),
        )

        total_chunks = 0

        for path in docs:
            total_chunks += self.ingest_path(path)

        logger.info(
            "Auto-indexing complete: %d chunk(s) indexed.",
            total_chunks,
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_path(self, path: Path) -> int:
        """Chunk, embed and store one document."""

        document = load_document(path)

        chunks = chunk_text(document.text)

        if not chunks:
            logger.warning("No content extracted from %s", path)
            return 0

        embeddings = self._embedder.embed_texts(chunks)

        ids = [new_id("chunk_") for _ in chunks]

        metadatas = [
            {
                "source": document.filename,
                "path": document.path,
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]

        try:
            self._store.add(
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
            )

        except Exception as exc:
            raise RAGError(
                f"Failed to index document '{document.filename}': {exc}"
            ) from exc

        logger.info(
            "Indexed '%s' into %d chunks.",
            document.filename,
            len(chunks),
        )

        return len(chunks)

    def ingest_all_personal_documents(self) -> int:
        """Re-index every personal document."""

        total = 0

        for path in discover_personal_documents():
            total += self.ingest_path(path)

        return total

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = config.RAG_TOP_K,
    ) -> list[str]:

        if not query.strip():
            return []

        query_embedding = self._embedder.embed_query(query)

        # FIX: previously we asked the vector store for exactly `top_k`
        # matches and returned them as-is. On a small corpus (a handful of
        # short files), the raw similarity ranking can easily be dominated
        # by multiple chunks from the SAME source file, leaving other
        # files with zero representation even though they're clearly
        # relevant to the query. We now pull a wider candidate pool and
        # diversify across sources before trimming back down to top_k.
        candidate_multiplier = getattr(config, "RAG_CANDIDATE_POOL_MULTIPLIER", 3)
        fetch_k = max(top_k * candidate_multiplier, top_k)

        matches = self._store.query(
            query_embedding,
            top_k=fetch_k,
        )

        logger.info("=" * 80)
        logger.info("User Query: %s", query)
        logger.info("Retrieved %d candidate match(es) (pool size %d)", len(matches), fetch_k)

        if not matches:
            logger.info("No matches found.")
            logger.info("=" * 80)
            return []

        # FIX: enforce a per-source cap so one file's chunks can't crowd
        # out every other file within the final top_k selection. Matches
        # are assumed to already be sorted by score (descending) by the
        # vector store.
        max_per_source = getattr(config, "RAG_MAX_CHUNKS_PER_SOURCE", 2)

        selected = []
        overflow = []
        per_source_count: dict[str, int] = {}

        for match in matches:
            source = match.get("metadata", {}).get("source")
            count = per_source_count.get(source, 0)

            if count < max_per_source:
                selected.append(match)
                per_source_count[source] = count + 1
            else:
                overflow.append(match)

            if len(selected) >= top_k:
                break

        # If the diversity cap left empty slots (e.g. fewer distinct
        # sources than top_k), fill remaining slots from overflow so we
        # still return up to top_k results.
        if len(selected) < top_k:
            for match in overflow:
                selected.append(match)
                if len(selected) >= top_k:
                    break

        matches = selected

        logger.info("Selected %d match(es) after source diversification", len(matches))

        for i, match in enumerate(matches, start=1):
            logger.info(
                "%d. score=%s source=%s chunk=%s",
                i,
                round(match.get("score", 0), 6),
                match.get("metadata", {}).get("source"),
                match.get("metadata", {}).get("chunk_index"),
            )

        relevant = []

        # Try threshold filtering first
        for match in matches:
            score = match.get("score", 0.0)
            if score >= config.RAG_SCORE_THRESHOLD:
                relevant.append(match["text"])

        # Fallback: if threshold removed everything,
        # still return top-k results instead of nothing.
        if not relevant:
            logger.warning(
                "No chunks passed threshold %.3f. "
                "Using top %d retrieved chunks instead.",
                config.RAG_SCORE_THRESHOLD,
                len(matches),
            )

            relevant = [m["text"] for m in matches]

        logger.info(
            "Returning %d context chunk(s).",
            len(relevant),
        )

        logger.info("=" * 80)

        return relevant