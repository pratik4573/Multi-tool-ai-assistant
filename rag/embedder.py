"""
Embedding model wrapper (BAAI/bge-small-en-v1.5 by default).

Wrapping sentence-transformers behind this class means swapping the
embedding model later only requires changing config.EMBEDDING_MODEL_NAME —
no other file needs to change.
"""

from __future__ import annotations

import os
from functools import lru_cache

# Prevent transformers from importing TensorFlow if it is installed, since the
# project relies on PyTorch-based sentence-transformers and the installed TF
# environment may be incompatible with the installed protobuf version.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Embedder:
    """Thin wrapper around a sentence-transformers embedding model."""

    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # local import: heavy dep
        except Exception as exc:
            raise RuntimeError(
                "Failed to import sentence-transformers; ensure compatible dependencies "
                "are installed (try pinning protobuf==4.23.4 and reinstalling). "
                f"Original error: {exc}"
            ) from exc

        self.model_name = model_name or config.EMBEDDING_MODEL_NAME
        self.device = device or config.EMBEDDING_DEVICE

        # FIX: the previous version unconditionally prefixed every query
        # with the BGE-specific instruction string, regardless of which
        # model was actually configured. BGE models ("BAAI/bge-...") are
        # trained to interpret that instruction text specially, but most
        # other sentence-transformers models (e.g. all-MiniLM-L6-v2) are
        # NOT — they just mean-pool every token, instruction text included,
        # into the embedding. That silently diluted every query embedding
        # with ~10 tokens of boilerplate that has no counterpart in the
        # (unprefixed) passage embeddings, skewing similarity scores in an
        # unpredictable way. We now only apply the prefix for models that
        # are actually BGE-family, detected from the model name.
        self._use_bge_query_prefix = "bge" in self.model_name.lower()

        logger.info("Loading embedding model '%s' on %s...", self.model_name, self.device)
        self._model = SentenceTransformer(self.model_name, device=self.device)
        logger.info("Embedding model ready.")

        if not self._use_bge_query_prefix:
            logger.info(
                "Model '%s' is not BGE-family; queries will be embedded "
                "without an instruction prefix.",
                self.model_name,
            )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # bge models recommend a "passage" framing for documents (no prefix needed
        # for bge-small specifically, but normalization matters for cosine sim).
        vectors = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        # FIX: only BGE-family models get the instruction-style prefix.
        # Other models embed the raw query, matching how embed_texts()
        # embeds raw passages, so query and passage embeddings stay
        # comparable.
        if self._use_bge_query_prefix:
            text = f"Represent this sentence for searching relevant passages: {query}"
        else:
            text = query

        vector = self._model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False
        )
        return vector[0].tolist()


@lru_cache(maxsize=1)
def get_shared_embedder() -> Embedder:
    """Module-level singleton so the (relatively heavy) model loads only once."""
    return Embedder()