"""
Loads and extracts raw text from personal documents (.md, .txt, .pdf).

Used both for the initial personal_docs/ ingestion and for files uploaded
at runtime via the /upload endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import config
from src.utils.exceptions import DocumentNotFoundError, RAGError
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


@dataclass
class LoadedDocument:
    path: str
    filename: str
    text: str


def discover_personal_documents() -> list[Path]:
    """Find every supported personal document under PERSONAL_DOCS_DIR (recursive)."""
    root = config.PERSONAL_DOCS_DIR
    return [
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in _SUPPORTED_SUFFIXES
    ]


def load_document(path: Path) -> LoadedDocument:
    if not path.exists():
        raise DocumentNotFoundError(f"Document not found: {path}")

    suffix = path.suffix.lower()
    try:
        if suffix in (".md", ".txt"):
            text = path.read_text(encoding="utf-8", errors="ignore")
        elif suffix == ".pdf":
            text = _extract_pdf_text(path)
        else:
            raise RAGError(f"Unsupported document type: {suffix}")
    except Exception as exc:  # noqa: BLE001
        raise RAGError(f"Failed to read document '{path}': {exc}") from exc

    return LoadedDocument(path=str(path), filename=path.name, text=text)


def _extract_pdf_text(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)
