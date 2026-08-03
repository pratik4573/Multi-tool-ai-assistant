"""POST /upload — upload and index a personal document for RAG."""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

import config
from src.api.schemas import UploadResponse
from src.rag.retriever import Retriever
from src.utils.exceptions import RAGError
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["upload"])

_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever

_ALLOWED_SUFFIXES = {".md", ".txt", ".pdf"}


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    filename = file.filename or "uploaded_file"
    suffix = Path(filename).suffix.lower()

    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(_ALLOWED_SUFFIXES)}",
        )

    dest_path = config.UPLOADS_DIR / filename
    with dest_path.open("wb") as out_file:
        shutil.copyfileobj(file.file, out_file)

    try:
        chunks_indexed = get_retriever().ingest_path(dest_path)
    except RAGError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return UploadResponse(
        filename=filename,
        chunks_indexed=chunks_indexed,
        message=f"'{filename}' indexed into {chunks_indexed} chunk(s) for personal knowledge retrieval.",
    )
