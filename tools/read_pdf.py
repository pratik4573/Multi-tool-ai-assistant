"""Read-PDF tool — extracts and returns text content from a PDF file path."""

from __future__ import annotations

from pathlib import Path

from src.utils.text_utils import truncate

SPEC = {
    "name": "read_pdf",
    "description": "Extract and return the text content of a PDF file at a given file path.",
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Absolute or relative path to the PDF file."}
        },
        "required": ["file_path"],
    },
}


def run(args: dict) -> str:
    file_path = str(args.get("file_path", "")).strip()
    if not file_path:
        return "Error: no file_path provided."

    path = Path(file_path)
    if not path.exists() or path.suffix.lower() != ".pdf":
        return f"Error: '{file_path}' is not an existing PDF file."

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages_text = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(pages_text).strip()
        if not text:
            return "The PDF was read but no extractable text was found (it may be scanned/image-only)."
        return truncate(text, max_chars=6000)
    except Exception as exc:  # noqa: BLE001
        return f"Error: failed to read PDF ({exc})."
