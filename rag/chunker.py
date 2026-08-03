"""Markdown-header-aware text chunker for RAG ingestion.

Splits text along markdown header boundaries first (##, ###, ...) so a
section's heading and its own body/details always land in the same chunk —
this is what previously let a header like '## Bachelor of Science (PCM)' get
separated from its own 'Graduation: July 2024' line, hurting retrieval for
that section. Falls back to the original paragraph-based, size-bounded
splitting *within* a section only if that section alone is too large to fit
in one chunk.

Additionally, any section that is just a bare header line with no body text
of its own (e.g. a top-level title like '# Education' that only introduces
subheadings) is merged into the section that follows it. Without this, such
a header becomes its own near-empty chunk whose entire embedded content is
just that one word/phrase — which can score deceptively high against
queries matching the header text, while containing none of the actual
answer, and crowds out the real content chunk during retrieval.
"""

from __future__ import annotations

import re

import config

_HEADER_RE = re.compile(r"(?m)^#{1,6}\s+.*$")


def _split_into_sections(text: str) -> list[str]:
    """Split on markdown header lines, keeping each header attached to the
    content that follows it up to the next header (or end of text)."""
    matches = list(_HEADER_RE.finditer(text))
    if not matches:
        return [text]

    sections: list[str] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = text[start:end].strip()
        if section:
            sections.append(section)

    # Any text before the first header (rare — e.g. a stray top-level title
    # with no body of its own) gets folded into the first section so it
    # isn't dropped.
    preamble = text[: matches[0].start()].strip()
    if preamble and sections:
        sections[0] = f"{preamble}\n\n{sections[0]}"
    elif preamble:
        sections = [preamble]

    return _merge_headerless_sections(sections)


def _is_bare_header(section: str) -> bool:
    """True if a section is just a header line with no body text of its own
    (e.g. a title that only introduces subheadings, like '# Education'
    immediately followed by '## Bachelor of Science ...')."""
    lines = section.split("\n", 1)
    return bool(_HEADER_RE.match(lines[0])) and (
        len(lines) == 1 or not lines[1].strip()
    )


def _merge_headerless_sections(sections: list[str]) -> list[str]:
    """Fold any bare-header section into the section that follows it, so a
    lone title never becomes its own near-empty, low-information chunk.

    Without this, '# Education' on its own becomes a chunk whose entire
    embedded content is the word "education" — which scores deceptively
    high against queries like "what is your education?" while containing
    none of the actual answer, crowding out the real content chunk.
    """
    result: list[str] = []
    i = 0
    while i < len(sections):
        section = sections[i]

        if _is_bare_header(section) and i + 1 < len(sections):
            result.append(f"{section}\n\n{sections[i + 1]}")
            i += 2
        else:
            result.append(section)
            i += 1

    return result


def _find_word_boundary(text: str, start: int, ideal_end: int) -> int:
    """Return an end index <= ideal_end that doesn't fall inside a word.

    FIX: the previous hard-split walked a fixed character budget with no
    regard for word boundaries, which could slice a word in half (e.g.
    "Extractor" -> "Extra" / "ctor" landing as the start of the next
    chunk). This steps back to the nearest preceding whitespace. If no
    whitespace exists in range (one very long unbroken token), it falls
    back to the original hard cut rather than looping forever.
    """
    if ideal_end >= len(text):
        return len(text)

    end = ideal_end
    while end > start and not text[end].isspace():
        end -= 1

    if end == start:
        return ideal_end

    return end


def _chunk_paragraphs(section: str, chunk_size: int, overlap: int) -> list[str]:
    """Paragraph-based, size-bounded split for a single oversized section.
    The section's own header line is re-prefixed onto every piece produced
    here, so each sub-chunk keeps its heading context even after further
    splitting."""
    lines = section.split("\n", 1)
    header = lines[0].strip() if lines[0].lstrip().startswith("#") else ""
    body = section[len(lines[0]):].strip() if header else section

    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    def _flush() -> None:
        nonlocal current
        if current:
            chunks.append(f"{header}\n\n{current}" if header else current)
            current = ""

    # Reserve room in the budget for the header we'll re-prefix onto each piece.
    budget = chunk_size - (len(header) + 2 if header else 0)
    budget = max(budget, 1)

    for para in paragraphs:
        if len(current) + len(para) + 2 <= budget:
            current = f"{current}\n\n{para}" if current else para
            continue

        _flush()

        if len(para) <= budget:
            current = para
        else:
            # FIX: word-boundary-safe hard split for an oversized paragraph,
            # instead of cutting at a raw character index mid-word.
            start = 0
            while start < len(para):
                ideal_end = start + budget
                end = _find_word_boundary(para, start, min(ideal_end, len(para)))
                piece = para[start:end].strip()
                if piece:
                    chunks.append(f"{header}\n\n{piece}" if header else piece)
                start = end - overlap if end - overlap > start else end

    _flush()
    return chunks


def chunk_text(
    text: str,
    chunk_size: int = config.RAG_CHUNK_SIZE,
    overlap: int = config.RAG_CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks, respecting markdown section
    boundaries so a header and its own content are never separated.

    Falls back to the original paragraph-based, size-bounded splitting for
    any individual section too large to fit in one chunk.
    """
    text = text.strip()
    if not text:
        return []

    sections = _split_into_sections(text)
    all_chunks: list[str] = []

    for section in sections:
        if len(section) <= chunk_size:
            group = [section]
        else:
            group = _chunk_paragraphs(section, chunk_size, overlap)

        # FIX: overlap is now applied ONLY within a group of chunks that
        # came from splitting a single oversized section — that's the only
        # case where two chunks are genuinely continuous content and
        # benefit from a bridging tail. Previously, overlap was applied
        # globally across the flattened chunk list, which meant the tail
        # of one section (e.g. "## Name / Pratik Pandey") got glued onto
        # the front of the NEXT, unrelated section (e.g. "## Introduction
        # / ..."). That produced two separate stored chunks that were
        # mostly duplicates of each other, wasting retrieval slots on
        # redundant content instead of surfacing distinct information.
        if overlap > 0 and len(group) > 1:
            bridged = [group[0]]
            for i in range(1, len(group)):
                prev_tail = group[i - 1][-overlap:]
                bridged.append(f"{prev_tail} {group[i]}".strip())
            group = bridged

        all_chunks.extend(group)

    return all_chunks