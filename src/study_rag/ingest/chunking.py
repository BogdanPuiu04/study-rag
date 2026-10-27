"""Chunking: turning a Document into retrievable Chunks.

Strategy: split on Markdown headings first, so a chunk stays about one topic and its
`section` can be shown as a citation ("Week 1 > Bias"); then split sections that are
longer than `size` on paragraph boundaries, packing paragraphs greedily. Pieces cut from
the same section start with the last `overlap` characters of the previous piece, so an
idea that straddles a boundary is whole in at least one of them. A single paragraph
longer than `size` is hard-split into fixed windows.
"""

from __future__ import annotations

import re

from study_rag.models import Chunk, Document

HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


def chunk_document(doc: Document, size: int = 800, overlap: int = 120) -> list[Chunk]:
    """Split a Document into retrievable Chunks.

    `size` is the maximum number of characters per chunk, `overlap` is the number of
    characters to repeat from the previous chunk.
    """
    # With overlap >= size the window would never advance.
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    chunks: list[Chunk] = []
    path: list[str] = []  # the heading stack: path[n] is the open heading at level n+1
    lines: list[str] = []  # lines of the section being collected

    def flush() -> None:
        text = "\n".join(lines).strip()
        lines.clear()
        if not text:  # a heading directly followed by another heading
            return
        for piece in _split(text, size, overlap):
            i = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}:{i}",
                    doc_id=doc.doc_id,
                    text=piece,
                    source=doc.source,
                    section=" > ".join(path),
                    position=i,
                )
            )

    for line in doc.text.splitlines():
        heading = HEADING.match(line)
        if not heading:
            lines.append(line)
            continue
        flush()  # the text so far belongs to the previous heading path
        level = len(heading.group(1))
        del path[level - 1 :]
        path.append(heading.group(2).strip())
    flush()
    return chunks


def _split(text: str, size: int, overlap: int) -> list[str]:
    """Cut one section into pieces of at most `size` (+ overlap prefix) characters."""
    pieces: list[str] = []
    current = ""
    for paragraph in PARAGRAPH_BREAK.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) > size:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(_windows(paragraph, size, overlap))
            continue
        if current and len(current) + 2 + len(paragraph) <= size:
            current += "\n\n" + paragraph
            continue
        if current:
            pieces.append(current)
        tail = pieces[-1][max(0, len(pieces[-1]) - overlap) :] if pieces else ""
        current = f"{tail}\n\n{paragraph}" if tail else paragraph
    if current:
        pieces.append(current)
    return pieces


def _windows(text: str, size: int, overlap: int) -> list[str]:
    """Fixed-size windows for a paragraph that cannot be split any other way."""
    step = size - overlap
    windows = []
    start = 0
    while True:
        windows.append(text[start : start + size])
        if start + size >= len(text):
            return windows
        start += step
