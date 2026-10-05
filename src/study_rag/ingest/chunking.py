"""Chunking: turning a Document into retrievable Chunks.

EXERCISE 2. Before writing code, decide and write down (in the README's
design-decisions table) which strategy you choose and why:
  - fixed-size windows with overlap
  - split on markdown headings, then by size
  - split on sentences/paragraphs
What does your actual course material look like? Slides, dense PDFs, notes?
"""

from __future__ import annotations

from study_rag.models import Chunk, Document


def chunk_document(doc: Document, size: int = 800, overlap: int = 120) -> list[Chunk]:
    """Split `doc.text` into Chunks.

    Requirements (enforced by tests/test_chunking.py):
      - raise ValueError if overlap >= size
      - chunk_id unique per document, e.g. f"{doc.doc_id}:{i}"; position = 0, 1, 2, ...
      - copy doc_id and source from the document
      - no chunk longer than roughly size + overlap
      - if you split on headings: `section` = heading path, e.g. "Week 1 > Bias"
    """
    raise NotImplementedError("exercise 2")
