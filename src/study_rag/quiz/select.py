"""Which chunks a quiz is built from. Pure and deterministic: no I/O, no randomness, so
the same index and arguments always give the same quiz inputs (testable, cacheable)."""

from __future__ import annotations

from collections import defaultdict

from study_rag.models import Chunk, ScoredChunk
from study_rag.retrieval import course_of

# Below this, a chunk is usually a heading or a one-line slide: nothing to ask about.
MIN_CHARS = 200


def select_chunks(
    chunks: list[Chunk], n: int, course: str | None = None, min_chars: int = MIN_CHARS
) -> list[Chunk]:
    """Pick up to `n` chunks spread across sections, so a quiz covers the material instead
    of five variations of the first heading.

    Round-robin over (source, section) groups: take the first unused chunk of each group in
    turn until `n` are chosen. Groups and the chunks inside them keep index order.
    """
    groups: dict[tuple[str, str], list[Chunk]] = defaultdict(list)
    for c in chunks:
        if len(c.text.strip()) < min_chars:
            continue
        if course and course_of(ScoredChunk(c, 0.0)) != course:
            continue
        groups[(c.source, c.section)].append(c)

    picked: list[Chunk] = []
    queues = [list(g) for g in groups.values()]
    while len(picked) < n and any(queues):
        for q in queues:
            if q and len(picked) < n:
                picked.append(q.pop(0))
    return picked
