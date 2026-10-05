import pytest

from study_rag.ingest.chunking import chunk_document
from study_rag.models import Document


def doc(text: str) -> Document:
    return Document(doc_id="d1", title="t", text=text, source="notes.md")


def test_headings_become_section_paths():
    chunks = chunk_document(doc("# Week 1\n\n## Bias\n\nBias text.\n\n## Variance\n\nVar text."))
    assert [c.section for c in chunks] == ["Week 1 > Bias", "Week 1 > Variance"]


def test_long_sections_are_split_within_size_with_overlap():
    body = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(10))
    chunks = chunk_document(doc(body), size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c.text) <= 300 + 50 + 2 for c in chunks)


def test_chunk_ids_are_unique_and_ordered():
    chunks = chunk_document(doc("# A\n\none\n\n# B\n\ntwo\n\n# C\n\nthree"))
    assert [c.position for c in chunks] == [0, 1, 2]
    assert len({c.chunk_id for c in chunks}) == 3


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_document(doc("x"), size=100, overlap=100)
