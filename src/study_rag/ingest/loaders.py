from __future__ import annotations

import hashlib
import re
from pathlib import Path

from study_rag.models import Document

SUPPORTED = {".md", ".txt", ".pdf"}


def _doc_id(path: Path) -> str:
    return hashlib.sha1(str(path).encode()).hexdigest()[:12]


def _clean_layout(text: str) -> str:
    # Layout mode pads words with runs of spaces to mimic the page; collapse them, and
    # keep at most one blank line so paragraph breaks survive for the chunker.
    lines = (" ".join(line.split()) for line in text.splitlines())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    # Default extraction returns one word per line for many exported PDFs (Google Docs,
    # Word), which leaves the chunker no paragraph breaks. Layout mode keeps real lines.
    # Page boundaries are marked so chunks can cite "p. N".
    return "\n\n".join(
        f"[page {i}]\n{_clean_layout(page.extract_text(extraction_mode='layout') or '')}"
        for i, page in enumerate(reader.pages, start=1)
    )


def load_file(path: Path, root: Path | None = None) -> Document:
    text = _read_pdf(path) if path.suffix.lower() == ".pdf" else path.read_text(encoding="utf-8")
    rel = path.relative_to(root) if root else path
    # A subfolder, if any, doubles as metadata (<course>/<file>); flat files have no course.
    course = rel.parts[0] if len(rel.parts) > 1 else ""
    return Document(
        doc_id=_doc_id(rel),
        title=path.stem.replace("_", " ").replace("-", " "),
        text=text,
        source=str(rel),
        metadata={"course": course},
    )


def load_directory(root: str | Path) -> list[Document]:
    root = Path(root)
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED and p.is_file())
    return [load_file(p, root) for p in files]
