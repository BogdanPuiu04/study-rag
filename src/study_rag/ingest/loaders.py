from __future__ import annotations

import hashlib
from pathlib import Path

from study_rag.models import Document

SUPPORTED = {".md", ".txt", ".pdf"}


def _doc_id(path: Path) -> str:
    return hashlib.sha1(str(path).encode()).hexdigest()[:12]


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    # Mark page boundaries so chunks can cite "p. N".
    return "\n\n".join(
        f"[page {i}]\n{page.extract_text() or ''}" for i, page in enumerate(reader.pages, start=1)
    )


def load_file(path: Path, root: Path | None = None) -> Document:
    text = _read_pdf(path) if path.suffix.lower() == ".pdf" else path.read_text(encoding="utf-8")
    rel = path.relative_to(root) if root else path
    # Folder layout doubles as metadata: data/<course>/<file>
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
