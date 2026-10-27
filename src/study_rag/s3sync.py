"""Index lifecycle on AWS.

Upload a file under s3://<bucket>/docs/<course>/ -> S3 event -> Lambda -> reindex():
download all docs, rebuild the index (embedding cache makes this incremental in
cost: only new/changed chunks hit the model), upload index + cache back to S3.

Every warm Lambda instance checks the index object's ETag at most once a minute
(IndexWatcher) and reloads when it changed, so all instances converge.

Concurrency: two uploads at once can trigger two full rebuilds; both read the
full docs prefix, so the last writer still publishes a complete index.
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

from study_rag.config import Settings

log = logging.getLogger(__name__)


def _split(uri: str) -> tuple[str, str]:
    bucket, _, key = uri.removeprefix("s3://").partition("/")
    return bucket, key


def _s3():
    import boto3

    return boto3.client("s3")


def download_prefix(uri: str, dest: Path) -> int:
    bucket, prefix = _split(uri)
    s3, n = _s3(), 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(prefix):].lstrip("/")
            if not rel or rel.endswith("/"):
                continue
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, obj["Key"], str(target))
            n += 1
    return n


def _try_download(uri: str, dest: str) -> bool:
    from botocore.exceptions import ClientError

    bucket, key = _split(uri)
    try:
        _s3().download_file(bucket, key, dest)
        return True
    except ClientError:
        return False


def reindex(settings: Settings) -> dict[str, int]:
    """Rebuild the index from the docs prefix and publish it. Returns counts."""
    from study_rag.cache import CachedEmbedder
    from study_rag.pipeline import StudyRAG

    if not (settings.docs_s3_uri and settings.index_s3_uri):
        raise RuntimeError("STUDY_RAG_DOCS_S3 and STUDY_RAG_INDEX_S3 must be set")
    with tempfile.TemporaryDirectory() as tmp:
        docs = Path(tmp) / "docs"
        n_files = download_prefix(settings.docs_s3_uri, docs)
        if settings.cache_s3_uri:
            _try_download(settings.cache_s3_uri, settings.embed_cache_path)
        # Don't download/check the old index: we're replacing it.
        build = settings.model_copy(update={"index_s3_uri": ""})
        rag = StudyRAG.from_settings(build, check_index=False)
        n_chunks = rag.ingest(docs)

        bucket, key = _split(settings.index_s3_uri)
        _s3().upload_file(settings.index_path, bucket, key)
        result = {"files": n_files, "chunks": n_chunks}
        if isinstance(rag.embedder, CachedEmbedder):
            result["embedded"], result["cache_hits"] = rag.embedder.misses, rag.embedder.hits
            if settings.cache_s3_uri and Path(settings.embed_cache_path).exists():
                cb, ck = _split(settings.cache_s3_uri)
                _s3().upload_file(settings.embed_cache_path, cb, ck)
    log.info("reindex done: %s", result)
    return result


class IndexWatcher:
    """Cheap staleness check: one HEAD request per `interval` seconds, per instance."""

    def __init__(self, index_s3_uri: str, interval: float = 60.0) -> None:
        self.uri, self.interval = index_s3_uri, interval
        self._etag: str | None = None
        self._checked = 0.0

    def _current_etag(self) -> str | None:
        bucket, key = _split(self.uri)
        try:
            return _s3().head_object(Bucket=bucket, Key=key)["ETag"]
        except Exception:  # transient S3 error: keep serving the current index
            log.warning("index ETag check failed", exc_info=True)
            return self._etag

    def mark_loaded(self) -> None:
        self._etag, self._checked = self._current_etag(), time.monotonic()

    def changed(self) -> bool:
        if not self.uri or time.monotonic() - self._checked < self.interval:
            return False
        self._checked = time.monotonic()
        etag = self._current_etag()
        return etag is not None and etag != self._etag
