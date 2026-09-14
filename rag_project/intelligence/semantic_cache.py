from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import struct
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from rag_project.retrieval.hybrid_retriever import RetrievalHit


class SemanticRetrievalCache:
    """Embedding-aware retrieval cache with explicit dependency injection."""

    SCHEMA_VERSION = 5

    def __init__(
        self,
        db_path: str | Path,
        ttl_seconds: float = 7 * 24 * 60 * 60,
        *,
        embed_query: Callable[[str], Sequence[float]] | None = None,
        similarity_threshold: float = 0.95,
        max_entries: int = 10_000,
        expected_dimension: int = 768,
        cache_namespace: str = "legacy",
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embed_query = embed_query
        self.ttl_seconds = float(ttl_seconds)
        self.similarity_threshold = float(similarity_threshold)
        self.max_entries = max(1, int(max_entries))
        self.expected_dimension = int(expected_dimension)
        self.cache_namespace = str(cache_namespace or "legacy")
        if not 0.0 < self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in (0, 1].")
        with sqlite3.connect(self.db_path) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS semantic_retrieval_cache (cache_id INTEGER PRIMARY KEY AUTOINCREMENT, query TEXT NOT NULL, embedding BLOB NOT NULL, dimension INTEGER NOT NULL, namespace TEXT NOT NULL DEFAULT 'legacy', payload TEXT NOT NULL, created REAL NOT NULL, accessed REAL NOT NULL, hits INTEGER NOT NULL DEFAULT 0)")
            columns = {str(row[1]) for row in db.execute("PRAGMA table_info(semantic_retrieval_cache)").fetchall()}
            if "namespace" not in columns:
                db.execute("ALTER TABLE semantic_retrieval_cache ADD COLUMN namespace TEXT NOT NULL DEFAULT 'legacy'")
            db.execute("CREATE INDEX IF NOT EXISTS idx_semantic_cache_namespace ON semantic_retrieval_cache(namespace, created, cache_id)")
            db.execute("CREATE INDEX IF NOT EXISTS idx_semantic_cache_created ON semantic_retrieval_cache(created, cache_id)")
            db.execute("CREATE INDEX IF NOT EXISTS idx_semantic_cache_accessed ON semantic_retrieval_cache(accessed, cache_id)")
            db.commit()

    @staticmethod
    def _identity_namespace(identity: Any | None, fallback: str = "legacy") -> str:
        """Return a stable namespace derived solely from embedding identity."""
        if identity is None:
            return str(fallback or "legacy")
        if isinstance(identity, dict):
            payload = {key: identity.get(key) for key in ("provider", "model", "model_version", "dimension", "fingerprint", "embedding_id")}
        else:
            payload = {key: getattr(identity, key, None) for key in ("provider", "model", "model_version", "dimension", "fingerprint", "embedding_id")}
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return f"embedding:{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]}"

    @staticmethod
    def _pack(values: Sequence[float]) -> bytes:
        return struct.pack(f"<{len(values)}f", *[float(v) for v in values])

    @staticmethod
    def _unpack(blob: bytes, dimension: int) -> list[float]:
        if not blob or dimension <= 0 or len(blob) != 4 * int(dimension):
            return []
        return list(struct.unpack(f"<{dimension}f", blob))

    @staticmethod
    def cosine(left: Sequence[float], right: Sequence[float]) -> float:
        if len(left) != len(right) or not left:
            return 0.0
        dot = sum(float(a) * float(b) for a, b in zip(left, right, strict=True))
        left_norm = math.sqrt(sum(float(a) * float(a) for a in left))
        right_norm = math.sqrt(sum(float(b) * float(b) for b in right))
        denominator = left_norm * right_norm
        return dot / denominator if denominator > 1e-12 else 0.0

    def _query_embedding(self, query: str) -> list[float] | None:
        if not callable(self.embed_query):
            return None
        try:
            vector = [float(value) for value in self.embed_query(str(query))]
        except Exception:
            return None
        if not vector or any(not math.isfinite(v) for v in vector):
            return None
        if self.expected_dimension > 0 and len(vector) != self.expected_dimension:
            return None
        if self.expected_dimension <= 0:
            self.expected_dimension = len(vector)
        return vector

    def get(self, query: str, *, scope_active: bool = False) -> tuple[list[RetrievalHit], dict[str, Any]] | None:
        if scope_active:
            return None
        vector = self._query_embedding(query)
        if vector is None:
            return None
        now = time.time()
        with sqlite3.connect(self.db_path) as db:
            if self.ttl_seconds <= 0:
                db.execute("DELETE FROM semantic_retrieval_cache WHERE namespace=?", (self.cache_namespace,))
                db.commit()
                return None
            stale_before = now - self.ttl_seconds
            db.execute("DELETE FROM semantic_retrieval_cache WHERE namespace=? AND created < ?", (self.cache_namespace, stale_before))
            rows = db.execute("SELECT cache_id, embedding, dimension, payload FROM semantic_retrieval_cache WHERE namespace=?", (self.cache_namespace,)).fetchall()
            best: tuple[float, tuple[Any, ...]] | None = None
            for row in rows:
                if int(row[2]) != len(vector):
                    continue
                cached_vector = self._unpack(row[1], int(row[2]))
                similarity = self.cosine(vector, cached_vector)
                if similarity >= self.similarity_threshold and (best is None or similarity > best[0] or (math.isclose(similarity, best[0]) and int(row[0]) > int(best[1][0]))):
                    best = (similarity, row)
            if best is None:
                db.commit()
                return None
            similarity, row = best
            try:
                payload = json.loads(row[3])
            except (TypeError, ValueError, json.JSONDecodeError):
                db.execute("DELETE FROM semantic_retrieval_cache WHERE cache_id=?", (row[0],))
                db.commit()
                return None
            db.execute("UPDATE semantic_retrieval_cache SET accessed=?, hits=hits+1 WHERE cache_id=?", (now, row[0]))
            db.commit()
        restored = self.restore(payload)
        return restored, {"similarity": float(similarity), "hits": len(restored), "cache_id": int(row[0]), "query": "", "namespace": self.cache_namespace}

    def put(self, query: str, hits: Sequence[RetrievalHit], *, scope_active: bool = False) -> bool:
        if scope_active:
            return False
        vector = self._query_embedding(query)
        if vector is None or self.ttl_seconds <= 0:
            return False
        payload = [{"doc_id": hit.doc_id, "text": hit.text, "metadata": hit.metadata, "score": float(hit.score), "vector_score": float(hit.vector_score), "lexical_score": float(hit.lexical_score)} for hit in list(hits)[:24]]
        now = time.time()
        with sqlite3.connect(self.db_path) as db:
            db.execute("DELETE FROM semantic_retrieval_cache WHERE namespace=? AND lower(query)=lower(?)", (self.cache_namespace, str(query)[:3000]))
            db.execute("INSERT INTO semantic_retrieval_cache (query,embedding,dimension,namespace,payload,created,accessed,hits) VALUES(?,?,?,?,?,?,?,0)", (str(query)[:3000], self._pack(vector), len(vector), self.cache_namespace, json.dumps(payload, ensure_ascii=False), now, now))
            overflow = db.execute("SELECT cache_id FROM semantic_retrieval_cache WHERE namespace=? ORDER BY accessed DESC, cache_id DESC LIMIT -1 OFFSET ?", (self.cache_namespace, self.max_entries)).fetchall()
            if overflow:
                db.executemany("DELETE FROM semantic_retrieval_cache WHERE cache_id=?", overflow)
            db.commit()
        return True

    def delete_all(self) -> None:
        with sqlite3.connect(self.db_path) as db:
            db.execute("DELETE FROM semantic_retrieval_cache WHERE namespace=?", (self.cache_namespace,))
            db.commit()

    def stats(self) -> dict[str, Any]:
        with sqlite3.connect(self.db_path) as db:
            count = int(db.execute("SELECT COUNT(*) FROM semantic_retrieval_cache WHERE namespace=?", (self.cache_namespace,)).fetchone()[0])
            hits = int(db.execute("SELECT COALESCE(SUM(hits),0) FROM semantic_retrieval_cache WHERE namespace=?", (self.cache_namespace,)).fetchone()[0])
        return {"schema_version": self.SCHEMA_VERSION, "entries": count, "max_entries": self.max_entries, "ttl_seconds": self.ttl_seconds, "similarity_threshold": self.similarity_threshold, "expected_dimension": self.expected_dimension, "recorded_hits": hits, "semantic": True, "namespace": self.cache_namespace}

    @staticmethod
    def restore(payload: Sequence[Any]) -> list[RetrievalHit]:
        restored: list[RetrievalHit] = []
        for item in payload or ():
            if isinstance(item, RetrievalHit):
                restored.append(item)
                continue
            try:
                restored.append(RetrievalHit(str(item.get("doc_id", "unknown")), str(item.get("text", "")), dict(item.get("metadata") or {}), float(item.get("score", 0.0)), float(item.get("vector_score", 0.0)), float(item.get("lexical_score", 0.0))))
            except (AttributeError, TypeError, ValueError):
                continue
        return restored


def create_for_system(system: Any, *, db_path: str | Path | None = None, **kwargs: Any) -> SemanticRetrievalCache:
    """Create an isolated cache from a concrete runtime system; no global binding."""
    settings = getattr(system, "settings", None)
    if db_path is None:
        configured = getattr(settings, "semantic_cache_db_path", None) or getattr(settings, "cache_db_path", None)
        root = getattr(settings, "project_root", None)
        db_path = configured or (Path(root) / "data" / "semantic_cache.sqlite3" if root else Path("semantic_cache.sqlite3"))
    embedder = getattr(system, "embedding_service", None)
    embed_query = kwargs.pop("embed_query", getattr(embedder, "embed_query", None))
    identity = kwargs.pop("identity", getattr(embedder, "identity", None))
    dimension = kwargs.pop("expected_dimension", getattr(embedder, "dimension", 768) or 768)
    namespace = kwargs.pop("cache_namespace", SemanticRetrievalCache._identity_namespace(identity))
    return SemanticRetrievalCache(db_path, embed_query=embed_query, expected_dimension=int(dimension), cache_namespace=namespace, **kwargs)


__all__ = ["SemanticRetrievalCache", "create_for_system"]
