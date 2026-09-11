from __future__ import annotations

import json
import math
import sqlite3
import struct
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from rag_project.retrieval.hybrid_retriever import RetrievalHit


class SemanticRetrievalCache:
    """Embedding-aware retrieval cache with bounded TTL and cosine matching.

    Production defaults intentionally mirror the MedEvidence Pro plan:
    cosine similarity >= 0.95, seven-day TTL, and at most 10,000 entries.
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        db_path: str | Path,
        *,
        embed_query: Callable[[str], Sequence[float]] | None,
        ttl_seconds: float = 7 * 24 * 60 * 60,
        similarity_threshold: float = 0.95,
        max_entries: int = 10_000,
        expected_dimension: int = 768,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embed_query = embed_query
        self.ttl_seconds = float(ttl_seconds)
        self.similarity_threshold = float(similarity_threshold)
        self.max_entries = max(1, int(max_entries))
        self.expected_dimension = int(expected_dimension)
        if not 0.0 < self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in (0, 1].")
        with sqlite3.connect(self.db_path) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                """CREATE TABLE IF NOT EXISTS semantic_retrieval_cache (
                    cache_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    dimension INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    created REAL NOT NULL,
                    accessed REAL NOT NULL,
                    hits INTEGER NOT NULL DEFAULT 0
                )"""
            )
            db.execute("CREATE INDEX IF NOT EXISTS idx_semantic_cache_created ON semantic_retrieval_cache(created)")
            db.commit()

    @staticmethod
    def _pack(values: Sequence[float]) -> bytes:
        return struct.pack(f"<{len(values)}f", *[float(v) for v in values])

    @staticmethod
    def _unpack(blob: bytes, dimension: int) -> list[float]:
        if not blob or dimension <= 0:
            return []
        expected = 4 * int(dimension)
        if len(blob) != expected:
            return []
        return list(struct.unpack(f"<{dimension}f", blob))

    @staticmethod
    def cosine(left: Sequence[float], right: Sequence[float]) -> float:
        if len(left) != len(right) or not left:
            return 0.0
        dot = 0.0
        left_norm = 0.0
        right_norm = 0.0
        for a, b in zip(left, right, strict=True):
            fa = float(a)
            fb = float(b)
            dot += fa * fb
            left_norm += fa * fa
            right_norm += fb * fb
        denominator = math.sqrt(left_norm) * math.sqrt(right_norm)
        if denominator <= 1e-12:
            return 0.0
        return dot / denominator

    def _query_embedding(self, query: str) -> list[float] | None:
        if self.embed_query is None:
            return None
        try:
            vector = [float(value) for value in self.embed_query(str(query))]
        except Exception:
            return None
        if not vector or any(not math.isfinite(v) for v in vector):
            return None
        if len(vector) != self.expected_dimension:
            # A deterministic test embedding is allowed to use another dimension;
            # production embeddings must satisfy the plan's 768-dim contract.
            if not str(getattr(self, "mode", "production")) == "test":
                return None
        return vector

    def get(self, query: str) -> tuple[list[RetrievalHit], dict[str, Any]] | None:
        vector = self._query_embedding(query)
        if vector is None:
            return None
        now = time.time()
        with sqlite3.connect(self.db_path) as db:
            stale_before = now - self.ttl_seconds if self.ttl_seconds > 0 else None
            if stale_before is not None:
                db.execute("DELETE FROM semantic_retrieval_cache WHERE created < ?", (stale_before,))
            rows = db.execute(
                "SELECT cache_id, embedding, dimension, payload, created, accessed, hits FROM semantic_retrieval_cache"
            ).fetchall()
            best: tuple[float, tuple[Any, ...]] | None = None
            for row in rows:
                cached_vector = self._unpack(row[1], int(row[2]))
                similarity = self.cosine(vector, cached_vector)
                if similarity >= self.similarity_threshold and (best is None or similarity > best[0]):
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
            db.execute(
                "UPDATE semantic_retrieval_cache SET accessed=?, hits=hits+1 WHERE cache_id=?",
                (now, row[0]),
            )
            db.commit()
        return self.restore(payload), {
            "similarity": round(float(similarity), 6),
            "created": float(row[4]),
            "age_seconds": max(0.0, now - float(row[4])),
            "hits": int(row[6]) + 1,
        }

    def put(self, query: str, hits: Sequence[RetrievalHit]) -> bool:
        vector = self._query_embedding(query)
        if vector is None:
            return False
        payload = [
            {
                "doc_id": hit.doc_id,
                "text": hit.text,
                "metadata": hit.metadata,
                "score": float(hit.score),
                "vector_score": float(hit.vector_score),
                "lexical_score": float(hit.lexical_score),
            }
            for hit in list(hits)[:24]
        ]
        now = time.time()
        with sqlite3.connect(self.db_path) as db:
            db.execute(
                "INSERT INTO semantic_retrieval_cache(query,embedding,dimension,payload,created,accessed,hits) VALUES(?,?,?,?,?,?,0)",
                (str(query)[:3000], self._pack(vector), len(vector), json.dumps(payload, ensure_ascii=False), now, now),
            )
            overflow = db.execute(
                "SELECT cache_id FROM semantic_retrieval_cache ORDER BY accessed DESC LIMIT -1 OFFSET ?",
                (self.max_entries,),
            ).fetchall()
            if overflow:
                db.executemany("DELETE FROM semantic_retrieval_cache WHERE cache_id=?", overflow)
            db.commit()
        return True

    def delete_all(self) -> None:
        with sqlite3.connect(self.db_path) as db:
            db.execute("DELETE FROM semantic_retrieval_cache")
            db.commit()

    def stats(self) -> dict[str, Any]:
        with sqlite3.connect(self.db_path) as db:
            count = int(db.execute("SELECT COUNT(*) FROM semantic_retrieval_cache").fetchone()[0])
            hits = int(db.execute("SELECT COALESCE(SUM(hits),0) FROM semantic_retrieval_cache").fetchone()[0])
        return {
            "schema_version": self.SCHEMA_VERSION,
            "entries": count,
            "max_entries": self.max_entries,
            "ttl_seconds": self.ttl_seconds,
            "similarity_threshold": self.similarity_threshold,
            "expected_dimension": self.expected_dimension,
            "recorded_hits": hits,
            "semantic": True,
        }

    @staticmethod
    def restore(payload: Sequence[dict[str, Any]]) -> list[RetrievalHit]:
        restored: list[RetrievalHit] = []
        for item in payload or []:
            try:
                restored.append(
                    RetrievalHit(
                        str(item.get("doc_id", "unknown")),
                        str(item.get("text", "")),
                        dict(item.get("metadata") or {}),
                        float(item.get("score", 0.0)),
                        float(item.get("vector_score", 0.0)),
                        float(item.get("lexical_score", 0.0)),
                    )
                )
            except (TypeError, ValueError):
                continue
        return restored


def install(system_module: Any) -> None:
    """Bind the canonical retriever to the production semantic cache implementation."""
    from rag_project.intelligence import med_evidence_pro

    med_evidence_pro.SemanticCache = SemanticRetrievalCache
