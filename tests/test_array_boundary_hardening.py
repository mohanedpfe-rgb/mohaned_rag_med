from pathlib import Path

import numpy as np

from rag_project.retrieval.hybrid_retriever import HybridRetriever
from rag_project.runtime_stability_v6 import _as_list, _safe_search, _safe_vector, _safe_validate_document_index
from rag_project.storage.vector_store import VectorStore


class _FakeCollection:
    def __init__(self, vectors=None):
        self.vectors = np.asarray(vectors if vectors is not None else [])
        self.metadata = {"dimension": int(self.vectors.shape[1]) if self.vectors.ndim == 2 and self.vectors.shape[0] else 3}

    def get(self, **kwargs):
        if "ids" in kwargs:
            ids = list(kwargs["ids"])
            return {"ids": ids, "embeddings": self.vectors[: len(ids)]}
        return {
            "ids": [f"id-{i}" for i in range(len(self.vectors))],
            "metadatas": [{"document_id": "doc", "chunk_id": f"chunk-{i}", "version_id": "v1", "index_state": "READY"} for i in range(len(self.vectors))],
        }


def test_vector_validator_handles_numpy_matrix_without_ambiguous_truth_value():
    vector_rows = _as_list(np.array([[1.0, 0.0, 0.5]]))
    assert isinstance(vector_rows, list)
    assert vector_rows[0].tolist() == [1.0, 0.0, 0.5]
    assert _safe_vector(vector_rows[0]) == [1.0, 0.0, 0.5]


def test_document_index_validation_handles_numpy_chroma_embeddings():
    fake = type("FakeVectorStore", (), {})()
    fake.collection = _FakeCollection([[1.0, 0.0, 0.5], [0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])
    fake._coerce_metadata = VectorStore._coerce_metadata.__get__(fake)
    fake._collection_dim = lambda: 3
    fake._valid_vector = VectorStore._valid_vector

    report = _safe_validate_document_index(fake, "doc", "v1")

    assert report["valid"] is True
    assert report["count"] == 3
    assert not report["issues"]


def test_vector_search_normalizes_numpy_query_before_legacy_boolean_checks():
    called = {}
    fake = type("FakeVectorStore", (), {})()

    def original(_self, embedding, n_results=5, where=None):
        called["embedding"] = embedding
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    fake._runtime_v6_original_search = original.__get__(fake)
    result = _safe_search(fake, np.array([1.0, 2.0, 3.0]))

    assert result["ids"] == [[]]
    assert called["embedding"] == [1.0, 2.0, 3.0]
    assert type(called["embedding"]) is list


def test_hybrid_unpack_results_accepts_numpy_arrays_in_backend_payloads():
    payload = {
        "ids": np.array(["a", "b"]),
        "documents": np.array(["alpha", "beta"]),
        "metadatas": np.array([{"document_id": "d1"}, {"document_id": "d2"}], dtype=object),
        "distances": np.array([0.1, 0.2]),
    }

    ids, documents, metadatas, distances = HybridRetriever._unpack_results(payload)

    assert ids == ["a", "b"]
    assert documents == ["alpha", "beta"]
    assert [item["document_id"] for item in metadatas] == ["d1", "d2"]
    assert distances == [0.1, 0.2]


def test_active_vector_paths_have_no_common_array_truthiness_antipatterns():
    root = Path(__file__).resolve().parents[1] / "rag_project"
    active_sources = (
        root / "storage" / "vector_store.py",
        root / "retrieval" / "hybrid_retriever.py",
        root / "runtime_stability_v6.py",
    )
    forbidden = (
        '.get("embeddings") or []',
        '.get("embedding") or []',
        'if embeddings:',
        'if not embeddings:',
        'if embedding:',
        'if not embedding:',
    )
    offenders = []
    for path in active_sources:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                offenders.append(f"{path}: {token}")
    assert not offenders, "Array-truthiness antipatterns remain: " + "; ".join(offenders)
