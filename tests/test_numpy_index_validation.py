from __future__ import annotations

import numpy as np

from rag_project.runtime_stability_v6 import _safe_validate_document_index


class _Collection:
    def __init__(self):
        self.calls = 0

    def get(self, **kwargs):
        self.calls += 1
        if "metadatas" in kwargs.get("include", []):
            return {
                "ids": ["doc-1-chunk-0", "doc-1-chunk-1"],
                "metadatas": [
                    {"document_id": "doc-1", "chunk_id": "doc-1-chunk-0", "version_id": "v1", "index_state": "BUILDING"},
                    {"document_id": "doc-1", "chunk_id": "doc-1-chunk-1", "version_id": "v1", "index_state": "BUILDING"},
                ],
            }
        return {"ids": kwargs["ids"], "embeddings": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)}


class _Store:
    def __init__(self):
        self.collection = _Collection()

    def _coerce_metadata(self, value):
        return dict(value or {})

    def _collection_dim(self):
        return 2

    @staticmethod
    def _valid_vector(vector, expected_dimension=0):
        values = [float(value) for value in vector]
        return bool(values) and len(values) == expected_dimension and all(np.isfinite(values))


def test_validation_handles_numpy_embedding_array_without_ambiguous_truth_value():
    result = _safe_validate_document_index(_Store(), "doc-1", "v1", sample_size=2)
    assert result["valid"] is True
    assert result["sampled_embeddings"] == 2
    assert result["issues"] == []
