import numpy as np

from rag_project.runtime_stability_v7 import _normalize_embedding_result


def test_embedding_result_is_normalized_to_plain_list():
    client = type("FakeClient", (), {})()
    client._runtime_v7_original_embed_texts = lambda _texts: np.array([[0.1, 0.2], [0.3, 0.4]])
    result = _normalize_embedding_result(client, ["a", "b"])
    assert type(result) is list
    assert len(result) == 2
    assert all(type(vector) is np.ndarray for vector in result)


def test_embedding_result_none_becomes_empty_list():
    client = type("FakeClient", (), {})()
    client._runtime_v7_original_embed_texts = lambda _texts: None
    assert _normalize_embedding_result(client, ["a"]) == []
