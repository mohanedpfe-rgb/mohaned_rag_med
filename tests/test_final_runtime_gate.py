from __future__ import annotations

from datetime import datetime, timedelta, timezone

from rag_project.runtime_final_gate import _lease_matches, _model_digest


def test_lease_matches_requires_exact_owner_and_live_expiry():
    class Store:
        pass

    import rag_project.runtime_final_gate as gate

    gate._LEASE_CONTEXT.state_store = Store()
    gate._LEASE_CONTEXT.document_id = "doc"
    gate._LEASE_CONTEXT.worker_id = "worker-a"
    try:
        live = {
            "lease_owner": "worker-a",
            "lease_expires_at": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
        }
        assert _lease_matches(live, "doc")
        assert not _lease_matches({**live, "lease_owner": "worker-b"}, "doc")
        assert not _lease_matches(
            {**live, "lease_expires_at": (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()},
            "doc",
        )
    finally:
        for name in ("state_store", "document_id", "worker_id"):
            if hasattr(gate._LEASE_CONTEXT, name):
                delattr(gate._LEASE_CONTEXT, name)


def test_model_digest_uses_ollama_tag_digest(monkeypatch):
    import rag_project.runtime_final_gate as gate

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"models": [{"name": "nomic-embed-text:latest", "digest": "sha256:abc"}]}

    monkeypatch.setattr(gate.requests, "get", lambda *args, **kwargs: Response())

    class Service:
        base_url = "http://ollama"
        model = "nomic-embed-text:latest"
        test_mode = False

    service = Service()
    assert _model_digest(service) == "sha256:abc"
