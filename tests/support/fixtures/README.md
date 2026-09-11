# High-level fixtures

All fixtures are isolated per test and rooted under pytest's temporary directory. The i5/16GB profile uses deterministic embedding-test mode and lazy model loading; live Ollama access is opt-in with `@pytest.mark.requires_ollama`.

The `clean_system` fixture creates the canonical `rag_project.application.create_rag_system` runtime and ingests one controlled diabetes document before answer-path assertions execute.