# BookRAG security baseline

## Application controls implemented

The production composition root installs security guards before a RAG system is returned. The controls include explicit destructive-action confirmation, project-root filesystem jailing, strict Ollama host allowlisting with DNS/IP checks, redirect rejection, PDF signature and structural validation, upload and page limits, query limits, global ingestion/answer concurrency limits, per-session query rate limiting, prompt/evidence sanitization, multilingual high-risk medical abstention, log/control-character sanitization, audit events, restrictive POSIX permissions, and CI dependency auditing.

Application-level administrator authentication is **not** treated as a security boundary in the current production Streamlit path. A password must never be hardcoded into the repository or documented as a permanent built-in credential. Deployments that require user authentication must place BookRAG behind an authenticated reverse proxy or another independently managed access-control layer.

## Required deployment controls

BookRAG must remain bound to loopback unless a separately authenticated reverse proxy with TLS is deployed. Do not publish the Streamlit port directly to the Internet or an untrusted LAN.

BookRAG currently uses Chroma's embedded `PersistentClient` on the local filesystem; it does not instantiate Chroma's network `HttpClient`/server API. This reduces exposure to Chroma server-side tenant/authorization vulnerabilities, but it does **not** make the Chroma package risk-free. As of September 2026, the repository's CI intentionally tracks several Chroma advisories as accepted upstream dependency risk; those exceptions must be re-evaluated whenever a patched release becomes available.

Medical source material, Chroma/HNSW data, SQLite files, and application logs are persisted on the configured filesystem. For confidential medical data, the host volume must use operating-system or infrastructure-level encryption at rest and access controls. Application code cannot honestly guarantee protection against an attacker who can copy the entire live vector-store directory and its encryption material; that boundary belongs to the storage platform.

For remote Ollama, use an exact hostname in `BOOKRAG_OLLAMA_ALLOWLIST`; do not use a wildcard, IP range, or an HTTP endpoint exposed to untrusted networks. HTTPS is recommended for any remote endpoint.

## Verification

CI runs Python compilation, the complete pytest suite on Python 3.11 and 3.12, and `pip-audit` against the generated lockfile. Any dependency exception must remain explicit, documented, and temporary rather than being treated as proof that the dependency is vulnerability-free.

## Residual architectural boundary

RAG prompt injection can be reduced but never mathematically eliminated while untrusted source text is presented to an LLM. The application therefore treats retrieved text as data, sanitizes instruction-like payloads, requires grounding/citations for high-risk medical queries, and applies a deterministic post-generation medical safety boundary. Human verification remains required for clinical decisions.
