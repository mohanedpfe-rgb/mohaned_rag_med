# BookRAG security baseline

## Application controls implemented

The production composition root installs security guards before a RAG system is returned. The controls include mandatory administrator authentication, session expiry and lockout, explicit destructive-action confirmation, project-root filesystem jailing, strict Ollama host allowlisting with DNS/IP checks, redirect rejection, PDF signature and structural validation, upload and page limits, query limits, global ingestion/answer concurrency limits, per-session query rate limiting, prompt/evidence sanitization, multilingual high-risk medical abstention, log/control-character sanitization, audit events, restrictive POSIX permissions, and CI dependency auditing.

## Required deployment controls

BookRAG must remain bound to loopback unless a separately authenticated reverse proxy with TLS is deployed. Do not publish the Streamlit port directly to the Internet or an untrusted LAN.

BookRAG currently uses Chroma's embedded `PersistentClient` on the local filesystem; it does not instantiate Chroma's network `HttpClient`/server API. This materially reduces exposure to Chroma server-side tenant/authorization vulnerabilities, but it does **not** make the vulnerable Chroma package itself risk-free. As of September 2026, upstream advisories CVE-2026-45830, CVE-2026-45831 and CVE-2026-45833 affect the currently released ChromaDB line, with no patched package available in the advisory data. The project therefore treats those advisories as an explicit accepted upstream dependency risk, keeps the Chroma store local-only, does not enable `trust_remote_code`, and requires dependency-audit review before any network-facing deployment.

Medical source material, Chroma/HNSW data, SQLite files, and application logs are persisted on the configured filesystem. For confidential medical data, the host volume must use operating-system or infrastructure-level encryption at rest and access controls. Application code cannot honestly guarantee protection against an attacker who can copy the entire live vector-store directory and its encryption material; that boundary belongs to the storage platform.

Store `BOOKRAG_ADMIN_PASSWORD` outside the repository with filesystem/process-environment protections. Configure a unique strong value of at least 12 characters and rotate it when operators change. There is no built-in/default password.

For remote Ollama, use an exact hostname in `BOOKRAG_OLLAMA_ALLOWLIST`; do not use a wildcard, IP range, or an HTTP endpoint exposed to untrusted networks. HTTPS is recommended for any remote endpoint.

## Verification

CI runs Python compilation, the complete pytest suite on Python 3.11 and 3.12, and `pip-audit` against the generated lockfile. The Chroma advisories are currently ignored only because upstream has not published a fixed release; CI still fails on any other vulnerability. This is a documented compensating-control exception, not a claim that the dependency is vulnerability-free.

## Residual architectural boundary

RAG prompt injection can be reduced but never mathematically eliminated while untrusted source text is presented to an LLM. The application therefore treats retrieved text as data, sanitizes instruction-like payloads, requires grounding/citations for high-risk medical queries, and applies a deterministic post-generation medical safety boundary. Human verification remains required for clinical decisions.
