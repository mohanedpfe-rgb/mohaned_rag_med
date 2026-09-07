from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class Settings:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    incoming_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "data" / "incoming")
    processed_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "data" / "processed")
    failed_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "data" / "failed")
    archive_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "data" / "archive")
    vector_db_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "data" / "vector_db")
    log_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "logs")
    ollama_base_url: str = "http://127.0.0.1:11434"
    embedding_model: str = "qwen3-embedding:latest"
    generation_model: str = "llama3.2:3b"
    generation_timeout_seconds: float = 180.0
    generation_latency_budget_seconds: float = 60.0
    generation_max_output_tokens: int = 512
    chunk_size: int = 700
    chunk_overlap: int = 120
    contextual_retrieval: bool = True
    top_k: int = 6
    temperature: float = 0.2
    ingestion_db_path: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "data" / "ingestion.sqlite3")
    page_batch_size: int = 16
    chunk_batch_size: int = 32
    embedding_batch_size: int = 4
    embedding_retries: int = 4
    embedding_timeout_seconds: float = 180.0
    embedding_test_mode: bool = False
    lexical_mode: str = "hybrid"
    vector_weight: float = 0.7
    context_token_budget: int = 5000
    neighbor_expansion: bool = True
    max_workers: int = 4
    max_memory_target: int = 1024
    embedding_cache_size: int = 128
    embedding_cache_ttl_seconds: float = 900.0
    ingestion_lease_seconds: int = 900
    log_level: str = "INFO"
    ollama_concurrency: int = 2
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    ollama_failure_circuit_threshold: int = 5
    ollama_circuit_open_seconds: float = 30.0

    @staticmethod
    def _parse_bool(value: str | bool | None, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

    @classmethod
    def from_env(cls) -> "Settings":
        env_loaded = False
        for dotenv_candidate in (
            Path.cwd() / ".env",
            Path(__file__).resolve().parents[2] / ".env",
            Path.home() / ".bookrag.env",
        ):
            if dotenv_candidate.is_file():
                load_dotenv(dotenv_candidate, override=False)
                env_loaded = True
                break
        if not env_loaded:
            project_root_env = os.getenv("PROJECT_ROOT")
            if project_root_env:
                candidate = Path(project_root_env) / ".env"
                if candidate.is_file():
                    load_dotenv(candidate, override=False)
        project_root = Path(os.getenv("PROJECT_ROOT", ".")).resolve()
        if not env_loaded:
            additional = project_root / ".env"
            if additional.is_file():
                load_dotenv(additional, override=False)
        settings = cls(
            project_root=project_root,
            incoming_dir=project_root / os.getenv("INCOMING_DIR", "data/incoming"),
            processed_dir=project_root / os.getenv("PROCESSED_DIR", "data/processed"),
            failed_dir=project_root / os.getenv("FAILED_DIR", "data/failed"),
            archive_dir=project_root / os.getenv("ARCHIVE_DIR", "data/archive"),
            vector_db_dir=project_root / os.getenv("VECTOR_DB_DIR", "data/vector_db"),
            log_dir=project_root / os.getenv("LOG_DIR", "logs"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "qwen3-embedding:latest"),
            generation_model=os.getenv("GENERATION_MODEL", "llama3.2:3b"),
            generation_timeout_seconds=float(os.getenv("GENERATION_TIMEOUT_SECONDS", "180")),
            generation_latency_budget_seconds=float(
                os.getenv("GENERATION_LATENCY_BUDGET_SECONDS", "60")
            ),
            generation_max_output_tokens=int(os.getenv("GENERATION_MAX_OUTPUT_TOKENS", "512")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "700")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
            top_k=int(os.getenv("TOP_K", "6")),
            temperature=float(os.getenv("TEMPERATURE", "0.2")),
            ingestion_db_path=project_root / os.getenv("INGESTION_DB_PATH", "data/ingestion.sqlite3"),
            page_batch_size=int(os.getenv("PAGE_BATCH_SIZE", "16")),
            chunk_batch_size=int(os.getenv("CHUNK_BATCH_SIZE", "32")),
            embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
            embedding_retries=int(os.getenv("EMBEDDING_RETRIES", "3")),
            embedding_timeout_seconds=float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "300")),
            embedding_test_mode=cls._parse_bool(os.getenv("EMBEDDING_TEST_MODE", "false"), False),
            lexical_mode=os.getenv("LEXICAL_MODE", "hybrid"),
            vector_weight=float(os.getenv("VECTOR_WEIGHT", "0.7")),
            context_token_budget=int(os.getenv("CONTEXT_TOKEN_BUDGET", "5000")),
            neighbor_expansion=cls._parse_bool(os.getenv("NEIGHBOR_EXPANSION", "true"), True),
            max_workers=int(os.getenv("MAX_WORKERS", "4")),
            max_memory_target=int(os.getenv("MAX_MEMORY_TARGET", "1024")),
            embedding_cache_size=int(os.getenv("EMBEDDING_CACHE_SIZE", "128")),
            embedding_cache_ttl_seconds=float(os.getenv("EMBEDDING_CACHE_TTL_SECONDS", "900")),
            ingestion_lease_seconds=int(os.getenv("INGESTION_LEASE_SECONDS", "900")),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            ollama_concurrency=int(os.getenv("OLLAMA_CONCURRENCY", "2")),
            reranker_model=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
            ollama_failure_circuit_threshold=int(os.getenv("OLLAMA_FAILURE_CIRCUIT_THRESHOLD", "5")),
            ollama_circuit_open_seconds=float(os.getenv("OLLAMA_CIRCUIT_OPEN_SECONDS", "30.0")),
        )
        # Ensure required data directories exist
        for _dir in [settings.incoming_dir, settings.processed_dir, settings.failed_dir, settings.archive_dir, settings.vector_db_dir, settings.log_dir]:
            _dir.mkdir(parents=True, exist_ok=True)
        valid_levels = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
        if settings.log_level not in valid_levels:
            settings.log_level = "INFO"
        return settings


def create_default_settings() -> Settings:
    return Settings.from_env()
