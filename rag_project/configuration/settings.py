from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from rag_project.configuration.config_i5_16gb import DEVICE_PRESETS, get_preset


@dataclass
class Settings:
    device_mode: str = "i5_16gb"
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    incoming_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "data" / "incoming")
    processed_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "data" / "processed")
    failed_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "data" / "failed")
    archive_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "data" / "archive")
    vector_db_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "data" / "vector_db")
    log_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "logs")
    ollama_base_url: str = "http://127.0.0.1:11434"
    embedding_model: str = "nomic-embed-text"
    generation_model: str = "llama3.2:3b"
    generation_timeout_seconds: float = 180.0
    generation_latency_budget_seconds: float = 60.0
    generation_max_output_tokens: int = 512
    chunk_size: int = 600
    chunk_overlap: int = 100
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
    context_token_budget: int = 3200
    neighbor_expansion: bool = True
    max_workers: int = 2
    max_memory_target: int = 1024
    embedding_cache_size: int = 128
    embedding_cache_ttl_seconds: float = 900.0
    ingestion_lease_seconds: int = 900
    log_level: str = "INFO"
    ollama_concurrency: int = 1
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ollama_failure_circuit_threshold: int = 5
    ollama_circuit_open_seconds: float = 30.0
    ocr_enabled: bool = False
    ocr_confidence_threshold: float = 0.55
    ocr_min_char_density: float = 0.001
    ocr_image_coverage_threshold: float = 0.55
    lazy_model_loading: bool = True

    @staticmethod
    def _parse_bool(value: str | bool | None, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _parse_float(value: str | float | None, default: float) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        if value is None:
            return default
        try:
            return float(str(value).strip())
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _parse_int(value: str | int | None, default: int) -> int:
        if isinstance(value, int):
            return value
        if value is None:
            return default
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return default

    def apply_device_preset(self, device_mode: str) -> None:
        preset = get_preset(device_mode)
        for key, value in preset.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.device_mode = device_mode

    def override_from_dict(self, overrides: dict[str, Any]) -> list[str]:
        warnings: list[str] = []
        for key, value in (overrides or {}).items():
            if not hasattr(self, key):
                warnings.append(f"Unknown setting {key!r}; skipped.")
                continue
            setattr(self, key, value)
        if "device_mode" in overrides and overrides["device_mode"] in DEVICE_PRESETS:
            preset = get_preset(overrides["device_mode"])
            for key, value in preset.items():
                if key not in overrides and hasattr(self, key):
                    setattr(self, key, value)
        return warnings

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
        default_root = Path(__file__).resolve().parents[2]
        project_root = Path(os.getenv("PROJECT_ROOT", str(default_root))).resolve()
        if not env_loaded:
            additional = project_root / ".env"
            if additional.is_file():
                load_dotenv(additional, override=False)

        device_mode = os.getenv("DEVICE_MODE", "i5_16gb")
        preset = get_preset(device_mode)
        settings = cls(
            device_mode=device_mode,
            project_root=project_root,
            incoming_dir=project_root / os.getenv("INCOMING_DIR", "data/incoming"),
            processed_dir=project_root / os.getenv("PROCESSED_DIR", "data/processed"),
            failed_dir=project_root / os.getenv("FAILED_DIR", "data/failed"),
            archive_dir=project_root / os.getenv("ARCHIVE_DIR", "data/archive"),
            vector_db_dir=project_root / os.getenv("VECTOR_DB_DIR", "data/vector_db"),
            log_dir=project_root / os.getenv("LOG_DIR", "logs"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            embedding_model=os.getenv("EMBEDDING_MODEL", preset["embedding_model"]),
            generation_model=os.getenv("GENERATION_MODEL", preset["generation_model"]),
            generation_timeout_seconds=float(os.getenv("GENERATION_TIMEOUT_SECONDS", str(preset["generation_timeout_seconds"]))),
            generation_latency_budget_seconds=float(
                os.getenv("GENERATION_LATENCY_BUDGET_SECONDS", str(preset["generation_latency_budget_seconds"]))
            ),
            generation_max_output_tokens=int(os.getenv("GENERATION_MAX_OUTPUT_TOKENS", str(preset["generation_max_output_tokens"]))),
            chunk_size=int(os.getenv("CHUNK_SIZE", str(preset["chunk_size"]))),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", str(preset["chunk_overlap"]))),
            top_k=int(os.getenv("TOP_K", str(preset["top_k"]))),
            temperature=float(os.getenv("TEMPERATURE", "0.2")),
            ingestion_db_path=project_root / os.getenv("INGESTION_DB_PATH", "data/ingestion.sqlite3"),
            page_batch_size=int(os.getenv("PAGE_BATCH_SIZE", "16")),
            chunk_batch_size=int(os.getenv("CHUNK_BATCH_SIZE", "32")),
            embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", str(preset["embedding_batch_size"]))),
            embedding_retries=int(os.getenv("EMBEDDING_RETRIES", str(preset["embedding_retries"]))),
            embedding_timeout_seconds=float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", str(preset["embedding_timeout_seconds"]))),
            embedding_test_mode=cls._parse_bool(os.getenv("EMBEDDING_TEST_MODE", "false"), False),
            lexical_mode=os.getenv("LEXICAL_MODE", "hybrid"),
            vector_weight=float(os.getenv("VECTOR_WEIGHT", "0.7")),
            context_token_budget=int(os.getenv("CONTEXT_TOKEN_BUDGET", str(preset["context_token_budget"]))),
            neighbor_expansion=cls._parse_bool(os.getenv("NEIGHBOR_EXPANSION", "true"), True),
            contextual_retrieval=cls._parse_bool(
                os.getenv("CONTEXTUAL_RETRIEVAL", str(preset.get("contextual_retrieval", True))),
                True,
            ),
            max_workers=int(os.getenv("MAX_WORKERS", str(preset["max_workers"]))),
            max_memory_target=int(os.getenv("MAX_MEMORY_TARGET", str(preset["max_memory_target"]))),
            embedding_cache_size=int(os.getenv("EMBEDDING_CACHE_SIZE", str(preset["embedding_cache_size"]))),
            embedding_cache_ttl_seconds=float(os.getenv("EMBEDDING_CACHE_TTL_SECONDS", str(preset["embedding_cache_ttl_seconds"]))),
            ingestion_lease_seconds=int(os.getenv("INGESTION_LEASE_SECONDS", str(preset["ingestion_lease_seconds"]))),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            ollama_concurrency=int(os.getenv("OLLAMA_CONCURRENCY", str(preset["ollama_concurrency"]))),
            reranker_model=os.getenv("RERANKER_MODEL", preset["reranker_model"]),
            ollama_failure_circuit_threshold=int(os.getenv("OLLAMA_FAILURE_CIRCUIT_THRESHOLD", str(preset["ollama_failure_circuit_threshold"]))),
            ollama_circuit_open_seconds=float(os.getenv("OLLAMA_CIRCUIT_OPEN_SECONDS", str(preset["ollama_circuit_open_seconds"]))),
            ocr_enabled=cls._parse_bool(os.getenv("OCR_ENABLED", str(preset["ocr_enabled"])), preset["ocr_enabled"]),
            ocr_confidence_threshold=cls._parse_float(os.getenv("OCR_CONFIDENCE_THRESHOLD", None), 0.55),
            ocr_min_char_density=cls._parse_float(os.getenv("OCR_MIN_CHAR_DENSITY", None), 0.001),
            ocr_image_coverage_threshold=cls._parse_float(os.getenv("OCR_IMAGE_COVERAGE_THRESHOLD", None), 0.55),
            lazy_model_loading=cls._parse_bool(os.getenv("LAZY_MODEL_LOADING", "true"), True),
        )
        for _dir in [settings.incoming_dir, settings.processed_dir, settings.failed_dir, settings.archive_dir, settings.vector_db_dir, settings.log_dir]:
            _dir.mkdir(parents=True, exist_ok=True)
        valid_levels = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
        if settings.log_level not in valid_levels:
            settings.log_level = "INFO"
        return settings


def create_default_settings() -> Settings:
    return Settings.from_env()
