from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from rag_project.configuration.config_i5_16gb import DEVICE_PRESETS, get_preset
from rag_project.security import validate_storage_path


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
    ocr_enabled: bool = True
    auto_ocr: bool = True
    universal_pdf_mode: bool = True
    ocr_confidence_threshold: float = 0.55
    ocr_min_char_density: float = 0.001
    ocr_image_coverage_threshold: float = 0.55
    lazy_model_loading: bool = True
    retrieval_candidate_multiplier: int = 5
    evidence_min_confidence: float = 0.25
    answer_verification_enabled: bool = True
    numeric_verification_enabled: bool = True
    contradiction_detection_enabled: bool = True
    query_decomposition_enabled: bool = True
    multi_query_retrieval_enabled: bool = True
    max_query_variants: int = 8

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

    def __post_init__(self) -> None:
        self.device_mode = self.device_mode if self.device_mode in DEVICE_PRESETS else "i5_16gb"
        self.project_root = Path(self.project_root).expanduser().resolve()
        self.ollama_base_url = str(self.ollama_base_url).strip().rstrip("/") or "http://127.0.0.1:11434"
        self.chunk_size = max(200, int(self.chunk_size))
        self.chunk_overlap = max(0, min(int(self.chunk_overlap), self.chunk_size - 1))
        self.top_k = max(1, min(int(self.top_k), 50))
        self.temperature = max(0.0, min(float(self.temperature), 1.0))
        self.vector_weight = max(0.0, min(float(self.vector_weight), 1.0))
        self.page_batch_size = max(1, int(self.page_batch_size))
        self.chunk_batch_size = max(1, int(self.chunk_batch_size))
        self.embedding_batch_size = max(1, int(self.embedding_batch_size))
        self.embedding_retries = max(0, int(self.embedding_retries))
        self.embedding_timeout_seconds = max(1.0, float(self.embedding_timeout_seconds))
        self.generation_timeout_seconds = max(1.0, float(self.generation_timeout_seconds))
        self.generation_latency_budget_seconds = max(1.0, float(self.generation_latency_budget_seconds))
        self.generation_max_output_tokens = max(32, int(self.generation_max_output_tokens))
        self.context_token_budget = max(256, int(self.context_token_budget))
        self.max_workers = max(1, int(self.max_workers))
        self.max_memory_target = max(256, int(self.max_memory_target))
        self.ollama_concurrency = max(1, int(self.ollama_concurrency))
        self.embedding_cache_size = max(0, int(self.embedding_cache_size))
        self.embedding_cache_ttl_seconds = max(0.0, float(self.embedding_cache_ttl_seconds))
        self.ingestion_lease_seconds = max(30, int(self.ingestion_lease_seconds))
        self.ollama_failure_circuit_threshold = max(1, int(self.ollama_failure_circuit_threshold))
        self.ollama_circuit_open_seconds = max(1.0, float(self.ollama_circuit_open_seconds))
        self.ocr_confidence_threshold = max(0.0, min(float(self.ocr_confidence_threshold), 1.0))
        self.ocr_min_char_density = max(0.0, float(self.ocr_min_char_density))
        self.ocr_image_coverage_threshold = max(0.0, min(float(self.ocr_image_coverage_threshold), 1.0))
        self.retrieval_candidate_multiplier = max(2, min(12, int(self.retrieval_candidate_multiplier)))
        self.evidence_min_confidence = max(0.0, min(1.0, float(self.evidence_min_confidence)))
        self.max_query_variants = max(1, min(12, int(self.max_query_variants)))

    def apply_device_preset(self, device_mode: str) -> None:
        normalized = device_mode if device_mode in DEVICE_PRESETS else "i5_16gb"
        preset = get_preset(normalized)
        for key, value in preset.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.device_mode = normalized
        self.__post_init__()

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
        self.__post_init__()
        return warnings

    @classmethod
    def from_env(cls) -> "Settings":
        for dotenv_candidate in (Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env", Path.home() / ".bookrag.env"):
            if dotenv_candidate.is_file():
                load_dotenv(dotenv_candidate, override=False)
                break
        else:
            project_root_env = os.getenv("PROJECT_ROOT")
            if project_root_env:
                candidate = Path(project_root_env) / ".env"
                if candidate.is_file():
                    load_dotenv(candidate, override=False)

        default_root = Path(__file__).resolve().parents[2]
        project_root = Path(os.getenv("PROJECT_ROOT", str(default_root))).expanduser().resolve()
        additional = project_root / ".env"
        if additional.is_file():
            load_dotenv(additional, override=False)

        device_mode = os.getenv("DEVICE_MODE", "i5_16gb")
        if device_mode not in DEVICE_PRESETS:
            device_mode = "i5_16gb"
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
            generation_timeout_seconds=cls._parse_float(os.getenv("GENERATION_TIMEOUT_SECONDS"), float(preset["generation_timeout_seconds"])),
            generation_latency_budget_seconds=cls._parse_float(os.getenv("GENERATION_LATENCY_BUDGET_SECONDS"), float(preset["generation_latency_budget_seconds"])),
            generation_max_output_tokens=cls._parse_int(os.getenv("GENERATION_MAX_OUTPUT_TOKENS"), int(preset["generation_max_output_tokens"])),
            chunk_size=cls._parse_int(os.getenv("CHUNK_SIZE"), int(preset["chunk_size"])),
            chunk_overlap=cls._parse_int(os.getenv("CHUNK_OVERLAP"), int(preset["chunk_overlap"])),
            top_k=cls._parse_int(os.getenv("TOP_K"), int(preset["top_k"])),
            temperature=cls._parse_float(os.getenv("TEMPERATURE"), 0.2),
            ingestion_db_path=project_root / os.getenv("INGESTION_DB_PATH", "data/ingestion.sqlite3"),
            page_batch_size=cls._parse_int(os.getenv("PAGE_BATCH_SIZE"), 16),
            chunk_batch_size=cls._parse_int(os.getenv("CHUNK_BATCH_SIZE"), 32),
            embedding_batch_size=cls._parse_int(os.getenv("EMBEDDING_BATCH_SIZE"), int(preset["embedding_batch_size"])),
            embedding_retries=cls._parse_int(os.getenv("EMBEDDING_RETRIES"), int(preset["embedding_retries"])),
            embedding_timeout_seconds=cls._parse_float(os.getenv("EMBEDDING_TIMEOUT_SECONDS"), float(preset["embedding_timeout_seconds"])),
            embedding_test_mode=cls._parse_bool(os.getenv("EMBEDDING_TEST_MODE"), False),
            lexical_mode=os.getenv("LEXICAL_MODE", "hybrid"),
            vector_weight=cls._parse_float(os.getenv("VECTOR_WEIGHT"), 0.7),
            context_token_budget=cls._parse_int(os.getenv("CONTEXT_TOKEN_BUDGET"), int(preset["context_token_budget"])),
            neighbor_expansion=cls._parse_bool(os.getenv("NEIGHBOR_EXPANSION"), True),
            contextual_retrieval=cls._parse_bool(os.getenv("CONTEXTUAL_RETRIEVAL"), bool(preset.get("contextual_retrieval", True))),
            max_workers=cls._parse_int(os.getenv("MAX_WORKERS"), int(preset["max_workers"])),
            max_memory_target=cls._parse_int(os.getenv("MAX_MEMORY_TARGET"), int(preset["max_memory_target"])),
            embedding_cache_size=cls._parse_int(os.getenv("EMBEDDING_CACHE_SIZE"), int(preset["embedding_cache_size"])),
            embedding_cache_ttl_seconds=cls._parse_float(os.getenv("EMBEDDING_CACHE_TTL_SECONDS"), float(preset["embedding_cache_ttl_seconds"])),
            ingestion_lease_seconds=cls._parse_int(os.getenv("INGESTION_LEASE_SECONDS"), int(preset["ingestion_lease_seconds"])),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            ollama_concurrency=cls._parse_int(os.getenv("OLLAMA_CONCURRENCY"), int(preset["ollama_concurrency"])),
            reranker_model=os.getenv("RERANKER_MODEL", preset["reranker_model"]),
            ollama_failure_circuit_threshold=cls._parse_int(os.getenv("OLLAMA_FAILURE_CIRCUIT_THRESHOLD"), int(preset["ollama_failure_circuit_threshold"])),
            ollama_circuit_open_seconds=cls._parse_float(os.getenv("OLLAMA_CIRCUIT_OPEN_SECONDS"), float(preset["ollama_circuit_open_seconds"])),
            ocr_enabled=cls._parse_bool(os.getenv("OCR_ENABLED"), True),
            auto_ocr=cls._parse_bool(os.getenv("AUTO_OCR"), True),
            universal_pdf_mode=cls._parse_bool(os.getenv("UNIVERSAL_PDF_MODE"), True),
            ocr_confidence_threshold=cls._parse_float(os.getenv("OCR_CONFIDENCE_THRESHOLD"), 0.55),
            ocr_min_char_density=cls._parse_float(os.getenv("OCR_MIN_CHAR_DENSITY"), 0.001),
            ocr_image_coverage_threshold=cls._parse_float(os.getenv("OCR_IMAGE_COVERAGE_THRESHOLD"), 0.55),
            lazy_model_loading=cls._parse_bool(os.getenv("LAZY_MODEL_LOADING"), True),
            retrieval_candidate_multiplier=cls._parse_int(os.getenv("RETRIEVAL_CANDIDATE_MULTIPLIER"), 5),
            evidence_min_confidence=cls._parse_float(os.getenv("EVIDENCE_MIN_CONFIDENCE"), 0.25),
            answer_verification_enabled=cls._parse_bool(os.getenv("ANSWER_VERIFICATION_ENABLED"), True),
            numeric_verification_enabled=cls._parse_bool(os.getenv("NUMERIC_VERIFICATION_ENABLED"), True),
            contradiction_detection_enabled=cls._parse_bool(os.getenv("CONTRADICTION_DETECTION_ENABLED"), True),
            query_decomposition_enabled=cls._parse_bool(os.getenv("QUERY_DECOMPOSITION_ENABLED"), True),
            multi_query_retrieval_enabled=cls._parse_bool(os.getenv("MULTI_QUERY_RETRIEVAL_ENABLED"), True),
            max_query_variants=cls._parse_int(os.getenv("MAX_QUERY_VARIANTS"), 8),
        )

        runtime_paths = {
            "incoming_dir": settings.incoming_dir,
            "processed_dir": settings.processed_dir,
            "failed_dir": settings.failed_dir,
            "archive_dir": settings.archive_dir,
            "vector_db_dir": settings.vector_db_dir,
            "log_dir": settings.log_dir,
            "ingestion_db_path": settings.ingestion_db_path,
        }
        for label, path in runtime_paths.items():
            validate_storage_path(settings.project_root, path, label)

        for _dir in [settings.incoming_dir, settings.processed_dir, settings.failed_dir, settings.archive_dir, settings.vector_db_dir, settings.log_dir]:
            _dir.mkdir(parents=True, exist_ok=True)
        valid_levels = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
        if settings.log_level not in valid_levels:
            settings.log_level = "INFO"
        return settings


def create_default_settings() -> Settings:
    return Settings.from_env()
