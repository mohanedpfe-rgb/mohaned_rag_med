from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEVICE_MODE = "i5_16gb"

I5_16GB_PRESET: dict[str, Any] = {
    "device_mode": "i5_16gb",
    "embedding_model": "nomic-embed-text",
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "generation_model": "llama3.2:3b",
    "embedding_batch_size": 16,
    "max_workers": 2,
    "ollama_concurrency": 1,
    "context_token_budget": 3200,
    "ocr_enabled": True,
    "chunk_size": 600,
    "chunk_overlap": 100,
    "generation_max_output_tokens": 512,
    "top_k": 6,
    "embedding_retries": 2,
    "embedding_timeout_seconds": 180.0,
    "embedding_cache_size": 128,
    "embedding_cache_ttl_seconds": 900.0,
    "ingestion_lease_seconds": 900,
    "generation_timeout_seconds": 180.0,
    "generation_latency_budget_seconds": 60.0,
    "ollama_failure_circuit_threshold": 5,
    "ollama_circuit_open_seconds": 30.0,
    "max_memory_target": 1024,
}

FULL_PRESET: dict[str, Any] = {
    "device_mode": "full",
    "embedding_model": "nomic-embed-text",
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "generation_model": "llama3.2:3b",
    "embedding_batch_size": 32,
    "max_workers": 4,
    "ollama_concurrency": 2,
    "context_token_budget": 5000,
    "ocr_enabled": True,
    "chunk_size": 700,
    "chunk_overlap": 120,
    "generation_max_output_tokens": 1024,
    "top_k": 8,
    "embedding_retries": 4,
    "embedding_timeout_seconds": 300.0,
    "embedding_cache_size": 10000,
    "embedding_cache_ttl_seconds": 86400.0,
    "ingestion_lease_seconds": 900,
    "generation_timeout_seconds": 300.0,
    "generation_latency_budget_seconds": 120.0,
    "ollama_failure_circuit_threshold": 5,
    "ollama_circuit_open_seconds": 30.0,
    "max_memory_target": 2048,
}

DEVICE_PRESETS: dict[str, dict[str, Any]] = {
    "i5_16gb": I5_16GB_PRESET,
    "full": FULL_PRESET,
}


@dataclass(frozen=True)
class DeviceProfile:
    name: str
    description: str
    ram_target_gb: tuple[float, float]
    typical_ingest_minutes_per_400_pages: tuple[int, int]

    def matches_mode(self, mode: str) -> bool:
        return mode == self.name


I5_16GB_PROFILE = DeviceProfile(
    name="i5_16gb",
    description="Optimized for Intel i5 / 16GB RAM laptops. Balances speed, memory, and stability.",
    ram_target_gb=(6.0, 10.0),
    typical_ingest_minutes_per_400_pages=(20, 30),
)

FULL_PROFILE = DeviceProfile(
    name="full",
    description="Full feature mode for machines with more CPU cores and RAM.",
    ram_target_gb=(10.0, 16.0),
    typical_ingest_minutes_per_400_pages=(10, 20),
)

DEVICE_PROFILES: dict[str, DeviceProfile] = {
    "i5_16gb": I5_16GB_PROFILE,
    "full": FULL_PROFILE,
}


def get_preset(device_mode: str) -> dict[str, Any]:
    if device_mode in DEVICE_PRESETS:
        return dict(DEVICE_PRESETS[device_mode])
    return dict(DEVICE_PRESETS["i5_16gb"])


def get_profile(device_mode: str) -> DeviceProfile:
    if device_mode in DEVICE_PROFILES:
        return DEVICE_PROFILES[device_mode]
    return I5_16GB_PROFILE
