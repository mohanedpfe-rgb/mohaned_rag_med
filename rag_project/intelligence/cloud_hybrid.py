"""Optional Phase-6 cloud/enterprise controls for MedEvidence Pro.

Local-first by default. Cloud escalation is opt-in, consent-gated, rate-limited,
PII-redacted, audited and budget-aware. The provider interface is generic and
Anthropic-compatible without forcing a new dependency on the offline runtime.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping, Protocol
from urllib.request import Request, urlopen


class CloudProvider(Protocol):
    def generate(self, prompt: str, *, timeout: float = 20.0) -> tuple[str, float]: ...


@dataclass(frozen=True)
class CloudConfig:
    enabled: bool = False
    endpoint: str = ""
    api_key_env: str = "ANTHROPIC_API_KEY"
    model: str = "claude-3-5-sonnet-latest"
    timeout_seconds: float = 20.0
    max_requests_per_minute: int = 10
    max_cost_usd_per_day: float = 5.0
    price_input_per_million: float = 3.0
    price_output_per_million: float = 15.0
    cloud_threshold: float = 0.70
    require_consent: bool = True
    pii_redaction: bool = True

    @classmethod
    def from_env(cls) -> "CloudConfig":
        return cls(
            enabled=os.getenv("MEDEVIDENCE_CLOUD_ENABLED", "false").casefold() == "true",
            endpoint=os.getenv("MEDEVIDENCE_CLOUD_ENDPOINT", "https://api.anthropic.com/v1/messages"),
            api_key_env=os.getenv("MEDEVIDENCE_CLOUD_API_KEY_ENV", "ANTHROPIC_API_KEY"),
            model=os.getenv("MEDEVIDENCE_CLOUD_MODEL", "claude-3-5-sonnet-latest"),
            timeout_seconds=float(os.getenv("MEDEVIDENCE_CLOUD_TIMEOUT", "20")),
            max_requests_per_minute=int(os.getenv("MEDEVIDENCE_CLOUD_RPM", "10")),
            max_cost_usd_per_day=float(os.getenv("MEDEVIDENCE_CLOUD_DAILY_BUDGET", "5")),
            price_input_per_million=float(os.getenv("MEDEVIDENCE_CLOUD_INPUT_PRICE", "3")),
            price_output_per_million=float(os.getenv("MEDEVIDENCE_CLOUD_OUTPUT_PRICE", "15")),
            cloud_threshold=float(os.getenv("MEDEVIDENCE_CLOUD_THRESHOLD", "0.70")),
            require_consent=os.getenv("MEDEVIDENCE_CLOUD_REQUIRE_CONSENT", "true").casefold() == "true",
            pii_redaction=os.getenv("MEDEVIDENCE_CLOUD_PII_REDACTION", "true").casefold() == "true",
        )


class ConsentStore:
    def __init__(self, path: str | Path):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists(): self.path.write_text("{}", encoding="utf-8")
        self._lock = threading.Lock()

    def get(self, subject: str) -> bool:
        try: data=json.loads(self.path.read_text(encoding="utf-8"))
        except Exception: data={}
        return bool(data.get(subject, False))

    def set(self, subject: str, granted: bool) -> None:
        with self._lock:
            try: data=json.loads(self.path.read_text(encoding="utf-8"))
            except Exception: data={}
            data[str(subject)] = bool(granted)
            self.path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")


class PIIAnonymizer:
    PATTERNS = [
        (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[EMAIL]"),
        (re.compile(r"\b(?:\+?\d[\d .()-]{7,}\d)\b"), "[PHONE]"),
        (re.compile(r"\b(?:mr|mrs|ms|dr)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b"), "[NAME]"),
        (re.compile(r"\b\d{13,19}\b"), "[IDENTIFIER]"),
    ]

    @classmethod
    def redact(cls, text: str) -> str:
        value = str(text or "")
        for pattern, replacement in cls.PATTERNS:
            value = pattern.sub(replacement, value)
        return value


class RateLimiter:
    def __init__(self, requests_per_minute: int):
        self.capacity=max(1, int(requests_per_minute)); self.timestamps=[]; self._lock=threading.Lock()

    def allow(self) -> bool:
        now=time.time()
        with self._lock:
            self.timestamps=[t for t in self.timestamps if now-t < 60]
            if len(self.timestamps) >= self.capacity: return False
            self.timestamps.append(now); return True


class CostTracker:
    def __init__(self, daily_budget: float):
        self.daily_budget=max(0.0,float(daily_budget)); self._lock=threading.Lock(); self._by_day={}

    def _day(self) -> str: return time.strftime("%Y-%m-%d")

    def estimate(self, input_tokens: int, output_tokens: int, config: CloudConfig) -> float:
        return (max(0,input_tokens)/1_000_000)*config.price_input_per_million + (max(0,output_tokens)/1_000_000)*config.price_output_per_million

    def reserve(self, amount: float) -> bool:
        with self._lock:
            day=self._day(); used=float(self._by_day.get(day,0.0))
            if used+amount > self.daily_budget: return False
            self._by_day[day]=used+max(0.0,amount); return True

    def spent_today(self) -> float:
        with self._lock: return float(self._by_day.get(self._day(),0.0))


class AnthropicCompatibleProvider:
    def __init__(self, config: CloudConfig):
        self.config=config

    def generate(self, prompt: str, *, timeout: float | None = None) -> tuple[str, float]:
        key=os.getenv(self.config.api_key_env, "").strip()
        if not key: raise RuntimeError("cloud_api_key_missing")
        payload={"model":self.config.model,"max_tokens":500,"temperature":0.0,"messages":[{"role":"user","content":prompt}]}
        body=json.dumps(payload).encode("utf-8")
        request=Request(self.config.endpoint,data=body,method="POST",headers={"content-type":"application/json","x-api-key":key,"anthropic-version":"2023-06-01"})
        started=time.perf_counter()
        with urlopen(request,timeout=timeout or self.config.timeout_seconds) as response:
            data=json.loads(response.read().decode("utf-8"))
        text=data.get("content", [{}])[0].get("text", "")
        return str(text), (time.perf_counter()-started)*1000.0


class CloudAuditLog:
    def __init__(self, path: str | Path):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True); self._lock=threading.Lock()

    def record(self, *, subject: str, query: str, outcome: str, latency_ms: float=0.0, cost_usd: float=0.0, reason: str="") -> None:
        row={"timestamp":time.time(),"subject_hash":hashlib.sha256(subject.encode()).hexdigest()[:16],"query_hash":hashlib.sha256(query.encode()).hexdigest()[:16],"outcome":outcome,"latency_ms":latency_ms,"cost_usd":cost_usd,"reason":reason}
        with self._lock:
            with self.path.open("a",encoding="utf-8") as f: f.write(json.dumps(row,ensure_ascii=False)+"\n")


class HybridRouter:
    def __init__(self, config: CloudConfig, *, consent: ConsentStore | None = None, audit: CloudAuditLog | None = None):
        self.config=config; self.consent=consent; self.audit=audit

    def should_escalate(self, *, confidence: float, emergency: bool=False, user_requested: bool=False,
                        time_sensitive: bool=False, quality_sensitive: bool=False, subject: str="default") -> tuple[bool,str]:
        if not self.config.enabled: return False,"cloud_disabled"
        if self.config.require_consent and (self.consent is None or not self.consent.get(subject)):
            return False,"consent_required"
        if emergency or user_requested: return True,"explicit_or_emergency"
        if quality_sensitive: return True,"quality_sensitive"
        if time_sensitive: return False,"time_sensitive_local"
        if confidence < self.config.cloud_threshold: return True,"low_confidence"
        return False,"local_sufficient"

    def build_prompt(self, query: str, evidence: str, *, pii_redaction: bool | None=None) -> str:
        should_redact=self.config.pii_redaction if pii_redaction is None else bool(pii_redaction)
        q=PIIAnonymizer.redact(query) if should_redact else query
        e=PIIAnonymizer.redact(evidence) if should_redact else evidence
        return ("You are the optional cloud escalation stage of MedEvidence Pro. "
                "Use only the provided evidence, cite claims as [S#], do not diagnose or prescribe, "
                "and say when evidence is insufficient.\n\nQuestion: "+q[:3000]+"\n\nEvidence:\n"+e[:8000])

    def escalate(self, provider: CloudProvider, *, subject: str, query: str, evidence: str,
                 confidence: float, emergency: bool=False, user_requested: bool=False,
                 time_sensitive: bool=False, quality_sensitive: bool=False,
                 input_tokens: int=0, output_tokens: int=500) -> dict[str, Any]:
        allowed, reason=self.should_escalate(confidence=confidence, emergency=emergency,user_requested=user_requested,time_sensitive=time_sensitive,quality_sensitive=quality_sensitive,subject=subject)
        if not allowed:
            return {"escalated":False,"reason":reason,"answer":""}
        limiter=getattr(self,"rate_limiter",None)
        if limiter is not None and not limiter.allow(): return {"escalated":False,"reason":"rate_limited","answer":""}
        tracker=getattr(self,"cost_tracker",None)
        estimated=tracker.estimate(input_tokens,output_tokens,self.config) if tracker else 0.0
        if tracker and not tracker.reserve(estimated): return {"escalated":False,"reason":"daily_budget_exceeded","answer":""}
        try:
            answer,latency=provider.generate(self.build_prompt(query,evidence))
            if self.audit: self.audit.record(subject=subject,query=query,outcome="success",latency_ms=latency,cost_usd=estimated,reason=reason)
            return {"escalated":True,"reason":reason,"answer":answer,"latency_ms":latency,"cost_usd":estimated}
        except Exception as exc:
            if self.audit: self.audit.record(subject=subject,query=query,outcome="error",cost_usd=estimated,reason=type(exc).__name__)
            return {"escalated":False,"reason":"provider_error","answer":"","error":type(exc).__name__}


def create_hybrid_router(config: CloudConfig | None=None, root: str | Path | None=None) -> HybridRouter:
    cfg=config or CloudConfig.from_env(); base=Path(root or "data")
    router=HybridRouter(cfg, consent=ConsentStore(base/"cloud_consent.json"), audit=CloudAuditLog(base/"cloud_audit.jsonl"))
    router.rate_limiter=RateLimiter(cfg.max_requests_per_minute); router.cost_tracker=CostTracker(cfg.max_cost_usd_per_day)
    return router


@dataclass(frozen=True)
class EnterpriseRole:
    name: str
    permissions: tuple[str,...]

DEFAULT_ROLES={
    "admin": EnterpriseRole("admin",("view_queries","adjust_settings","export_data","verify_answers")),
    "clinician": EnterpriseRole("clinician",("verify_answers","view_high_stakes")),
    "researcher": EnterpriseRole("researcher",("export_data",)),
}


def has_permission(role: str, permission: str) -> bool:
    return permission in DEFAULT_ROLES.get(role, EnterpriseRole(role,())).permissions


__all__=["CloudConfig","ConsentStore","PIIAnonymizer","RateLimiter","CostTracker","AnthropicCompatibleProvider","CloudAuditLog","HybridRouter","create_hybrid_router","EnterpriseRole","DEFAULT_ROLES","has_permission"]
