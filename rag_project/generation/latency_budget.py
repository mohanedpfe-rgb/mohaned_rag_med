from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import time
from typing import Iterator

_DEFAULT_HARD_CAP_SECONDS = 45.0
_MIN_REQUEST_BUDGET_SECONDS = 1.0

_deadline: ContextVar[float | None] = ContextVar("rag_generation_deadline", default=None)
_started: ContextVar[float | None] = ContextVar("rag_generation_started", default=None)
_budget: ContextVar[float | None] = ContextVar("rag_generation_budget", default=None)


def configured_budget(settings: object | None = None) -> float:
    """Return the effective per-answer budget, capped to a predictable ceiling."""
    configured = getattr(settings, "generation_latency_budget_seconds", _DEFAULT_HARD_CAP_SECONDS)
    try:
        value = float(configured)
    except (TypeError, ValueError):
        value = _DEFAULT_HARD_CAP_SECONDS
    return max(_MIN_REQUEST_BUDGET_SECONDS, min(value, _DEFAULT_HARD_CAP_SECONDS))


def has_budget() -> bool:
    deadline = _deadline.get()
    return deadline is None or time.monotonic() < deadline


def remaining(default_seconds: float | None = None) -> float | None:
    deadline = _deadline.get()
    if deadline is None:
        return default_seconds
    return max(0.0, deadline - time.monotonic())


def elapsed() -> float:
    started = _started.get()
    return 0.0 if started is None else max(0.0, time.monotonic() - started)


def budget_seconds() -> float | None:
    return _budget.get()


def exhausted() -> bool:
    deadline = _deadline.get()
    return deadline is not None and time.monotonic() >= deadline


@contextmanager
def request_budget(settings: object | None = None) -> Iterator[float]:
    """Install one deadline shared by every generation call made during an answer."""
    seconds = configured_budget(settings)
    now = time.monotonic()
    deadline_token = _deadline.set(now + seconds)
    started_token = _started.set(now)
    budget_token = _budget.set(seconds)
    try:
        yield seconds
    finally:
        _budget.reset(budget_token)
        _started.reset(started_token)
        _deadline.reset(deadline_token)


@contextmanager
def budget_scope(seconds: float) -> Iterator[float]:
    """Create a latency deadline, preserving an existing outer answer deadline.

    Nested scopes are observationally shared with the outer scope rather than
    resetting the deadline. This prevents a helper operation from extending the
    total answer budget.
    """
    value = max(0.0, float(seconds))
    outer_deadline = _deadline.get()
    if outer_deadline is not None:
        yield max(0.0, outer_deadline - time.monotonic())
        return

    now = time.monotonic()
    deadline_token = _deadline.set(now + value)
    started_token = _started.set(now)
    budget_token = _budget.set(value)
    try:
        yield value
    finally:
        _budget.reset(budget_token)
        _started.reset(started_token)
        _deadline.reset(deadline_token)
