from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def rss(pid: int) -> int | None:
    status = Path(f"/proc/{pid}/status")
    if not status.exists():
        return None
    for line in status.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("VmRSS:"):
            try:
                return int(line.split()[1]) * 1024
            except (IndexError, ValueError):
                return None
    return None


def fd_count(pid: int) -> int | None:
    directory = Path(f"/proc/{pid}/fd")
    try:
        return len(list(directory.iterdir()))
    except OSError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=20.0)
    args = parser.parse_args()
    from rag_project.testing.advanced_phases import _cleanup_store, _embedding, _fixture_chunks, _store_fixture

    started = time.monotonic()
    deadline = started + max(5.0, args.duration)
    samples: list[int] = []
    fds: list[int] = []
    iterations = 0
    while time.monotonic() < deadline:
        chunks = _fixture_chunks()
        store, tmp = _store_fixture(chunks)
        try:
            store.search_lexical("diabetes diagnosis", n_results=3)
            store.search(_embedding("diabetes diagnosis"), n_results=3)
        finally:
            _cleanup_store(tmp)
        current_rss = rss(os.getpid())
        current_fd = fd_count(os.getpid())
        if current_rss is not None:
            samples.append(current_rss)
        if current_fd is not None:
            fds.append(current_fd)
        iterations += 1
    observed = time.monotonic() - started
    payload = {
        "workload": "SemanticChunker + VectorStore add/search loop",
        "observed_seconds": observed,
        "sample_count": len(samples),
        "iterations": iterations,
        "rss_first_bytes": samples[0] if samples else None,
        "rss_last_bytes": samples[-1] if samples else None,
        "rss_peak_bytes": max(samples) if samples else None,
        "rss_delta_bytes": (samples[-1] - samples[0]) if len(samples) >= 2 else None,
        "fd_first": fds[0] if fds else None,
        "fd_last": fds[-1] if fds else None,
        "fd_delta": (fds[-1] - fds[0]) if len(fds) >= 2 else None,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
