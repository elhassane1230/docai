"""Latency & throughput benchmarking used by the BERT-vs-DistilBERT ablation.

Reports p50/p95/p99 latency and throughput for any callable that maps a string
to a prediction — works uniformly across the HF pipeline, the ONNX wrapper, and
the TF-IDF baseline, so all conditions are measured identically.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass
from typing import Callable


@dataclass
class LatencyReport:
    name: str
    n: int
    warmup: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    throughput_qps: float

    def as_dict(self) -> dict:
        return asdict(self)


def benchmark(fn: Callable[[str], object], samples: list[str], *,
              name: str = "model", warmup: int = 3) -> LatencyReport:
    # Warm up (JIT, caches, lazy weight loads) before timing.
    for i in range(min(warmup, len(samples))):
        fn(samples[i])

    latencies: list[float] = []
    t_start = time.perf_counter()
    for s in samples:
        t0 = time.perf_counter()
        fn(s)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    wall = time.perf_counter() - t_start

    latencies.sort()

    def pct(p: float) -> float:
        k = min(len(latencies) - 1, int(round(p / 100 * (len(latencies) - 1))))
        return latencies[k]

    return LatencyReport(
        name=name,
        n=len(samples),
        warmup=warmup,
        p50_ms=round(statistics.median(latencies), 3),
        p95_ms=round(pct(95), 3),
        p99_ms=round(pct(99), 3),
        mean_ms=round(statistics.mean(latencies), 3),
        throughput_qps=round(len(samples) / wall, 2) if wall > 0 else 0.0,
    )


def compare(reports: list[LatencyReport], baseline: str) -> dict:
    """Compute latency reduction of each report vs. the named baseline."""
    base = next(r for r in reports if r.name == baseline)
    out = {}
    for r in reports:
        out[r.name] = {
            **r.as_dict(),
            "latency_reduction_vs_baseline_pct": round(
                100 * (base.p50_ms - r.p50_ms) / base.p50_ms, 1
            ) if base.p50_ms else 0.0,
        }
    return out
