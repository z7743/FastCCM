#!/usr/bin/env python3
"""Benchmark FastCCM CPU usage across richer nx, ny, T cases."""

from __future__ import annotations

import os
import resource
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

DEVICE = "cuda"
DTYPE = "float32"
METHOD = "simplex"
MEMORY_BUDGET_GB = 1.0
XTWX_PRECOMPUTE = True
XTWY_PRECOMPUTE = False
TP = 0
X_EMBEDDING_DIM = 5
Y_EMBEDDING_DIM = 1
EXCLUSION_WINDOW = 5
LIBRARY_SIZE: int | str | None = None
SAMPLE_SIZE: int | str | None = None
BATCH_SIZE: int | str | None = "auto"
ATTEMPTS = 1
SEED = 1234

BENCHMARK_CASES: list[tuple[int, int, int]] = [
    (50, 50, 500),
    (50, 50, 1000),
    (50, 50, 4000),
    (50, 50, 8000),
    (100, 100, 500),
    (100, 100, 1000),
    (100, 100, 4000),
    (100, 100, 8000),
    (200, 200, 500),
    (200, 200, 1000),
    (200, 200, 4000),
    (200, 200, 8000),
    (400, 400, 500),
    (400, 400, 1000),
    (400, 400, 4000),
    (400, 400, 8000),
    (800, 800, 500),
    (800, 800, 1000),
    (800, 800, 4000),
    (800, 800, 8000),
]
TORCH_NUM_THREADS = int(
    os.environ.get(
        "FASTCCM_TORCH_NUM_THREADS",
        os.environ.get("TORCH_NUM_THREADS", min(os.cpu_count() or 1, 20)),
    )
)
TORCH_NUM_INTEROP_THREADS = int(
    os.environ.get(
        "FASTCCM_TORCH_NUM_INTEROP_THREADS",
        os.environ.get("TORCH_NUM_INTEROP_THREADS", 1),
    )
)

torch.set_num_threads(TORCH_NUM_THREADS)
torch.set_num_interop_threads(TORCH_NUM_INTEROP_THREADS)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fastccm import PairwiseCCM


@dataclass(frozen=True)
class BenchmarkCase:
    n_x: int
    n_y: int
    ts_length: int
    ex: int


def build_cases() -> list[BenchmarkCase]:
    return [
        BenchmarkCase(
            n_x=n_x,
            n_y=n_y,
            ts_length=ts_length,
            ex=X_EMBEDDING_DIM,
        )
        for n_x, n_y, ts_length in BENCHMARK_CASES
    ]


def resolve_size(size: int | str | None, available_points: int, auto_divisor: int) -> int:
    if size is None:
        return available_points
    if size == "auto":
        return min(max(available_points // auto_divisor, 1), available_points)
    return min(int(size), available_points)


def resolve_exclusion_window(case: BenchmarkCase, library_size: int) -> int:
    if METHOD != "simplex":
        return EXCLUSION_WINDOW

    required_neighbors = case.ex + 1
    if library_size <= required_neighbors:
        raise ValueError(
            "Invalid benchmark case: library_size must exceed the simplex neighbor "
            f"count. Got library_size={library_size}, required>{required_neighbors} "
            f"for ex={case.ex}, ts_length={case.ts_length}."
        )

    max_exclusion_window = max((library_size - required_neighbors - 1) // 2, 0)
    return min(EXCLUSION_WINDOW, max_exclusion_window)


def generate_random_embeddings(
    rng: np.random.Generator,
    n_x: int,
    n_y: int,
    ts_length: int,
    ex: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    x_emb = [
        rng.standard_normal((ts_length, ex), dtype=np.float32)
        for _ in range(n_x)
    ]
    y_emb = [
        rng.standard_normal((ts_length, Y_EMBEDDING_DIM), dtype=np.float32)
        for _ in range(n_y)
    ]
    return x_emb, y_emb


def get_process_cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return float(usage.ru_utime + usage.ru_stime)


def run_case(
    ccm: PairwiseCCM,
    case: BenchmarkCase,
    attempts: int,
    base_seed: int,
) -> dict[str, float | int]:
    valid_points = case.ts_length - TP
    if valid_points <= 1:
        raise ValueError(
            f"ts_length={case.ts_length} leaves no valid query points with tp={TP}."
        )

    library_size = resolve_size(LIBRARY_SIZE, valid_points, auto_divisor=2)
    sample_size = resolve_size(SAMPLE_SIZE, valid_points, auto_divisor=6)
    exclusion_window = resolve_exclusion_window(case, library_size)
    wall_seconds: list[float] = []
    cpu_seconds: list[float] = []
    cpu_pct: list[float] = []

    for attempt in range(attempts):
        rng = np.random.default_rng(base_seed + attempt)
        x_emb, y_emb = generate_random_embeddings(
            rng=rng,
            n_x=case.n_x,
            n_y=case.n_y,
            ts_length=case.ts_length,
            ex=case.ex,
        )

        cpu_t0 = get_process_cpu_seconds()
        t0 = time.perf_counter()
        _ = ccm.score_matrix(
            X_emb=x_emb,
            Y_emb=y_emb,
            library_size=library_size,
            sample_size=sample_size,
            exclusion_window=exclusion_window,
            tp=TP,
            method=METHOD,
            xtwx_precompute=XTWX_PRECOMPUTE,
            xtwy_precompute=XTWY_PRECOMPUTE,
            batch_size=BATCH_SIZE,
            seed=base_seed + attempt,
            clean_after=False,
        )
        wall_sec = time.perf_counter() - t0
        cpu_sec = get_process_cpu_seconds() - cpu_t0
        wall_seconds.append(wall_sec)
        cpu_seconds.append(cpu_sec)
        cpu_pct.append(100.0 * cpu_sec / max(wall_sec, 1e-12))

    return {
        "n_x": case.n_x,
        "n_y": case.n_y,
        "ts_length": case.ts_length,
        "ex": case.ex,
        "ey": Y_EMBEDDING_DIM,
        "library_size": library_size,
        "sample_size": sample_size,
        "exclusion_window": exclusion_window,
        "attempts": attempts,
        "avg_sec": statistics.fmean(wall_seconds),
        "min_sec": min(wall_seconds),
        "max_sec": max(wall_seconds),
        "avg_cpu_sec": statistics.fmean(cpu_seconds),
        "min_cpu_sec": min(cpu_seconds),
        "max_cpu_sec": max(cpu_seconds),
        "avg_cpu_pct": statistics.fmean(cpu_pct),
        "min_cpu_pct": min(cpu_pct),
        "max_cpu_pct": max(cpu_pct),
        "avg_cpu_cores": statistics.fmean(cpu_pct) / 100.0,
    }


def main() -> None:
    ccm = PairwiseCCM(
        device=DEVICE,
        dtype=DTYPE,
        memory_budget_gb=MEMORY_BUDGET_GB,
        verbose=0,
    )
    cases = build_cases()

    print("Benchmark settings:")
    print("  scenario=cpu_usage")
    print(f"  device={DEVICE}")
    print(f"  dtype={DTYPE}")
    print(f"  method={METHOD}")
    print(f"  tp={TP}")
    print(f"  exclusion_window={EXCLUSION_WINDOW}")
    print(f"  library_size={LIBRARY_SIZE if LIBRARY_SIZE is not None else 'all points'}")
    print(f"  sample_size={SAMPLE_SIZE if SAMPLE_SIZE is not None else 'all points'}")
    print(f"  batch_size={BATCH_SIZE}")
    print(f"  memory_budget_gb={MEMORY_BUDGET_GB}")
    print(f"  xtwx_precompute={XTWX_PRECOMPUTE}")
    print(f"  xtwy_precompute={XTWY_PRECOMPUTE}")
    print(f"  attempts={ATTEMPTS}")
    print(f"  benchmark_cases={BENCHMARK_CASES}")
    print(f"  x_embedding_dim={X_EMBEDDING_DIM}")
    print(f"  torch_num_threads={TORCH_NUM_THREADS}")
    print(f"  torch_num_interop_threads={TORCH_NUM_INTEROP_THREADS}")
    print("  cpu_pct_note=process CPU usage, can exceed 100% when multiple CPU cores are busy")
    print()
    print(
        "n_x,n_y,ts_length,ex,ey,library_size,sample_size,exclusion_window,attempts,"
        "avg_sec,min_sec,max_sec,avg_cpu_sec,min_cpu_sec,max_cpu_sec,avg_cpu_pct,min_cpu_pct,max_cpu_pct,avg_cpu_cores"
    )

    for idx, case in enumerate(cases, start=1):
        result = run_case(
            ccm=ccm,
            case=case,
            attempts=ATTEMPTS,
            base_seed=SEED + idx * 1000,
        )
        print(
            f"{result['n_x']},{result['n_y']},{result['ts_length']},{result['ex']},{result['ey']},"
            f"{result['library_size']},{result['sample_size']},{result['exclusion_window']},"
            f"{result['attempts']},{result['avg_sec']:.6f},{result['min_sec']:.6f},{result['max_sec']:.6f},"
            f"{result['avg_cpu_sec']:.6f},{result['min_cpu_sec']:.6f},{result['max_cpu_sec']:.6f},"
            f"{result['avg_cpu_pct']:.2f},{result['min_cpu_pct']:.2f},{result['max_cpu_pct']:.2f},"
            f"{result['avg_cpu_cores']:.2f}"
        )


if __name__ == "__main__":
    main()