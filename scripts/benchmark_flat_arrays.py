#!/usr/bin/env python3
"""Benchmark FastCCM performance for highly asymmetric source/target matrices."""

from __future__ import annotations

import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from benchmark_report import update_report_section

DEVICE = "cpu"
DTYPE = "float32"
METHOD = "simplex"
MEMORY_BUDGET_GB = 4.0
XTWX_PRECOMPUTE = True
XTWY_PRECOMPUTE = False
TP = 0
X_EMBEDDING_DIM = 5
Y_EMBEDDING_DIM = 1
EXCLUSION_WINDOW = 5
LIBRARY_SIZE: int | str | None = None
SAMPLE_SIZE: int | str | None = None
BATCH_SIZE: int | str | None = "auto"
ATTEMPTS = 3
SEED = 12345

BENCHMARK_CASES: list[tuple[int, int, int]] = [
    (1000, 1, 1000),
    (5000, 1, 1000),
    (20000, 1, 1000),
    (20000, 1, 2000),
    (1, 1000, 1000),
    (1, 5000, 1000),
    (1, 20000, 1000),
    (1, 20000, 2000),
]
TORCH_NUM_THREADS = int(
    os.environ.get(
        "FASTCCM_TORCH_NUM_THREADS",
        os.environ.get("TORCH_NUM_THREADS", min(os.cpu_count() or 1, 10)),
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
    timings: list[float] = []

    for attempt in range(attempts):
        rng = np.random.default_rng(base_seed + attempt)
        x_emb, y_emb = generate_random_embeddings(
            rng=rng,
            n_x=case.n_x,
            n_y=case.n_y,
            ts_length=case.ts_length,
            ex=case.ex,
        )

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
        timings.append(time.perf_counter() - t0)

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
        "avg_sec": statistics.fmean(timings),
        "min_sec": min(timings),
        "max_sec": max(timings),
    }


def main() -> None:
    ccm = PairwiseCCM(
        device=DEVICE,
        dtype=DTYPE,
        memory_budget_gb=MEMORY_BUDGET_GB,
        verbose=0,
    )
    cases = build_cases()
    settings = {
        "scenario": "flat_arrays",
        "device": DEVICE,
        "dtype": DTYPE,
        "method": METHOD,
        "tp": TP,
        "exclusion_window": EXCLUSION_WINDOW,
        "library_size": LIBRARY_SIZE if LIBRARY_SIZE is not None else "all points",
        "sample_size": SAMPLE_SIZE if SAMPLE_SIZE is not None else "all points",
        "batch_size": BATCH_SIZE,
        "memory_budget_gb": MEMORY_BUDGET_GB,
        "xtwx_precompute": XTWX_PRECOMPUTE,
        "xtwy_precompute": XTWY_PRECOMPUTE,
        "attempts": ATTEMPTS,
        "benchmark_cases": BENCHMARK_CASES,
        "x_embedding_dim": X_EMBEDDING_DIM,
        "torch_num_threads": TORCH_NUM_THREADS,
        "torch_num_interop_threads": TORCH_NUM_INTEROP_THREADS,
    }
    columns = [
        "n_x",
        "n_y",
        "ts_length",
        "ex",
        "ey",
        "library_size",
        "sample_size",
        "exclusion_window",
        "attempts",
        "avg_sec",
        "min_sec",
        "max_sec",
    ]
    markdown_rows: list[list[str]] = []

    print("Benchmark settings:")
    for key, value in settings.items():
        print(f"  {key}={value}")
    print()
    print(
        "n_x,n_y,ts_length,ex,ey,library_size,sample_size,exclusion_window,"
        "attempts,avg_sec,min_sec,max_sec"
    )

    for idx, case in enumerate(cases, start=1):
        result = run_case(
            ccm=ccm,
            case=case,
            attempts=ATTEMPTS,
            base_seed=SEED + idx * 1000,
        )
        print(
            f"{result['n_x']},{result['n_y']},{result['ts_length']},{result['ex']},"
            f"{result['ey']},{result['library_size']},{result['sample_size']},"
            f"{result['exclusion_window']},{result['attempts']},{result['avg_sec']:.6f},"
            f"{result['min_sec']:.6f},{result['max_sec']:.6f}"
        )
        markdown_rows.append(
            [
                str(result["n_x"]),
                str(result["n_y"]),
                str(result["ts_length"]),
                str(result["ex"]),
                str(result["ey"]),
                str(result["library_size"]),
                str(result["sample_size"]),
                str(result["exclusion_window"]),
                str(result["attempts"]),
                f"{result['avg_sec']:.6f}",
                f"{result['min_sec']:.6f}",
                f"{result['max_sec']:.6f}",
            ]
        )

    update_report_section(
        section_id="flat-arrays",
        title="Flat Arrays",
        script_name=Path(__file__).name,
        settings=settings,
        columns=columns,
        rows=markdown_rows,
    )


if __name__ == "__main__":
    main()
