#!/usr/bin/env python3
"""Benchmark single-series self-prediction for explicit lengths."""

from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

DEVICE = "cuda"
DTYPE = "float32"
METHOD = "simplex"
MEMORY_BUDGET_GB = 4.0
XTWX_PRECOMPUTE = True
XTWY_PRECOMPUTE = True
EMBED_DIM = 20
TAU = 1
TP = 1
EXCLUSION_WINDOW = 10
LIBRARY_SIZE: int | str | None = None
SAMPLE_SIZE: int | str | None = None
BATCH_SIZE: int | str | None = "auto"
LENGTHS = [2_000, 8_000, 32_000, 128_000]
ATTEMPTS = 3
SEED = 1234
TORCH_NUM_THREADS = int(
    os.environ.get(
        "FASTCCM_TORCH_NUM_THREADS",
        os.environ.get("TORCH_NUM_THREADS", min(os.cpu_count() or 1, 8)),
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

from fastccm import PairwiseCCM  # noqa: E402
from fastccm.utils import get_td_embedding_np  # noqa: E402
def resolve_size(size: int | str | None, available_points: int, auto_divisor: int) -> int:
    if size is None:
        return available_points
    if size == "auto":
        return min(max(available_points // auto_divisor, 1), available_points)
    return min(int(size), available_points)


def resolve_case_parameters(length: int) -> tuple[int, int, int, int]:
    embedded_length = length - (EMBED_DIM - 1) * TAU
    valid_points = embedded_length - TP
    if valid_points <= 0:
        raise ValueError(
            f"length={length} is too short for EMBED_DIM={EMBED_DIM}, TAU={TAU}, TP={TP}."
        )

    library_size = resolve_size(LIBRARY_SIZE, valid_points, auto_divisor=2)
    sample_size = resolve_size(SAMPLE_SIZE, valid_points, auto_divisor=6)

    if METHOD != "simplex":
        return embedded_length, library_size, sample_size, EXCLUSION_WINDOW

    required_neighbors = EMBED_DIM + 1
    if library_size <= required_neighbors:
        raise ValueError(
            "Invalid benchmark case: library_size must exceed the simplex neighbor "
            f"count. Got library_size={library_size}, required>{required_neighbors}."
        )

    max_exclusion_window = max((library_size - required_neighbors - 1) // 2, 0)
    exclusion_window = min(EXCLUSION_WINDOW, max_exclusion_window)
    return embedded_length, library_size, sample_size, exclusion_window


def run_one(
    ccm: PairwiseCCM,
    length: int,
    attempts: int,
    base_seed: int,
) -> dict[str, float | int]:
    embedded_length, library_size, sample_size, exclusion_window = resolve_case_parameters(
        length
    )
    timings: list[float] = []

    for attempt in range(attempts):
        rng = np.random.default_rng(base_seed + attempt)
        series = rng.standard_normal(length, dtype=np.float32)
        x_emb = get_td_embedding_np(series[:, None], dim=EMBED_DIM, stride=TAU)[
            :, :, 0
        ].astype(np.float32, copy=False)
        y_emb = series[:, None].astype(np.float32, copy=False)

        t0 = time.perf_counter()
        _ = ccm.score_matrix(
            X_emb=[x_emb],
            Y_emb=[y_emb],
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
        "length": length,
        "embedded_length": embedded_length,
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
        verbose=2,
    )

    print("Benchmark settings:")
    print("  scenario=single-series self-prediction")
    print(f"  device={DEVICE}")
    print(f"  dtype={DTYPE}")
    print(f"  method={METHOD}")
    print(f"  E={EMBED_DIM}")
    print(f"  tau={TAU}")
    print(f"  tp={TP}")
    print(f"  exclusion_window={EXCLUSION_WINDOW}")
    print(f"  library_size={LIBRARY_SIZE if LIBRARY_SIZE is not None else 'all valid points'}")
    print(f"  sample_size={SAMPLE_SIZE if SAMPLE_SIZE is not None else 'all valid points'}")
    print(f"  batch_size={BATCH_SIZE}")
    print(f"  memory_budget_gb={MEMORY_BUDGET_GB}")
    print(f"  xtwx_precompute={XTWX_PRECOMPUTE}")
    print(f"  xtwy_precompute={XTWY_PRECOMPUTE}")
    print(f"  attempts={ATTEMPTS}")
    print(f"  lengths={LENGTHS}")
    print(f"  torch_num_threads={TORCH_NUM_THREADS}")
    print(f"  torch_num_interop_threads={TORCH_NUM_INTEROP_THREADS}")
    print()
    print("length,embedded_length,library_size,sample_size,exclusion_window,attempts,avg_sec,min_sec,max_sec")

    for idx, length in enumerate(LENGTHS, start=1):
        result = run_one(
            ccm=ccm,
            length=length,
            attempts=ATTEMPTS,
            base_seed=SEED + idx * 10_000,
        )
        print(
            f"{result['length']},{result['embedded_length']},{result['library_size']},"
            f"{result['sample_size']},{result['exclusion_window']},{result['attempts']},"
            f"{result['avg_sec']:.6f},{result['min_sec']:.6f},{result['max_sec']:.6f}"
        )


if __name__ == "__main__":
    main()
