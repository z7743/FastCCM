#!/usr/bin/env python3
"""Benchmark simplex target-batch policies for PairwiseCCM.

This script sweeps explicit `target_batch_size` policies on the simplex score
path while leaving the outer query `batch_size` on FastCCM's normal `"auto"`
setting. The goal is to identify which target-series chunking strategy gives
the best wall time and CPU utilization across representative `(n_x, n_y, T)`
cases before hard-coding a calibration policy in the runtime.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import resource
import statistics
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

DEFAULT_CASES: list[tuple[int, int, int]] = [
    (50, 50, 500),
    (50, 50, 4000),
    (100, 100, 1000),
    (100, 100, 8000),
    (200, 200, 1000),
    (200, 200, 4000),
    (400, 400, 1000),
    (400, 400, 4000),
]
DEFAULT_POLICIES = [
    "auto",
    "fixed:8",
    "fixed:16",
    "fixed:32",
    "fixed:64",
]
DEFAULT_IMBALANCE_FACTORS = [2, 4, 5]
DEFAULT_MAX_GENERATED_DIMENSION = 1600
DEFAULT_NUM_THREADS = int(
    os.environ.get(
        "FASTCCM_TORCH_NUM_THREADS",
        os.environ.get("TORCH_NUM_THREADS", min(os.cpu_count() or 1, 20)),
    )
)
DEFAULT_NUM_INTEROP_THREADS = int(
    os.environ.get(
        "FASTCCM_TORCH_NUM_INTEROP_THREADS",
        os.environ.get("TORCH_NUM_INTEROP_THREADS", 1),
    )
)

DEVICE = "cuda"
DTYPE = "float32"
METHOD = "simplex"
MEMORY_BUDGET_GB = 1.0
TP = 0
X_EMBEDDING_DIM = 5
Y_EMBEDDING_DIM = 1
EXCLUSION_WINDOW = 5
LIBRARY_SIZE: int | str | None = None
SAMPLE_SIZE: int | str | None = None
BATCH_SIZE: int | str | None = "auto"
SEED = 1234

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def load_fastccm_modules():
    package_root = SRC_DIR / "fastccm"
    package_name = "fastccm"
    utils_package_name = "fastccm.utils"
    ccm_module_name = "fastccm.ccm"
    runtime_module_name = "fastccm.utils.runtime"

    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_root)]
        sys.modules[package_name] = package
    if utils_package_name not in sys.modules:
        utils_package = types.ModuleType(utils_package_name)
        utils_package.__path__ = [str(package_root / "utils")]
        sys.modules[utils_package_name] = utils_package

    if runtime_module_name not in sys.modules:
        runtime_spec = importlib.util.spec_from_file_location(
            runtime_module_name,
            package_root / "utils" / "runtime.py",
        )
        if runtime_spec is None or runtime_spec.loader is None:
            raise ImportError(f"Could not load {runtime_module_name} from source tree.")
        runtime_module = importlib.util.module_from_spec(runtime_spec)
        sys.modules[runtime_module_name] = runtime_module
        runtime_spec.loader.exec_module(runtime_module)
    else:
        runtime_module = sys.modules[runtime_module_name]

    if ccm_module_name not in sys.modules:
        ccm_spec = importlib.util.spec_from_file_location(
            ccm_module_name,
            package_root / "ccm.py",
        )
        if ccm_spec is None or ccm_spec.loader is None:
            raise ImportError(f"Could not load {ccm_module_name} from source tree.")
        ccm_module = importlib.util.module_from_spec(ccm_spec)
        sys.modules[ccm_module_name] = ccm_module
        ccm_spec.loader.exec_module(ccm_module)
    else:
        ccm_module = sys.modules[ccm_module_name]

    return ccm_module.PairwiseCCM, runtime_module


PairwiseCCM, runtime_module = load_fastccm_modules()
_calibrated_simplex_target_batch_size = runtime_module._calibrated_simplex_target_batch_size
_dtype_bytes = runtime_module._dtype_bytes
_resolve_batch_size = runtime_module._resolve_batch_size
_simplex_base_bytes = runtime_module._simplex_base_bytes
_simplex_per_sample_bytes = runtime_module._simplex_per_sample_bytes


@dataclass(frozen=True)
class BenchmarkCase:
    n_x: int
    n_y: int
    ts_length: int
    ex: int = X_EMBEDDING_DIM
    ey: int = Y_EMBEDDING_DIM


@dataclass(frozen=True)
class CalibratedPolicy:
    tile_mb: float
    min_batch: int
    max_batch: int
    search_ratio: float
    search_batch: int


@dataclass(frozen=True)
class Policy:
    label: str
    kind: str
    target_batch_size: int | None | str
    calibrated: CalibratedPolicy | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark simplex target_batch_size policies for PairwiseCCM."
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=None,
        metavar="X,Y,T",
        help="Explicit benchmark cases such as 50,50,500 200,200,4000.",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=DEFAULT_POLICIES,
        metavar="POLICY",
        help=(
            "Policy specs. Supported forms: auto, none, fixed:<int>, "
            "calibrated:tile_mb=<f>,min=<int>,max=<int>,search_ratio=<f>,search_batch=<int>."
        ),
    )
    parser.add_argument(
        "--imbalanced-companions",
        action="store_true",
        help="Append imbalanced companions for symmetric cases while preserving n_x * n_y.",
    )
    parser.add_argument(
        "--imbalance-factors",
        type=int,
        nargs="+",
        default=DEFAULT_IMBALANCE_FACTORS,
        help="Generate (n/f, n*f) and (n*f, n/f) companions for symmetric (n,n) cases when divisible.",
    )
    parser.add_argument(
        "--max-generated-dimension",
        type=int,
        default=DEFAULT_MAX_GENERATED_DIMENSION,
        help="Skip generated imbalanced cases whose larger dimension exceeds this value. Use <=0 to disable.",
    )
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--dtype", default=DTYPE, choices=["float32", "float64"])
    parser.add_argument("--memory-budget-gb", type=float, default=MEMORY_BUDGET_GB)
    parser.add_argument("--num-threads", type=int, default=DEFAULT_NUM_THREADS)
    parser.add_argument("--num-interop-threads", type=int, default=DEFAULT_NUM_INTEROP_THREADS)
    parser.add_argument("--batch-size", default=BATCH_SIZE)
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output path.")
    return parser.parse_args()


def parse_case(spec: str) -> BenchmarkCase:
    parts = [part.strip() for part in spec.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Invalid case {spec!r}. Expected X,Y,T.")
    n_x, n_y, ts_length = (int(part) for part in parts)
    return BenchmarkCase(n_x=n_x, n_y=n_y, ts_length=ts_length)


def parse_policy(spec: str) -> Policy:
    lowered = spec.strip().lower()
    if lowered == "auto":
        return Policy(label="auto", kind="auto", target_batch_size="auto")
    if lowered == "none":
        return Policy(label="none", kind="none", target_batch_size=None)
    if lowered.startswith("fixed:"):
        value = int(lowered.split(":", 1)[1])
        if value <= 0:
            raise ValueError(f"Invalid fixed policy {spec!r}.")
        return Policy(label=f"fixed:{value}", kind="fixed", target_batch_size=value)
    if lowered.startswith("calibrated:"):
        raw_items = lowered.split(":", 1)[1].split(",")
        options: dict[str, str] = {}
        for item in raw_items:
            key, _, value = item.partition("=")
            if not key or not value:
                raise ValueError(f"Invalid calibrated policy option {item!r} in {spec!r}.")
            options[key.strip()] = value.strip()
        calibrated = CalibratedPolicy(
            tile_mb=float(options["tile_mb"]),
            min_batch=int(options["min"]),
            max_batch=int(options["max"]),
            search_ratio=float(options["search_ratio"]),
            search_batch=int(options["search_batch"]),
        )
        label = (
            "calibrated:"
            f"tile={calibrated.tile_mb:g}MB,"
            f"min={calibrated.min_batch},"
            f"max={calibrated.max_batch},"
            f"ratio={calibrated.search_ratio:g},"
            f"search={calibrated.search_batch}"
        )
        return Policy(
            label=label,
            kind="calibrated",
            target_batch_size="auto",
            calibrated=calibrated,
        )
    raise ValueError(f"Unsupported policy spec: {spec!r}")


def combine_cases(base_cases: list[BenchmarkCase], extra_cases: list[BenchmarkCase]) -> list[BenchmarkCase]:
    combined: dict[tuple[int, int, int, int, int], BenchmarkCase] = {}
    for case in [*base_cases, *extra_cases]:
        key = (case.n_x, case.n_y, case.ts_length, case.ex, case.ey)
        combined.setdefault(key, case)
    return list(combined.values())


def build_imbalanced_companion_cases(
    base_cases: list[BenchmarkCase],
    factors: list[int],
    max_generated_dimension: int,
) -> list[BenchmarkCase]:
    companions: list[BenchmarkCase] = []
    seen: set[tuple[int, int, int, int, int]] = {
        (case.n_x, case.n_y, case.ts_length, case.ex, case.ey)
        for case in base_cases
    }
    max_dim = int(max_generated_dimension)

    for case in base_cases:
        if case.n_x != case.n_y:
            continue
        n = int(case.n_x)
        for factor in factors:
            factor = int(factor)
            if factor <= 1:
                continue
            if n % factor != 0:
                continue
            small = n // factor
            large = n * factor
            if max_dim > 0 and max(small, large) > max_dim:
                continue
            for n_x, n_y in ((small, large), (large, small)):
                key = (n_x, n_y, case.ts_length, case.ex, case.ey)
                if key in seen:
                    continue
                seen.add(key)
                companions.append(
                    BenchmarkCase(
                        n_x=n_x,
                        n_y=n_y,
                        ts_length=case.ts_length,
                        ex=case.ex,
                        ey=case.ey,
                    )
                )
    return companions


def resolve_size(size: int | str | None, available_points: int, auto_divisor: int) -> int:
    if size is None:
        return available_points
    if size == "auto":
        return min(max(available_points // auto_divisor, 1), available_points)
    return min(int(size), available_points)


def resolve_exclusion_window(case: BenchmarkCase, library_size: int) -> int:
    required_neighbors = case.ex + 1
    if library_size <= required_neighbors:
        raise ValueError(
            "Invalid benchmark case: library_size must exceed the simplex neighbor "
            f"count. Got library_size={library_size}, required>{required_neighbors} "
            f"for ex={case.ex}, ts_length={case.ts_length}."
        )
    max_exclusion_window = max((library_size - required_neighbors - 1) // 2, 0)
    return min(EXCLUSION_WINDOW, max_exclusion_window)


def get_process_cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return float(usage.ru_utime + usage.ru_stime)


def _round_pow2_clamped(value: float, min_value: int, max_value: int) -> int:
    min_value = max(1, int(min_value))
    max_value = max(min_value, int(max_value))
    if value <= min_value:
        return min_value
    if value >= max_value:
        return max_value
    lo = min_value
    while lo * 2 <= value and lo * 2 <= max_value:
        lo *= 2
    hi = min(max_value, lo * 2)
    if hi == lo:
        return lo
    return lo if abs(value - lo) <= abs(hi - value) else hi


def resolve_target_batch_size(
    policy: Policy,
    *,
    case: BenchmarkCase,
    library_size: int,
    sample_size: int,
    budget_gb: float,
    dtype: torch.dtype,
    compute_dtype: torch.dtype,
) -> int | None | str:
    if policy.kind != "calibrated":
        return policy.target_batch_size

    assert policy.calibrated is not None
    calibrated = policy.calibrated
    n_x = int(case.n_x)
    n_y = int(case.n_y)
    ey = int(case.ey)
    k = int(case.ex + 1)
    budget_bytes = int(float(budget_gb) * (1024 ** 3) * 0.90)
    base_bytes = _simplex_base_bytes(
        num_ts_X=n_x,
        library_size=library_size,
        max_E_X=case.ex,
        total_samples=sample_size,
        num_ts_Y=n_y,
        max_E_Y=ey,
        dtype=dtype,
        compute_dtype=compute_dtype,
        extra_base_bytes=int(case.n_x * sample_size * case.n_y * case.ey * _dtype_bytes(dtype)),
    )
    cbytes = _dtype_bytes(compute_dtype)
    dbytes = _dtype_bytes(dtype)
    ibytes = 8
    search_per_sample = cbytes * (n_x * library_size + n_x * k) + (dbytes + ibytes) * (n_x * k)
    reduce_per_sample_full = cbytes * (n_x * k * n_y * ey + n_x * n_y * ey) + dbytes * (n_x * n_y * ey)
    dominance = float(search_per_sample) / float(max(reduce_per_sample_full, 1))
    if dominance >= calibrated.search_ratio:
        return min(n_y, int(calibrated.search_batch))
    available_bytes = int(budget_bytes) - int(base_bytes)
    if available_bytes <= 0:
        return min(n_y, int(calibrated.min_batch))
    sample_batch_search = max(
        1,
        min(sample_size, available_bytes // max(int(search_per_sample), 1)),
    )
    tile_unit_bytes = max(cbytes * n_x * k * ey * sample_batch_search, 1)
    target_by = float(calibrated.tile_mb * 1024.0 * 1024.0) / float(tile_unit_bytes)
    return _round_pow2_clamped(
        target_by,
        min_value=min(int(calibrated.min_batch), n_y),
        max_value=min(int(calibrated.max_batch), n_y),
    )


def resolve_effective_target_batch_size(
    policy: Policy,
    *,
    case: BenchmarkCase,
    library_size: int,
    sample_size: int,
    budget_gb: float,
    dtype: torch.dtype,
    compute_dtype: torch.dtype,
) -> int:
    if policy.kind == "auto":
        return int(
            _calibrated_simplex_target_batch_size(
                num_ts_X=case.n_x,
                num_ts_Y=case.n_y,
                total_samples=sample_size,
                library_size=library_size,
                max_EY=case.ey,
                nbrs_num_max=case.ex + 1,
                dtype=dtype,
                compute_dtype=compute_dtype,
                budget_bytes=int(float(budget_gb) * (1024 ** 3) * 0.90),
                max_E_X=case.ex,
                extra_base_bytes=int(case.n_x * sample_size * case.n_y * case.ey * _dtype_bytes(dtype)),
            )
        )
    resolved = resolve_target_batch_size(
        policy,
        case=case,
        library_size=library_size,
        sample_size=sample_size,
        budget_gb=budget_gb,
        dtype=dtype,
        compute_dtype=compute_dtype,
    )
    if resolved in (None, "auto"):
        return int(case.n_y)
    return min(int(case.n_y), int(resolved))


def estimate_selected_sample_batch_size(
    *,
    case: BenchmarkCase,
    library_size: int,
    sample_size: int,
    effective_target_batch_size: int,
    budget_gb: float,
    dtype: torch.dtype,
    compute_dtype: torch.dtype,
) -> int:
    n_x = int(case.n_x)
    ey = int(case.ey)
    k = int(case.ex + 1)
    budget_bytes = int(float(budget_gb) * (1024 ** 3) * 0.90)
    base_bytes = _simplex_base_bytes(
        num_ts_X=n_x,
        library_size=library_size,
        max_E_X=case.ex,
        total_samples=sample_size,
        num_ts_Y=case.n_y,
        max_E_Y=ey,
        dtype=dtype,
        compute_dtype=compute_dtype,
        extra_base_bytes=int(case.n_x * sample_size * case.n_y * case.ey * _dtype_bytes(dtype)),
    )
    cbytes = _dtype_bytes(compute_dtype)
    dbytes = _dtype_bytes(dtype)
    ibytes = 8
    search_per_sample = cbytes * (n_x * library_size + n_x * k) + (dbytes + ibytes) * (n_x * k)
    reduce_per_sample = (
        cbytes * (n_x * k * effective_target_batch_size * ey + n_x * effective_target_batch_size * ey)
        + dbytes * (n_x * effective_target_batch_size * ey)
    )
    per_sample_bytes = _simplex_per_sample_bytes(search_per_sample, reduce_per_sample)
    batch_size = _resolve_batch_size(sample_size, budget_bytes, base_bytes, per_sample_bytes)
    return int(batch_size)


def resolve_batch_size_arg(value: str | int | None) -> int | str | None:
    if value is None or isinstance(value, int):
        return value
    lowered = str(value).strip().lower()
    if lowered == "auto":
        return "auto"
    if lowered == "none":
        return None
    return int(lowered)


def generate_random_embeddings(
    rng: np.random.Generator,
    case: BenchmarkCase,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    x_emb = [
        rng.standard_normal((case.ts_length, case.ex), dtype=np.float32)
        for _ in range(case.n_x)
    ]
    y_emb = [
        rng.standard_normal((case.ts_length, case.ey), dtype=np.float32)
        for _ in range(case.n_y)
    ]
    return x_emb, y_emb


def benchmark_policy(
    ccm: PairwiseCCM,
    case: BenchmarkCase,
    policy: Policy,
    attempts: int,
    warmups: int,
    base_seed: int,
    dtype: torch.dtype,
    compute_dtype: torch.dtype,
    memory_budget_gb: float,
    batch_size: int | str | None,
) -> dict[str, float | int | str]:
    valid_points = case.ts_length - TP
    if valid_points <= 1:
        raise ValueError(f"ts_length={case.ts_length} leaves no valid query points with tp={TP}.")

    library_size = resolve_size(LIBRARY_SIZE, valid_points, auto_divisor=2)
    sample_size = resolve_size(SAMPLE_SIZE, valid_points, auto_divisor=6)
    exclusion_window = resolve_exclusion_window(case, library_size)
    target_batch_size = resolve_target_batch_size(
        policy,
        case=case,
        library_size=library_size,
        sample_size=sample_size,
        budget_gb=memory_budget_gb,
        dtype=dtype,
        compute_dtype=compute_dtype,
    )
    effective_target_batch_size = resolve_effective_target_batch_size(
        policy,
        case=case,
        library_size=library_size,
        sample_size=sample_size,
        budget_gb=memory_budget_gb,
        dtype=dtype,
        compute_dtype=compute_dtype,
    )
    selected_sample_batch_size = estimate_selected_sample_batch_size(
        case=case,
        library_size=library_size,
        sample_size=sample_size,
        effective_target_batch_size=effective_target_batch_size,
        budget_gb=memory_budget_gb,
        dtype=dtype,
        compute_dtype=compute_dtype,
    )

    wall_seconds: list[float] = []
    cpu_seconds: list[float] = []
    cpu_pct: list[float] = []

    total_runs = warmups + attempts
    for run_idx in range(total_runs):
        rng = np.random.default_rng(base_seed + run_idx)
        x_emb, y_emb = generate_random_embeddings(rng, case)

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
            batch_size=batch_size,
            target_batch_size=target_batch_size,
            seed=base_seed + run_idx,
            clean_after=False,
        )
        wall_sec = time.perf_counter() - t0
        cpu_sec = get_process_cpu_seconds() - cpu_t0
        if run_idx >= warmups:
            wall_seconds.append(wall_sec)
            cpu_seconds.append(cpu_sec)
            cpu_pct.append(100.0 * cpu_sec / max(wall_sec, 1e-12))

    avg_sec = statistics.fmean(wall_seconds)
    avg_cpu_pct = statistics.fmean(cpu_pct)
    return {
        "policy": policy.label,
        "policy_kind": policy.kind,
        "n_x": case.n_x,
        "n_y": case.n_y,
        "ts_length": case.ts_length,
        "ex": case.ex,
        "ey": case.ey,
        "library_size": library_size,
        "sample_size": sample_size,
        "exclusion_window": exclusion_window,
        "attempts": attempts,
        "warmups": warmups,
        "avg_sec": avg_sec,
        "min_sec": min(wall_seconds),
        "max_sec": max(wall_seconds),
        "avg_cpu_sec": statistics.fmean(cpu_seconds),
        "min_cpu_sec": min(cpu_seconds),
        "max_cpu_sec": max(cpu_seconds),
        "avg_cpu_pct": avg_cpu_pct,
        "min_cpu_pct": min(cpu_pct),
        "max_cpu_pct": max(cpu_pct),
        "avg_cpu_cores": avg_cpu_pct / 100.0,
        "resolved_target_batch_size": effective_target_batch_size,
        "selected_sample_batch_size_est": selected_sample_batch_size,
        "explicit_target_batch_size": "auto" if target_batch_size == "auto" else (
            "none" if target_batch_size is None else int(target_batch_size)
        ),
        "explicit_target_batch_size_est": effective_target_batch_size,
    }


def summarise_results(results: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    best_by_case: dict[tuple[int, int, int], float] = {}
    for row in results:
        case_key = (int(row["n_x"]), int(row["n_y"]), int(row["ts_length"]))
        current = float(row["avg_sec"])
        best = best_by_case.get(case_key)
        if best is None or current < best:
            best_by_case[case_key] = current

    by_policy: dict[str, list[dict[str, float | int | str]]] = {}
    for row in results:
        by_policy.setdefault(str(row["policy"]), []).append(row)

    summary: list[dict[str, float | int | str]] = []
    for policy_label, rows in by_policy.items():
        speedups = []
        wins = 0
        for row in rows:
            case_key = (int(row["n_x"]), int(row["n_y"]), int(row["ts_length"]))
            best = best_by_case[case_key]
            current = float(row["avg_sec"])
            speedups.append(current / best)
            if math.isclose(current, best, rel_tol=1e-9, abs_tol=1e-12):
                wins += 1
        summary.append(
            {
                "policy": policy_label,
                "cases": len(rows),
                "wins": wins,
                "mean_rel_time": statistics.fmean(speedups),
                "geom_rel_time": math.exp(statistics.fmean(math.log(max(v, 1e-12)) for v in speedups)),
                "avg_cpu_pct": statistics.fmean(float(row["avg_cpu_pct"]) for row in rows),
                "avg_cpu_cores": statistics.fmean(float(row["avg_cpu_cores"]) for row in rows),
            }
        )
    summary.sort(key=lambda row: (float(row["geom_rel_time"]), float(row["mean_rel_time"])))
    return summary


def print_case_winners(results: list[dict[str, float | int | str]]) -> None:
    grouped: dict[tuple[int, int, int], list[dict[str, float | int | str]]] = {}
    for row in results:
        grouped.setdefault(
            (int(row["n_x"]), int(row["n_y"]), int(row["ts_length"])),
            [],
        ).append(row)

    print()
    print("Per-case winners:")
    print("n_x,n_y,ts_length,best_policy,best_avg_sec,auto_avg_sec,auto_rel_time_vs_best")
    for case_key in sorted(grouped):
        case_rows = sorted(grouped[case_key], key=lambda row: float(row["avg_sec"]))
        best_row = case_rows[0]
        auto_row = next((row for row in case_rows if str(row["policy"]) == "auto"), None)
        auto_avg_sec = "n/a"
        auto_rel = "n/a"
        if auto_row is not None:
            auto_avg = float(auto_row["avg_sec"])
            auto_avg_sec = f"{auto_avg:.6f}"
            auto_rel = f"{(auto_avg / max(float(best_row['avg_sec']), 1e-12)):.4f}"
        print(
            f"{case_key[0]},{case_key[1]},{case_key[2]},{best_row['policy']},"
            f"{float(best_row['avg_sec']):.6f},{auto_avg_sec},{auto_rel}"
        )


def print_summary(summary: list[dict[str, float | int | str]]) -> None:
    print()
    print("Policy summary (lower rel_time is better):")
    print("policy,cases,wins,mean_rel_time,geom_rel_time,avg_cpu_pct,avg_cpu_cores")
    for row in summary:
        print(
            f"{row['policy']},{row['cases']},{row['wins']},"
            f"{float(row['mean_rel_time']):.4f},{float(row['geom_rel_time']):.4f},"
            f"{float(row['avg_cpu_pct']):.2f},{float(row['avg_cpu_cores']):.2f}"
        )


def write_csv(path: Path, results: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "policy",
                "policy_kind",
                "n_x",
                "n_y",
                "ts_length",
                "ex",
                "ey",
                "library_size",
                "sample_size",
                "exclusion_window",
                "attempts",
                "warmups",
                "avg_sec",
                "min_sec",
                "max_sec",
                "avg_cpu_sec",
                "min_cpu_sec",
                "max_cpu_sec",
                "avg_cpu_pct",
                "min_cpu_pct",
                "max_cpu_pct",
                "avg_cpu_cores",
                "resolved_target_batch_size",
                "selected_sample_batch_size_est",
                "explicit_target_batch_size",
                "explicit_target_batch_size_est",
            ],
        )
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    args = parse_args()

    base_cases = [parse_case(spec) for spec in args.cases] if args.cases else [
        BenchmarkCase(*case) for case in DEFAULT_CASES
    ]
    generated_cases: list[BenchmarkCase] = []
    if args.imbalanced_companions:
        generated_cases = build_imbalanced_companion_cases(
            base_cases,
            factors=args.imbalance_factors,
            max_generated_dimension=args.max_generated_dimension,
        )
    cases = combine_cases(base_cases, generated_cases)
    policies = [parse_policy(spec) for spec in args.policies]
    batch_size = resolve_batch_size_arg(args.batch_size)

    torch.set_num_threads(args.num_threads)
    torch.set_num_interop_threads(args.num_interop_threads)

    ccm = PairwiseCCM(
        device=args.device,
        dtype=args.dtype,
        memory_budget_gb=args.memory_budget_gb,
        verbose=0,
    )
    dtype = ccm.dtype
    compute_dtype = ccm.compute_dtype

    print("Benchmark settings:")
    print("  scenario=simplex target_batch_size policy sweep")
    print(f"  device={args.device}")
    print(f"  dtype={args.dtype}")
    print(f"  method={METHOD}")
    print(f"  tp={TP}")
    print(f"  exclusion_window={EXCLUSION_WINDOW}")
    print(f"  library_size={LIBRARY_SIZE if LIBRARY_SIZE is not None else 'all points'}")
    print(f"  sample_size={SAMPLE_SIZE if SAMPLE_SIZE is not None else 'all points'}")
    print(f"  batch_size={batch_size}")
    print(f"  memory_budget_gb={args.memory_budget_gb}")
    print(f"  imbalanced_companions={args.imbalanced_companions}")
    if args.imbalanced_companions:
        print(f"  imbalance_factors={args.imbalance_factors}")
        print(f"  max_generated_dimension={args.max_generated_dimension}")
        print(f"  generated_cases={len(generated_cases)}")
    print(f"  attempts={args.attempts}")
    print(f"  warmups={args.warmups}")
    print(f"  benchmark_cases={[(c.n_x, c.n_y, c.ts_length) for c in cases]}")
    print(f"  policies={[policy.label for policy in policies]}")
    print(f"  x_embedding_dim={X_EMBEDDING_DIM}")
    print(f"  y_embedding_dim={Y_EMBEDDING_DIM}")
    print(f"  torch_num_threads={args.num_threads}")
    print(f"  torch_num_interop_threads={args.num_interop_threads}")
    print("  cpu_pct_note=process CPU usage, can exceed 100% when multiple CPU cores are busy")
    print()
    print(
        "policy,policy_kind,n_x,n_y,ts_length,ex,ey,library_size,sample_size,exclusion_window,attempts,warmups,"
        "avg_sec,min_sec,max_sec,avg_cpu_sec,min_cpu_sec,max_cpu_sec,avg_cpu_pct,min_cpu_pct,max_cpu_pct,"
        "avg_cpu_cores,resolved_target_batch_size,selected_sample_batch_size_est,explicit_target_batch_size,"
        "explicit_target_batch_size_est"
    )

    results: list[dict[str, float | int | str]] = []
    for case_idx, case in enumerate(cases, start=1):
        for policy_idx, policy in enumerate(policies, start=1):
            result = benchmark_policy(
                ccm=ccm,
                case=case,
                policy=policy,
                attempts=args.attempts,
                warmups=args.warmups,
                base_seed=args.seed + case_idx * 100_000 + policy_idx * 1_000,
                dtype=dtype,
                compute_dtype=compute_dtype,
                memory_budget_gb=args.memory_budget_gb,
                batch_size=batch_size,
            )
            results.append(result)
            print(
                f"{result['policy']},{result['policy_kind']},{result['n_x']},{result['n_y']},{result['ts_length']},"
                f"{result['ex']},{result['ey']},{result['library_size']},{result['sample_size']},"
                f"{result['exclusion_window']},{result['attempts']},{result['warmups']},"
                f"{float(result['avg_sec']):.6f},{float(result['min_sec']):.6f},{float(result['max_sec']):.6f},"
                f"{float(result['avg_cpu_sec']):.6f},{float(result['min_cpu_sec']):.6f},{float(result['max_cpu_sec']):.6f},"
                f"{float(result['avg_cpu_pct']):.2f},{float(result['min_cpu_pct']):.2f},{float(result['max_cpu_pct']):.2f},"
                f"{float(result['avg_cpu_cores']):.2f},{result['resolved_target_batch_size']},"
                f"{result['selected_sample_batch_size_est']},{result['explicit_target_batch_size']},"
                f"{result['explicit_target_batch_size_est']}"
            )

    summary = summarise_results(results)
    print_case_winners(results)
    print_summary(summary)

    if args.csv is not None:
        write_csv(args.csv, results)


if __name__ == "__main__":
    main()
