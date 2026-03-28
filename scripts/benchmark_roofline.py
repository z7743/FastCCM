#!/usr/bin/env python3
"""Approximate roofline benchmark for FastCCM on CPU/CUDA.

This script estimates arithmetic intensity and achieved FLOP/s for representative
FastCCM workloads, then compares them against simple sustained bandwidth and GEMM
microbenchmarks measured on the same device.

The analysis is intentionally approximate:
- FLOPs and bytes are algorithmic estimates for the dominant tensor kernels.
- Memory traffic is not gathered from hardware counters.
- `mps` is explicitly unsupported to keep the benchmark on devices we can model
  more consistently from this environment.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import statistics
import sys
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


DEFAULT_NUM_THREADS = int(
    os.environ.get(
        "FASTCCM_TORCH_NUM_THREADS",
        os.environ.get("TORCH_NUM_THREADS", min(os.cpu_count() or 1, 8)),
    )
)
DEFAULT_NUM_INTEROP_THREADS = int(
    os.environ.get(
        "FASTCCM_TORCH_NUM_INTEROP_THREADS",
        os.environ.get("TORCH_NUM_INTEROP_THREADS", 1),
    )
)

DEFAULT_PAIRWISE_CASES: list[tuple[int, int, int]] = [
    (100, 100, 1000),
    (200, 200, 1000),
    (800, 800, 500),
    (100, 100, 8000),
]
DEFAULT_SINGLE_LENGTHS = [2_000, 8_000, 32_000, 128_000]

DEFAULT_DTYPE = "float32"
DEFAULT_PAIRWISE_MEMORY_BUDGET_GB = 1.0
DEFAULT_SINGLE_MEMORY_BUDGET_GB = 2.0
DEFAULT_METHODS = ("simplex", "smap")
DEFAULT_SCENARIOS = ("pairwise", "single")
DEFAULT_SIMPLEX_NEIGHBOR_BACKENDS = ("torch", "pykeops")

PAIRWISE_X_EMBED_DIM = 5
PAIRWISE_Y_EMBED_DIM = 1
PAIRWISE_TP = 0
PAIRWISE_EXCLUSION_WINDOW = 5

SINGLE_EMBED_DIM = 20
SINGLE_TAU = 1
SINGLE_TP = 1
SINGLE_EXCLUSION_WINDOW = 10

LIBRARY_SIZE: int | str | None = None
SAMPLE_SIZE: int | str | None = None
BATCH_SIZE: int | str | None = "auto"
SEED = 1234

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def load_pairwise_ccm_class():
    package_root = SRC_DIR / "fastccm"
    package_name = "fastccm"
    utils_package_name = "fastccm.utils"
    module_name = "fastccm.ccm"

    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_root)]
        sys.modules[package_name] = package

    if utils_package_name not in sys.modules:
        utils_package = types.ModuleType(utils_package_name)
        utils_package.__path__ = [str(package_root / "utils")]
        sys.modules[utils_package_name] = utils_package

    if module_name in sys.modules:
        return sys.modules[module_name].PairwiseCCM

    spec = importlib.util.spec_from_file_location(module_name, package_root / "ccm.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_name} from source tree.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.PairwiseCCM


def get_td_embedding_np(time_series: np.ndarray, dim: int, stride: int) -> np.ndarray:
    num_points, num_dims = time_series.shape
    window_size = (dim - 1) * stride + 1
    if num_points < window_size:
        raise ValueError("Time series is too short for the given dimensions and stride.")

    shape = (num_points - window_size + 1, dim, num_dims)
    strides = (
        time_series.strides[0],
        stride * time_series.strides[0],
        time_series.strides[1],
    )
    return np.lib.stride_tricks.as_strided(time_series, shape=shape, strides=strides)


PairwiseCCM = load_pairwise_ccm_class()


@dataclass(frozen=True)
class PairwiseCase:
    x_matrix_size: int
    y_matrix_size: int
    ts_length: int
    ex: int = PAIRWISE_X_EMBED_DIM
    ey: int = PAIRWISE_Y_EMBED_DIM


@dataclass(frozen=True)
class SingleCase:
    length: int
    ex: int = SINGLE_EMBED_DIM
    ey: int = 1


@dataclass(frozen=True)
class RooflineEstimate:
    flops: float
    bytes_moved: float

    @property
    def arithmetic_intensity(self) -> float:
        return 0.0 if self.bytes_moved <= 0 else self.flops / self.bytes_moved


@dataclass(frozen=True)
class BenchmarkResult:
    scenario: str
    method: str
    device: str
    dtype: str
    case_label: str
    x_matrix_size: int
    y_matrix_size: int
    length: int
    ex: int
    ey: int
    library_size: int
    sample_size: int
    exclusion_window: int
    neighbor_backend: str
    attempts: int
    avg_sec: float
    min_sec: float
    max_sec: float
    estimated_flops: float
    estimated_bytes: float
    arithmetic_intensity: float
    achieved_gflops: float
    achieved_gbytes_per_sec: float
    roofline_gflops: float
    roofline_limit: str
    peak_bandwidth_gbytes_per_sec: float
    peak_compute_gflops: float
    xtwx_precompute: bool
    xtwy_precompute: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Approximate roofline benchmark for FastCCM on CPU/CUDA."
    )
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, or auto")
    parser.add_argument(
        "--dtype",
        default=DEFAULT_DTYPE,
        choices=["float32", "float64"],
    )
    parser.add_argument(
        "--scenarios",
        default="pairwise,single",
        help="Comma-separated list from: pairwise,single",
    )
    parser.add_argument(
        "--methods",
        default="simplex,smap",
        help="Comma-separated list from: simplex,smap",
    )
    parser.add_argument(
        "--neighbor-backends",
        default="torch,pykeops",
        help="Comma-separated simplex neighbor backends from: torch,pykeops",
    )
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-size", default=BATCH_SIZE)
    parser.add_argument(
        "--single-lengths",
        type=int,
        nargs="+",
        default=None,
        help="Optional explicit single-series lengths.",
    )
    parser.add_argument(
        "--pairwise-cases",
        nargs="+",
        default=None,
        metavar="X,Y,T",
        help="Optional explicit pairwise cases such as 100,100,1000 200,200,1000.",
    )
    parser.add_argument("--pairwise-memory-budget-gb", type=float, default=DEFAULT_PAIRWISE_MEMORY_BUDGET_GB)
    parser.add_argument("--single-memory-budget-gb", type=float, default=DEFAULT_SINGLE_MEMORY_BUDGET_GB)
    parser.add_argument("--num-threads", type=int, default=DEFAULT_NUM_THREADS)
    parser.add_argument("--num-interop-threads", type=int, default=DEFAULT_NUM_INTEROP_THREADS)
    parser.add_argument("--xtwx-precompute", dest="xtwx_precompute", action="store_true", default=True)
    parser.add_argument("--no-xtwx-precompute", dest="xtwx_precompute", action="store_false")
    parser.add_argument("--xtwy-precompute", dest="xtwy_precompute", action="store_true", default=False)
    parser.add_argument("--no-xtwy-precompute", dest="xtwy_precompute", action="store_false")
    parser.add_argument("--bandwidth-numel", type=int, default=None)
    parser.add_argument("--gemm-size", type=int, default=None)
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output path.")
    parser.add_argument("--plot", type=Path, default=None, help="Optional output path for a roofline PNG.")
    return parser.parse_args()


def parse_csv_choices(spec: str, allowed: tuple[str, ...], flag: str) -> list[str]:
    values = [item.strip().lower() for item in spec.split(",") if item.strip()]
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown {flag}: {', '.join(unknown)}")
    return values


def resolve_device(spec: str) -> str:
    lowered = spec.strip().lower()
    if lowered.startswith("mps"):
        raise ValueError("MPS is explicitly unsupported for this roofline script.")
    if lowered == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if lowered.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError(f"Requested device {spec!r}, but CUDA is not available.")
        return spec
    if lowered == "cpu":
        return "cpu"
    raise ValueError(f"Unsupported device: {spec!r}. Use cpu, cuda[:N], or auto.")


def resolve_dtype(name: str) -> torch.dtype:
    return getattr(torch, name)


def resolve_size(size: int | str | None, available_points: int, auto_divisor: int) -> int:
    if size is None:
        return available_points
    if size == "auto":
        return min(max(available_points // auto_divisor, 1), available_points)
    return min(int(size), available_points)


def resolve_pairwise_exclusion_window(method: str, case: PairwiseCase, library_size: int) -> int:
    if method != "simplex":
        return PAIRWISE_EXCLUSION_WINDOW

    required_neighbors = case.ex + 1
    if library_size <= required_neighbors:
        raise ValueError(
            "Invalid pairwise case: library_size must exceed simplex neighbor count. "
            f"Got library_size={library_size}, required>{required_neighbors}."
        )

    max_exclusion_window = max((library_size - required_neighbors - 1) // 2, 0)
    return min(PAIRWISE_EXCLUSION_WINDOW, max_exclusion_window)


def resolve_single_parameters(method: str, case: SingleCase) -> tuple[int, int, int, int]:
    embedded_length = case.length - (case.ex - 1) * SINGLE_TAU
    valid_points = embedded_length - SINGLE_TP
    if valid_points <= 0:
        raise ValueError(
            f"length={case.length} is too short for E={case.ex}, tau={SINGLE_TAU}, tp={SINGLE_TP}."
        )

    library_size = resolve_size(LIBRARY_SIZE, valid_points, auto_divisor=2)
    sample_size = resolve_size(SAMPLE_SIZE, valid_points, auto_divisor=6)

    if method != "simplex":
        return embedded_length, library_size, sample_size, SINGLE_EXCLUSION_WINDOW

    required_neighbors = case.ex + 1
    if library_size <= required_neighbors:
        raise ValueError(
            "Invalid single-series case: library_size must exceed simplex neighbor count. "
            f"Got library_size={library_size}, required>{required_neighbors}."
        )

    max_exclusion_window = max((library_size - required_neighbors - 1) // 2, 0)
    exclusion_window = min(SINGLE_EXCLUSION_WINDOW, max_exclusion_window)
    return embedded_length, library_size, sample_size, exclusion_window


def maybe_sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def dtype_num_bytes(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def normalized_seed(seed: int) -> int:
    return int(seed) % (2 ** 63 - 1)


def default_bandwidth_numel(device: str) -> int:
    return 64_000_000 if device.startswith("cuda") else 16_000_000


def default_gemm_size(device: str) -> int:
    return 4096 if device.startswith("cuda") else 2048


def measure_peak_bandwidth(
    *,
    device: str,
    dtype: torch.dtype,
    numel: int,
    attempts: int,
    warmups: int,
) -> float:
    x = torch.empty(numel, device=device, dtype=dtype)
    y = torch.randn(numel, device=device, dtype=dtype)
    z = torch.randn(numel, device=device, dtype=dtype)
    alpha = 1.61803398875

    with torch.inference_mode():
        for _ in range(warmups):
            torch.add(y, z, alpha=alpha, out=x)
            maybe_sync(device)

        best_time = math.inf
        for _ in range(attempts):
            maybe_sync(device)
            t0 = time.perf_counter()
            torch.add(y, z, alpha=alpha, out=x)
            maybe_sync(device)
            best_time = min(best_time, time.perf_counter() - t0)

    bytes_moved = 3.0 * numel * dtype_num_bytes(dtype)
    return bytes_moved / best_time


def measure_peak_gemm_flops(
    *,
    device: str,
    dtype: torch.dtype,
    size: int,
    attempts: int,
    warmups: int,
) -> float:
    a = torch.randn((size, size), device=device, dtype=dtype)
    b = torch.randn((size, size), device=device, dtype=dtype)

    with torch.inference_mode():
        for _ in range(warmups):
            _ = torch.matmul(a, b)
            maybe_sync(device)

        best_time = math.inf
        for _ in range(attempts):
            maybe_sync(device)
            t0 = time.perf_counter()
            _ = torch.matmul(a, b)
            maybe_sync(device)
            best_time = min(best_time, time.perf_counter() - t0)

    flops = 2.0 * size * size * size
    return flops / best_time


def parse_pairwise_cases(specs: list[str] | None) -> list[tuple[int, int, int]]:
    if not specs:
        return DEFAULT_PAIRWISE_CASES

    parsed: list[tuple[int, int, int]] = []
    for spec in specs:
        parts = [part.strip() for part in spec.split(",")]
        if len(parts) != 3:
            raise ValueError(
                f"Invalid pairwise case {spec!r}. Expected the form X,Y,T."
            )
        parsed.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return parsed


def build_pairwise_cases(specs: list[str] | None) -> list[PairwiseCase]:
    return [
        PairwiseCase(x_matrix_size=x, y_matrix_size=y, ts_length=t)
        for x, y, t in parse_pairwise_cases(specs)
    ]


def build_single_cases(lengths: list[int] | None) -> list[SingleCase]:
    return [SingleCase(length=length) for length in (lengths or DEFAULT_SINGLE_LENGTHS)]


def generate_pairwise_embeddings(
    rng: np.random.Generator,
    case: PairwiseCase,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    x_emb = [
        rng.standard_normal((case.ts_length, case.ex), dtype=np.float32)
        for _ in range(case.x_matrix_size)
    ]
    y_emb = [
        rng.standard_normal((case.ts_length, case.ey), dtype=np.float32)
        for _ in range(case.y_matrix_size)
    ]
    return x_emb, y_emb


def generate_single_embeddings(
    rng: np.random.Generator,
    case: SingleCase,
) -> tuple[np.ndarray, np.ndarray]:
    series = rng.standard_normal(case.length, dtype=np.float32)
    x_emb = get_td_embedding_np(series[:, None], dim=case.ex, stride=SINGLE_TAU)[:, :, 0]
    y_emb = series[:, None].astype(np.float32, copy=False)
    return x_emb.astype(np.float32, copy=False), y_emb


def estimate_simplex(
    *,
    n_x: int,
    n_y: int,
    library_size: int,
    sample_size: int,
    ex: int,
    ey: int,
    k: int,
    dtype_size: int,
    compute_dtype_size: int,
    neighbor_backend: str,
) -> RooflineEstimate:
    pairs = n_x * sample_size * library_size
    neighbors = n_x * sample_size * k
    gathered_targets = n_x * sample_size * k * n_y * ey
    predictions = sample_size * ey * n_y * n_x

    cdist_flops = pairs * (4.0 * ex + 5.0)
    weight_flops = neighbors * 4.0 + (n_x * sample_size * (2.0 * k))
    weighted_avg_flops = 2.0 * gathered_targets
    metric_flops = 8.0 * predictions
    total_flops = cdist_flops + weight_flops + weighted_avg_flops + metric_flops

    if neighbor_backend == "pykeops":
        cdist_bytes = compute_dtype_size * (
            (n_x * sample_size * ex) +
            (n_x * library_size * ex)
        )
        select_bytes = (compute_dtype_size + 8) * neighbors
    else:
        cdist_bytes = compute_dtype_size * (
            (n_x * sample_size * ex) +
            (n_x * library_size * ex) +
            pairs
        )
        select_bytes = (compute_dtype_size * pairs) + ((compute_dtype_size + 8) * neighbors)
    weight_bytes = (compute_dtype_size * neighbors * 3.0) + (dtype_size * neighbors)
    gather_bytes = compute_dtype_size * gathered_targets * 2.0
    weighted_avg_bytes = (
        (compute_dtype_size * gathered_targets) +
        (compute_dtype_size * neighbors) +
        (dtype_size * predictions)
    )
    metric_bytes = compute_dtype_size * predictions * 2.0
    total_bytes = (
        cdist_bytes +
        select_bytes +
        weight_bytes +
        gather_bytes +
        weighted_avg_bytes +
        metric_bytes
    )
    return RooflineEstimate(flops=total_flops, bytes_moved=total_bytes)


def estimate_smap(
    *,
    n_x: int,
    n_y: int,
    library_size: int,
    sample_size: int,
    ex: int,
    ey: int,
    dtype_size: int,
    compute_dtype_size: int,
    xtwx_precompute: bool,
    xtwy_precompute: bool,
) -> RooflineEstimate:
    ex1 = ex + 1
    rhs = n_y * ey
    pairs = n_x * sample_size * library_size
    predictions = sample_size * ey * n_y * n_x

    cdist_flops = pairs * (4.0 * ex + 5.0)
    local_weight_flops = pairs * 5.0
    square_flops = pairs

    xtwx_feature_flops = 0.0
    xtwx_feature_bytes = 0.0
    if xtwx_precompute:
        xtwx_feature_flops = 2.0 * n_x * library_size * ex1 * ex1
        xtwx_feature_bytes = compute_dtype_size * (
            (n_x * library_size * ex1) +
            (n_x * library_size * ex1 * ex1)
        )

    xtwy_feature_flops = 0.0
    xtwy_feature_bytes = 0.0
    if xtwy_precompute:
        xtwy_feature_flops = float(n_x * library_size * ex1 * rhs)
        xtwy_feature_bytes = compute_dtype_size * (
            (n_x * library_size * ex1) +
            (library_size * rhs) +
            (n_x * library_size * ex1 * rhs)
        )

    xtwx_flops = 2.0 * n_x * sample_size * library_size * ex1 * ex1
    xtwy_flops = 2.0 * n_x * sample_size * library_size * ex1 * rhs
    cholesky_flops = (n_x * sample_size) * (ex1 ** 3) / 3.0
    solve_flops = 2.0 * (n_x * sample_size) * (ex1 ** 2) * rhs
    predict_flops = 2.0 * n_x * sample_size * ex1 * rhs
    metric_flops = 8.0 * predictions

    total_flops = (
        cdist_flops +
        local_weight_flops +
        square_flops +
        xtwx_feature_flops +
        xtwy_feature_flops +
        xtwx_flops +
        xtwy_flops +
        cholesky_flops +
        solve_flops +
        predict_flops +
        metric_flops
    )

    local_weight_bytes = compute_dtype_size * (
        (n_x * sample_size * ex) +
        (n_x * library_size * ex) +
        (pairs * 2.0)
    )
    xtwx_bytes = compute_dtype_size * (
        pairs +
        (n_x * library_size * ex1 * ex1 if xtwx_precompute else n_x * library_size * ex1) +
        (n_x * sample_size * ex1 * ex1)
    )
    if xtwy_precompute:
        xtwy_input_bytes = n_x * library_size * ex1 * rhs
    else:
        xtwy_input_bytes = (n_x * sample_size * ex1 * library_size) + (library_size * rhs)
    xtwy_bytes = compute_dtype_size * (
        pairs +
        xtwy_input_bytes +
        (n_x * sample_size * ex1 * rhs)
    )
    solve_bytes = compute_dtype_size * (
        (n_x * sample_size * ex1 * ex1 * 2.0) +
        (n_x * sample_size * ex1 * rhs * 2.0)
    )
    predict_bytes = compute_dtype_size * (
        (n_x * sample_size * ex1) +
        (n_x * sample_size * ex1 * rhs) +
        predictions
    )
    metric_bytes = compute_dtype_size * predictions * 2.0
    total_bytes = (
        local_weight_bytes +
        xtwx_feature_bytes +
        xtwy_feature_bytes +
        xtwx_bytes +
        xtwy_bytes +
        solve_bytes +
        predict_bytes +
        metric_bytes +
        (dtype_size * predictions)
    )
    return RooflineEstimate(flops=total_flops, bytes_moved=total_bytes)


def roofline_limit_kind(arithmetic_intensity: float, peak_bandwidth_bytes_sec: float, peak_compute_flops_sec: float) -> str:
    ridge_point = peak_compute_flops_sec / peak_bandwidth_bytes_sec
    return "memory" if arithmetic_intensity < ridge_point else "compute"


def run_pairwise_case(
    *,
    method: str,
    case: PairwiseCase,
    ccm: PairwiseCCM,
    attempts: int,
    warmups: int,
    base_seed: int,
    batch_size: int | str | None,
    dtype: torch.dtype,
    peak_bandwidth_bytes_sec: float,
    peak_compute_flops_sec: float,
    xtwx_precompute: bool,
    xtwy_precompute: bool,
    neighbor_backend: str,
) -> BenchmarkResult:
    valid_points = case.ts_length - PAIRWISE_TP
    if valid_points <= 1:
        raise ValueError(f"ts_length={case.ts_length} leaves no valid query points.")

    library_size = resolve_size(LIBRARY_SIZE, valid_points, auto_divisor=2)
    sample_size = resolve_size(SAMPLE_SIZE, valid_points, auto_divisor=6)
    exclusion_window = resolve_pairwise_exclusion_window(method, case, library_size)

    score_kwargs = {}
    if method == "simplex":
        score_kwargs["neighbor_backend"] = neighbor_backend

    for warmup_idx in range(warmups):
        seed = normalized_seed(base_seed + 1_000_000 + warmup_idx)
        rng = np.random.default_rng(seed)
        x_emb, y_emb = generate_pairwise_embeddings(rng, case)
        _ = ccm.score_matrix(
            X_emb=x_emb,
            Y_emb=y_emb,
            library_size=library_size,
            sample_size=sample_size,
            exclusion_window=exclusion_window,
            tp=PAIRWISE_TP,
            method=method,
            metric="corr",
            batch_size=batch_size,
            seed=seed,
            xtwx_precompute=xtwx_precompute,
            xtwy_precompute=xtwy_precompute,
            clean_after=False,
            **score_kwargs,
        )
        maybe_sync(ccm.device)

    timings: list[float] = []
    for attempt in range(attempts):
        seed = normalized_seed(base_seed + attempt)
        rng = np.random.default_rng(seed)
        x_emb, y_emb = generate_pairwise_embeddings(rng, case)

        maybe_sync(ccm.device)
        t0 = time.perf_counter()
        _ = ccm.score_matrix(
            X_emb=x_emb,
            Y_emb=y_emb,
            library_size=library_size,
            sample_size=sample_size,
            exclusion_window=exclusion_window,
            tp=PAIRWISE_TP,
            method=method,
            metric="corr",
            batch_size=batch_size,
            seed=seed,
            xtwx_precompute=xtwx_precompute,
            xtwy_precompute=xtwy_precompute,
            clean_after=False,
            **score_kwargs,
        )
        maybe_sync(ccm.device)
        timings.append(time.perf_counter() - t0)

    estimate = (
        estimate_simplex(
            n_x=case.x_matrix_size,
            n_y=case.y_matrix_size,
            library_size=library_size,
            sample_size=sample_size,
            ex=case.ex,
            ey=case.ey,
            k=case.ex + 1,
            dtype_size=dtype_num_bytes(dtype),
            compute_dtype_size=dtype_num_bytes(dtype),
            neighbor_backend=neighbor_backend,
        )
        if method == "simplex"
        else estimate_smap(
            n_x=case.x_matrix_size,
            n_y=case.y_matrix_size,
            library_size=library_size,
            sample_size=sample_size,
            ex=case.ex,
            ey=case.ey,
            dtype_size=dtype_num_bytes(dtype),
            compute_dtype_size=dtype_num_bytes(dtype),
            xtwx_precompute=xtwx_precompute,
            xtwy_precompute=xtwy_precompute,
        )
    )

    avg_sec = statistics.fmean(timings)
    achieved_gflops = estimate.flops / avg_sec / 1e9
    achieved_gbytes_per_sec = estimate.bytes_moved / avg_sec / 1e9
    roofline_gflops = min(
        peak_compute_flops_sec / 1e9,
        (peak_bandwidth_bytes_sec / 1e9) * estimate.arithmetic_intensity,
    )
    limit = roofline_limit_kind(
        estimate.arithmetic_intensity,
        peak_bandwidth_bytes_sec,
        peak_compute_flops_sec,
    )
    return BenchmarkResult(
        scenario="pairwise",
        method=method,
        device=ccm.device,
        dtype=str(dtype).split(".")[-1],
        case_label=f"{case.x_matrix_size}x{case.y_matrix_size}_T{case.ts_length}",
        x_matrix_size=case.x_matrix_size,
        y_matrix_size=case.y_matrix_size,
        length=case.ts_length,
        ex=case.ex,
        ey=case.ey,
        library_size=library_size,
        sample_size=sample_size,
        exclusion_window=exclusion_window,
        neighbor_backend=neighbor_backend,
        attempts=attempts,
        avg_sec=avg_sec,
        min_sec=min(timings),
        max_sec=max(timings),
        estimated_flops=estimate.flops,
        estimated_bytes=estimate.bytes_moved,
        arithmetic_intensity=estimate.arithmetic_intensity,
        achieved_gflops=achieved_gflops,
        achieved_gbytes_per_sec=achieved_gbytes_per_sec,
        roofline_gflops=roofline_gflops,
        roofline_limit=limit,
        peak_bandwidth_gbytes_per_sec=peak_bandwidth_bytes_sec / 1e9,
        peak_compute_gflops=peak_compute_flops_sec / 1e9,
        xtwx_precompute=xtwx_precompute,
        xtwy_precompute=xtwy_precompute,
    )


def run_single_case(
    *,
    method: str,
    case: SingleCase,
    ccm: PairwiseCCM,
    attempts: int,
    warmups: int,
    base_seed: int,
    batch_size: int | str | None,
    dtype: torch.dtype,
    peak_bandwidth_bytes_sec: float,
    peak_compute_flops_sec: float,
    xtwx_precompute: bool,
    xtwy_precompute: bool,
    neighbor_backend: str,
) -> BenchmarkResult:
    _, library_size, sample_size, exclusion_window = resolve_single_parameters(method, case)

    score_kwargs = {}
    if method == "simplex":
        score_kwargs["neighbor_backend"] = neighbor_backend

    for warmup_idx in range(warmups):
        seed = normalized_seed(base_seed + 1_000_000 + warmup_idx)
        rng = np.random.default_rng(seed)
        x_emb, y_emb = generate_single_embeddings(rng, case)
        _ = ccm.score_matrix(
            X_emb=[x_emb],
            Y_emb=[y_emb],
            library_size=library_size,
            sample_size=sample_size,
            exclusion_window=exclusion_window,
            tp=SINGLE_TP,
            method=method,
            metric="corr",
            batch_size=batch_size,
            seed=seed,
            xtwx_precompute=xtwx_precompute,
            xtwy_precompute=xtwy_precompute,
            clean_after=False,
            **score_kwargs,
        )
        maybe_sync(ccm.device)

    timings: list[float] = []
    for attempt in range(attempts):
        seed = normalized_seed(base_seed + attempt)
        rng = np.random.default_rng(seed)
        x_emb, y_emb = generate_single_embeddings(rng, case)

        maybe_sync(ccm.device)
        t0 = time.perf_counter()
        _ = ccm.score_matrix(
            X_emb=[x_emb],
            Y_emb=[y_emb],
            library_size=library_size,
            sample_size=sample_size,
            exclusion_window=exclusion_window,
            tp=SINGLE_TP,
            method=method,
            metric="corr",
            batch_size=batch_size,
            seed=seed,
            xtwx_precompute=xtwx_precompute,
            xtwy_precompute=xtwy_precompute,
            clean_after=False,
            **score_kwargs,
        )
        maybe_sync(ccm.device)
        timings.append(time.perf_counter() - t0)

    effective_xtwy = xtwy_precompute if method == "smap" else False
    estimate = (
        estimate_simplex(
            n_x=1,
            n_y=1,
            library_size=library_size,
            sample_size=sample_size,
            ex=case.ex,
            ey=case.ey,
            k=case.ex + 1,
            dtype_size=dtype_num_bytes(dtype),
            compute_dtype_size=dtype_num_bytes(dtype),
            neighbor_backend=neighbor_backend,
        )
        if method == "simplex"
        else estimate_smap(
            n_x=1,
            n_y=1,
            library_size=library_size,
            sample_size=sample_size,
            ex=case.ex,
            ey=case.ey,
            dtype_size=dtype_num_bytes(dtype),
            compute_dtype_size=dtype_num_bytes(dtype),
            xtwx_precompute=xtwx_precompute,
            xtwy_precompute=effective_xtwy,
        )
    )

    avg_sec = statistics.fmean(timings)
    achieved_gflops = estimate.flops / avg_sec / 1e9
    achieved_gbytes_per_sec = estimate.bytes_moved / avg_sec / 1e9
    roofline_gflops = min(
        peak_compute_flops_sec / 1e9,
        (peak_bandwidth_bytes_sec / 1e9) * estimate.arithmetic_intensity,
    )
    limit = roofline_limit_kind(
        estimate.arithmetic_intensity,
        peak_bandwidth_bytes_sec,
        peak_compute_flops_sec,
    )
    return BenchmarkResult(
        scenario="single",
        method=method,
        device=ccm.device,
        dtype=str(dtype).split(".")[-1],
        case_label=f"T{case.length}",
        x_matrix_size=1,
        y_matrix_size=1,
        length=case.length,
        ex=case.ex,
        ey=case.ey,
        library_size=library_size,
        sample_size=sample_size,
        exclusion_window=exclusion_window,
        neighbor_backend=neighbor_backend,
        attempts=attempts,
        avg_sec=avg_sec,
        min_sec=min(timings),
        max_sec=max(timings),
        estimated_flops=estimate.flops,
        estimated_bytes=estimate.bytes_moved,
        arithmetic_intensity=estimate.arithmetic_intensity,
        achieved_gflops=achieved_gflops,
        achieved_gbytes_per_sec=achieved_gbytes_per_sec,
        roofline_gflops=roofline_gflops,
        roofline_limit=limit,
        peak_bandwidth_gbytes_per_sec=peak_bandwidth_bytes_sec / 1e9,
        peak_compute_gflops=peak_compute_flops_sec / 1e9,
        xtwx_precompute=xtwx_precompute,
        xtwy_precompute=effective_xtwy,
    )


def write_csv(path: Path, rows: list[BenchmarkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def plot_roofline(
    *,
    path: Path,
    rows: list[BenchmarkResult],
    peak_bandwidth_gbytes_per_sec: float,
    peak_compute_gflops: float,
    title: str,
) -> None:
    if plt is None:
        raise RuntimeError(
            "matplotlib is required for --plot, but it is not installed in this environment."
        )
    path.parent.mkdir(parents=True, exist_ok=True)

    x_values = np.array([max(row.arithmetic_intensity, 1e-6) for row in rows], dtype=float)
    y_values = np.array([max(row.achieved_gflops, 1e-6) for row in rows], dtype=float)

    xmin = max(min(x_values) / 2.0, 1e-6)
    xmax = max(x_values) * 2.0
    roof_x = np.logspace(math.log10(xmin), math.log10(xmax), 256)
    roof_y = np.minimum(peak_compute_gflops, peak_bandwidth_gbytes_per_sec * roof_x)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(roof_x, roof_y, color="black", linewidth=2.0, label="Measured roofline")

    colors = {
        ("pairwise", "simplex"): "#005f73",
        ("pairwise", "smap"): "#ae2012",
        ("single", "simplex"): "#0a9396",
        ("single", "smap"): "#ca6702",
    }
    for row in rows:
        key = (row.scenario, row.method)
        ax.scatter(
            max(row.arithmetic_intensity, 1e-6),
            max(row.achieved_gflops, 1e-6),
            color=colors.get(key, "#444444"),
            s=64,
        )
        ax.annotate(
            f"{row.scenario}:{row.method}:{row.neighbor_backend}:{row.case_label}",
            (max(row.arithmetic_intensity, 1e-6), max(row.achieved_gflops, 1e-6)),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )

    ridge_x = peak_compute_gflops / peak_bandwidth_gbytes_per_sec
    ax.axvline(ridge_x, linestyle="--", color="#666666", linewidth=1.0)

    ax.set_title(title)
    ax.set_xlabel("Arithmetic intensity [FLOP/byte]")
    ax.set_ylabel("Performance [GFLOP/s]")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    scenarios = parse_csv_choices(args.scenarios, DEFAULT_SCENARIOS, "--scenarios")
    methods = parse_csv_choices(args.methods, DEFAULT_METHODS, "--methods")
    simplex_neighbor_backends = parse_csv_choices(
        args.neighbor_backends,
        DEFAULT_SIMPLEX_NEIGHBOR_BACKENDS,
        "--neighbor-backends",
    )
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)

    batch_size: int | str | None
    if args.batch_size == "auto":
        batch_size = "auto"
    elif args.batch_size == "none":
        batch_size = None
    else:
        batch_size = int(args.batch_size)

    torch.set_num_threads(args.num_threads)
    torch.set_num_interop_threads(args.num_interop_threads)

    bandwidth_numel = args.bandwidth_numel or default_bandwidth_numel(device)
    gemm_size = args.gemm_size or default_gemm_size(device)

    peak_bandwidth_bytes_sec = measure_peak_bandwidth(
        device=device,
        dtype=dtype,
        numel=bandwidth_numel,
        attempts=max(args.attempts, 2),
        warmups=max(args.warmups, 1),
    )
    peak_compute_flops_sec = measure_peak_gemm_flops(
        device=device,
        dtype=dtype,
        size=gemm_size,
        attempts=max(args.attempts, 2),
        warmups=max(args.warmups, 1),
    )

    rows: list[BenchmarkResult] = []
    for method_idx, method in enumerate(methods):
        method_neighbor_backends = simplex_neighbor_backends if method == "simplex" else ["n/a"]
        if "pairwise" in scenarios:
            for backend_idx, neighbor_backend in enumerate(method_neighbor_backends):
                ccm = PairwiseCCM(
                    device=device,
                    dtype=args.dtype,
                    memory_budget_gb=args.pairwise_memory_budget_gb,
                    verbose=0,
                )
                for case_idx, case in enumerate(build_pairwise_cases(args.pairwise_cases), start=1):
                    rows.append(
                        run_pairwise_case(
                            method=method,
                            case=case,
                            ccm=ccm,
                            attempts=args.attempts,
                            warmups=args.warmups,
                            base_seed=args.seed + method_idx * 1_000_000 + backend_idx * 100_000 + case_idx * 10_000,
                            batch_size=batch_size,
                            dtype=dtype,
                            peak_bandwidth_bytes_sec=peak_bandwidth_bytes_sec,
                            peak_compute_flops_sec=peak_compute_flops_sec,
                            xtwx_precompute=args.xtwx_precompute,
                            xtwy_precompute=args.xtwy_precompute,
                            neighbor_backend=neighbor_backend,
                        )
                    )

        if "single" in scenarios:
            single_xtwy = args.xtwy_precompute or (method == "smap")
            for backend_idx, neighbor_backend in enumerate(method_neighbor_backends):
                ccm = PairwiseCCM(
                    device=device,
                    dtype=args.dtype,
                    memory_budget_gb=args.single_memory_budget_gb,
                    verbose=0,
                )
                for case_idx, case in enumerate(build_single_cases(args.single_lengths), start=1):
                    rows.append(
                        run_single_case(
                            method=method,
                            case=case,
                            ccm=ccm,
                            attempts=args.attempts,
                            warmups=args.warmups,
                            base_seed=args.seed + method_idx * 1_000_000 + 500_000 + backend_idx * 100_000 + case_idx * 10_000,
                            batch_size=batch_size,
                            dtype=dtype,
                            peak_bandwidth_bytes_sec=peak_bandwidth_bytes_sec,
                            peak_compute_flops_sec=peak_compute_flops_sec,
                            xtwx_precompute=args.xtwx_precompute,
                            xtwy_precompute=single_xtwy,
                            neighbor_backend=neighbor_backend,
                        )
                    )

    if not rows:
        raise ValueError("No benchmark rows were produced.")

    print("Roofline settings:")
    print(f"  device={device}")
    print(f"  dtype={args.dtype}")
    print(f"  scenarios={scenarios}")
    print(f"  methods={methods}")
    print(f"  simplex_neighbor_backends={simplex_neighbor_backends}")
    print(f"  attempts={args.attempts}")
    print(f"  warmups={args.warmups}")
    print(f"  batch_size={batch_size}")
    print(f"  pairwise_cases={args.pairwise_cases or DEFAULT_PAIRWISE_CASES}")
    print(f"  single_lengths={args.single_lengths or DEFAULT_SINGLE_LENGTHS}")
    print(f"  pairwise_memory_budget_gb={args.pairwise_memory_budget_gb}")
    print(f"  single_memory_budget_gb={args.single_memory_budget_gb}")
    print(f"  xtwx_precompute={args.xtwx_precompute}")
    print(f"  xtwy_precompute={args.xtwy_precompute}")
    print(f"  torch_num_threads={args.num_threads}")
    print(f"  torch_num_interop_threads={args.num_interop_threads}")
    print(f"  bandwidth_numel={bandwidth_numel}")
    print(f"  gemm_size={gemm_size}")
    print(f"  peak_bandwidth_gbytes_per_sec={peak_bandwidth_bytes_sec / 1e9:.3f}")
    print(f"  peak_compute_gflops={peak_compute_flops_sec / 1e9:.3f}")
    print()

    fieldnames = list(asdict(rows[0]).keys())
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(asdict(row))

    if args.csv is not None:
        write_csv(args.csv, rows)

    if args.plot is not None:
        plot_roofline(
            path=args.plot,
            rows=rows,
            peak_bandwidth_gbytes_per_sec=peak_bandwidth_bytes_sec / 1e9,
            peak_compute_gflops=peak_compute_flops_sec / 1e9,
            title=f"FastCCM Approximate Roofline ({device}, {args.dtype})",
        )


if __name__ == "__main__":
    main()
