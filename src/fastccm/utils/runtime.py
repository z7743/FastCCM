import gc
import logging
import math
import sys
import time
from contextlib import contextmanager

import torch
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def is_oom_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(s in msg for s in (
        "cuda out of memory", "out of memory", "failed to allocate",
        "can't allocate memory", "cublas status alloc failed", "mps allocation failed"
    ))


def soft_clear(logger: logging.Logger, device: str):
    logger.info("Running soft memory cleanup")
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Soft memory cleanup finished")


def hard_clear(logger: logging.Logger, device: str):
    logger.warning("Running hard memory cleanup after OOM")
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass
    logger.warning("Hard memory cleanup finished")


def format_bytes(n):
    if n is None:
        return "n/a"
    n = float(n)
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    i = 0
    while n >= 1024.0 and i < len(units) - 1:
        n /= 1024.0
        i += 1
    return f"{n:.2f}{units[i]}"


def tic(logger: logging.Logger, device: str):
    if logger.isEnabledFor(logging.DEBUG) and device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    return time.perf_counter()


def toc_ms(logger: logging.Logger, device: str, start):
    if logger.isEnabledFor(logging.DEBUG) and device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    return (time.perf_counter() - start) * 1000.0


def format_ms(ms):
    return f"{ms:.3f}ms"


@contextmanager
def time_block(logger: logging.Logger, device: str, timings: dict, key: str):
    if not logger.isEnabledFor(logging.DEBUG):
        yield
        return
    t0 = tic(logger, device)
    try:
        yield
    finally:
        timings[key] = toc_ms(logger, device, t0)


def timings_summary(timings: dict, order=None):
    keys = order if order is not None else timings.keys()
    return " ".join(f"{k}={format_ms(timings[k])}" for k in keys if k in timings)


def _dtype_bytes(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def _shape_bytes(shape, itemsize: int) -> int:
    return int(math.prod(int(dim) for dim in shape) * int(itemsize))


def _resolve_batch_size(total_samples: int, budget_bytes: int, base_bytes: int, per_sample_bytes: int):
    if total_samples <= 0:
        return 0
    if per_sample_bytes <= 0:
        return min(int(total_samples), 1)

    available_bytes = int(budget_bytes) - int(base_bytes)
    if available_bytes <= 0:
        return 1

    batch_size = available_bytes // int(per_sample_bytes)
    batch_size = max(1, int(batch_size))
    batch_size = min(batch_size, int(total_samples))
    return int(batch_size)


def auto_batch_size_smap(
    X_lib,
    X_sample,
    Y_lib_s,
    *,
    dtype,
    compute_dtype,
    budget_gb=2.0,
    xtwx_precompute=True,
    xtwy_precompute=False,
):
    num_ts_X, L, max_E_X = X_lib.shape
    num_ts_Y, _, max_E_Y = Y_lib_s.shape
    S = X_sample.shape[1]

    cbytes = _dtype_bytes(compute_dtype)
    ex1 = int(max_E_X + 1)
    nX = int(num_ts_X)
    nY = int(num_ts_Y)
    Ey = int(max_E_Y)
    L = int(L)
    budget_bytes = int(budget_gb * (1024 ** 3) * 0.90)

    base_bytes = cbytes * (
        nX * L * max_E_X +  # Xc
        nY * L * Ey +       # Yc
        nX * L +            # onesL
        nX * L * ex1 +      # Xint
        nX * ex1 * L +      # Xint_t
        L * nY * Ey         # Yc_flat
    )
    if xtwx_precompute:
        base_bytes += cbytes * (nX * L * ex1 * ex1)
    if xtwy_precompute:
        base_bytes += cbytes * (nX * L * ex1 * nY * Ey)

    local_weights_per_sample = cbytes * (nX * L)
    weighted_design_per_sample = 0 if xtwy_precompute else cbytes * (nX * ex1 * L)
    solve_per_sample = cbytes * (
        2 * nX * ex1 * ex1 +
        2 * nX * ex1 * nY * Ey +
        nX * ex1 +
        nX * nY * Ey
    )
    per_sample_bytes = int((local_weights_per_sample + weighted_design_per_sample + solve_per_sample) * 1.10)

    B = _resolve_batch_size(int(S), budget_bytes, base_bytes, per_sample_bytes)
    return B, {
        "base_bytes": int(base_bytes),
        "per_sample_bytes": int(per_sample_bytes),
        "budget_bytes": int(budget_bytes),
        "estimated_peak_bytes": int(base_bytes + B * max(per_sample_bytes, 0)),
    }


def auto_batch_size_simplex(
    X_lib,
    X_sample,
    Y_lib_s,
    nbrs_num_max,
    *,
    dtype,
    compute_dtype,
    budget_gb=2.0,
):
    num_ts_X, L, _ = X_lib.shape
    num_ts_Y, _, max_EY = Y_lib_s.shape
    S = X_sample.shape[1]
    nX, nY, Ey, K = int(num_ts_X), int(num_ts_Y), int(max_EY), int(nbrs_num_max)

    cbytes = _dtype_bytes(compute_dtype)
    dbytes = _dtype_bytes(dtype)
    ibytes = 8  # int64 indices
    budget_bytes = int(budget_gb * (1024 ** 3) * 0.90)

    base_bytes = cbytes * (L * nY * Ey)  # Y_lib_lne

    search_per_sample = (
        cbytes * (nX * L + nX * K) +
        (dbytes + ibytes) * (nX * K)
    )
    reduce_per_sample = (
        cbytes * (nX * K * nY * Ey + nX * nY * Ey) +
        dbytes * (nX * nY * Ey)
    )
    per_sample_bytes = int(max(search_per_sample, reduce_per_sample) * 1.10)

    B = _resolve_batch_size(int(S), budget_bytes, base_bytes, per_sample_bytes)
    return B, {
        "base_bytes": int(base_bytes),
        "per_sample_bytes": int(per_sample_bytes),
        "budget_bytes": int(budget_bytes),
        "estimated_peak_bytes": int(base_bytes + B * max(per_sample_bytes, 0)),
    }


def smap_xtwx_precompute_bytes(X_lib, *, compute_dtype):
    num_ts_X, L, max_E_X = X_lib.shape
    ex1 = int(max_E_X + 1)
    cbytes = torch.tensor([], dtype=compute_dtype).element_size()
    return int(num_ts_X) * int(L) * ex1 * ex1 * cbytes


def smap_xtwy_precompute_bytes(X_lib, Y_lib_s, *, compute_dtype):
    num_ts_X, L, max_E_X = X_lib.shape
    num_ts_Y, _, max_E_Y = Y_lib_s.shape
    ex1 = int(max_E_X + 1)
    cbytes = torch.tensor([], dtype=compute_dtype).element_size()
    return int(num_ts_X) * int(L) * ex1 * int(num_ts_Y) * int(max_E_Y) * cbytes


def batch_starts(logger: logging.Logger, total_samples, sample_batch_size, desc):
    starts = range(0, total_samples, sample_batch_size)
    if not logger.isEnabledFor(logging.INFO):
        return starts
    if tqdm is None:
        return starts
    return tqdm(
        starts,
        total=int(math.ceil(total_samples / sample_batch_size)),
        desc=desc,
        unit="batch",
        file=sys.stderr,
        leave=False,
        dynamic_ncols=True,
    )
