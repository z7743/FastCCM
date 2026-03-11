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


def auto_batch_size_smap(X_lib, X_sample, Y_lib_s, *, dtype, compute_dtype, budget_gb=2.0):
    num_ts_X, L, max_E_X = X_lib.shape
    num_ts_Y, _, max_E_Y = Y_lib_s.shape
    S = X_sample.shape[1]

    cbytes = torch.tensor([], dtype=compute_dtype).element_size()
    dbytes = torch.tensor([], dtype=dtype).element_size()
    ex1 = int(max_E_X + 1)
    nX = int(num_ts_X)
    nY = int(num_ts_Y)
    Ey = int(max_E_Y)
    L = int(L)

    per_sample_bytes = cbytes * (
        nX * L +                  # dist
        nX * L +                  # weights
        nX * L +                  # w2
        nX * L * int(max_E_X) +   # Xc
        nY * L * Ey +             # Yc
        nX * L +                  # onesL
        nX * L * ex1 +            # Xint
        nX * ex1 * ex1 +          # XTWX
        nX * ex1 * nY * Ey +      # XTWy
        nX * ex1 * ex1 +          # Lchol
        nX * ex1 * nY * Ey +      # beta
        nX * ex1 +                # Xq
        nX * nY * Ey              # pred_flat
    ) + dbytes * (nX * nY * Ey)   # output cast/store

    per_sample_bytes = int(per_sample_bytes * 1.15)
    budget_bytes = int(budget_gb * (1024 ** 3) * 0.90)

    if per_sample_bytes <= 0:
        B = min(int(S), 1)
        return B, {
            "per_sample_bytes": int(per_sample_bytes),
            "budget_bytes": int(budget_bytes),
            "estimated_peak_bytes": int(B * max(per_sample_bytes, 0)),
        }

    B = budget_bytes // per_sample_bytes
    if B < 1:
        B = 1
    if B > int(S):
        B = int(S)
    B = int(B)
    return B, {
        "per_sample_bytes": int(per_sample_bytes),
        "budget_bytes": int(budget_bytes),
        "estimated_peak_bytes": int(B * per_sample_bytes),
    }


def auto_batch_size_simplex(X_lib, X_sample, Y_lib_s, nbrs_num_max, *, dtype, compute_dtype, budget_gb=2.0):
    num_ts_X, L, _ = X_lib.shape
    num_ts_Y, _, max_EY = Y_lib_s.shape
    S = X_sample.shape[1]
    nX, nY, Ey, K = int(num_ts_X), int(num_ts_Y), int(max_EY), int(nbrs_num_max)

    cbytes = torch.tensor([], dtype=compute_dtype).element_size()
    dbytes = torch.tensor([], dtype=dtype).element_size()
    ibytes = 8  # int64 indices

    cdist_per_sample = cbytes * (nX * L)
    knn_per_sample = (dbytes + ibytes) * (nX * K)
    core_5d_per_sample = cbytes * (3 * K * Ey * nY * nX)
    out_per_sample = dbytes * (Ey * nY * nX)
    per_sample_bytes = int(cdist_per_sample + knn_per_sample + core_5d_per_sample + out_per_sample)

    budget_bytes = int(budget_gb * (1024 ** 3) * 0.90)
    if per_sample_bytes <= 0:
        B = min(int(S), 1)
        return B, {
            "per_sample_bytes": int(per_sample_bytes),
            "budget_bytes": int(budget_bytes),
            "estimated_peak_bytes": int(B * max(per_sample_bytes, 0)),
        }

    B = budget_bytes // per_sample_bytes
    if B < 1:
        B = 1
    if B > int(S):
        B = int(S)
    B = int(B)
    return B, {
        "per_sample_bytes": int(per_sample_bytes),
        "budget_bytes": int(budget_bytes),
        "estimated_peak_bytes": int(B * per_sample_bytes),
    }


def smap_xtwx_precompute_policy(X_lib, *, compute_dtype, budget_gb=2.0):
    num_ts_X, L, max_E_X = X_lib.shape
    ex1 = int(max_E_X + 1)
    cbytes = torch.tensor([], dtype=compute_dtype).element_size()
    budget_bytes = int(budget_gb * (1024 ** 3) * 0.90)
    precompute_budget_bytes = min(budget_bytes // 8, 512 * 1024 * 1024)
    feature_bytes = int(num_ts_X) * int(L) * ex1 * ex1 * cbytes
    use_precompute = feature_bytes <= precompute_budget_bytes
    return use_precompute, {
        "feature_bytes": int(feature_bytes),
        "budget_bytes": int(precompute_budget_bytes),
        "ex1": int(ex1),
    }


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
        file=sys.stdout,
        leave=False,
        dynamic_ncols=True,
    )
