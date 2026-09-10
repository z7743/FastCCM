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


# `base_bytes` is the resident inputs and the copies taken of them, so no batch
# size reclaims any of it: only the per-batch scratch is actually being sized.
# Subtracting an over-budget base leaves a negative allowance, and reading that
# as "batch of one" collapses throughput by orders of magnitude while saving
# nothing -- the base is allocated either way. Size scratch against a floor
# fraction of the budget instead, so an unreachable budget overshoots by a
# bounded amount rather than serialising the run.
BATCH_SCRATCH_FLOOR_FRACTION = 0.25


def _scratch_allowance(budget_bytes: int, base_bytes: int):
    """
    Bytes the batching policy may spend on per-batch scratch.

    Returns `(available_bytes, base_over_budget)`. `available_bytes` is always
    positive: once the base alone clears the budget the floor fraction applies
    and the flag says the estimate has been exceeded before batching begins.

    The allowance therefore jumps at the crossing rather than shrinking to zero.
    That is deliberate: a budget the base still fits inside is honoured exactly,
    and only a budget that cannot be met is traded for throughput.
    """
    available_bytes = int(budget_bytes) - int(base_bytes)
    if available_bytes > 0:
        return available_bytes, False
    return max(1, int(int(budget_bytes) * BATCH_SCRATCH_FLOOR_FRACTION)), True


def _resolve_batch_size(total_samples: int, budget_bytes: int, base_bytes: int, per_sample_bytes: int):
    if total_samples <= 0:
        return 0
    if per_sample_bytes <= 0:
        return int(total_samples)

    available_bytes, _ = _scratch_allowance(budget_bytes, base_bytes)

    batch_size = available_bytes // int(per_sample_bytes)
    batch_size = max(1, int(batch_size))
    batch_size = min(batch_size, int(total_samples))
    return int(batch_size)


def _simplex_base_bytes(
    *,
    num_ts_X: int,
    library_size: int,
    max_E_X: int,
    total_samples: int,
    num_ts_Y: int,
    max_E_Y: int,
    dtype,
    compute_dtype,
    extra_base_bytes: int = 0,
) -> int:
    dbytes = _dtype_bytes(dtype)
    cbytes = _dtype_bytes(compute_dtype)
    nX = int(num_ts_X)
    L = int(library_size)
    Ex = int(max_E_X)
    S = int(total_samples)
    nY = int(num_ts_Y)
    Ey = int(max_E_Y)
    return int(
        dbytes * (
            nX * L * Ex +      # X_lib sampled library embeddings
            nX * S * Ex +      # X_sample full query embeddings
            nY * L * Ey        # Y_lib_s original targets
        )
        + cbytes * (L * nY * Ey)  # Y_lib_lne compute copy
        + int(extra_base_bytes)
    )


# Deterministic simplex target auto-split constants calibrated from offline CPU
# sweeps. The goal is to keep the gathered target tile near a stable working-set
# size while falling back to a conservative `ny` chunk when search dominates.
# Matrix-shaped CPU sweeps favor smaller tiles once reduction and streaming
# metrics dominate neighbor search. A roughly 36 MiB gathered tile selects 16
# targets for 100x100 work and 8 targets once the source axis reaches 512.
SIMPLEX_SEARCH_SAFETY_FACTOR = 1.50
SIMPLEX_REDUCE_SAFETY_FACTOR = 1.10
SIMPLEX_CALIBRATED_TARGET_TILE_BYTES = 36 * 1024 * 1024
SIMPLEX_CALIBRATED_TARGET_BATCH_MIN = 16
SIMPLEX_CALIBRATED_LARGE_SOURCE_TARGET_BATCH_MIN = 8
SIMPLEX_CALIBRATED_LARGE_SOURCE_THRESHOLD = 512
SIMPLEX_CALIBRATED_TARGET_BATCH_MAX = 64
SIMPLEX_CALIBRATED_SEARCH_DOMINANT_RATIO = 8.0
SIMPLEX_CALIBRATED_SEARCH_DOMINANT_TARGET_BATCH = 32


def resolve_simplex_target_batch_size(
    num_targets: int,
    target_batch_size,
) -> int:
    nY = int(num_targets)
    if nY <= 0:
        return 0
    if target_batch_size is None:
        return nY
    if isinstance(target_batch_size, str):
        if target_batch_size == "auto":
            return nY
        raise ValueError("target_batch_size must be a positive int, None, or 'auto'.")

    y_batch = min(nY, int(target_batch_size))
    if y_batch <= 0:
        raise ValueError("target_batch_size must be positive, None, or 'auto'.")
    return int(y_batch)


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


def _calibrated_simplex_target_batch_size(
    *,
    num_ts_X: int,
    num_ts_Y: int,
    total_samples: int,
    library_size: int,
    max_EY: int,
    nbrs_num_max: int,
    dtype,
    compute_dtype,
    budget_bytes: int,
    max_E_X: int,
    extra_base_bytes: int = 0,
) -> int:
    nX = int(num_ts_X)
    nY = int(num_ts_Y)
    S = int(total_samples)
    Ey = int(max_EY)
    K = int(nbrs_num_max)
    L = int(library_size)

    target_batch_min = (
        SIMPLEX_CALIBRATED_LARGE_SOURCE_TARGET_BATCH_MIN
        if nX >= SIMPLEX_CALIBRATED_LARGE_SOURCE_THRESHOLD
        else SIMPLEX_CALIBRATED_TARGET_BATCH_MIN
    )
    if nY <= target_batch_min:
        return nY

    cbytes = _dtype_bytes(compute_dtype)
    dbytes = _dtype_bytes(dtype)
    ibytes = 8
    base_bytes = _simplex_base_bytes(
        num_ts_X=nX,
        library_size=L,
        max_E_X=max_E_X,
        total_samples=S,
        num_ts_Y=nY,
        max_E_Y=Ey,
        dtype=dtype,
        compute_dtype=compute_dtype,
        extra_base_bytes=extra_base_bytes,
    )

    search_per_sample = (
        cbytes * (nX * L + nX * K) +
        (dbytes + ibytes) * (nX * K)
    )
    reduce_per_sample_full = (
        cbytes * (nX * K * nY * Ey + nX * nY * Ey) +
        dbytes * (nX * nY * Ey)
    )
    dominance = float(search_per_sample) / float(max(reduce_per_sample_full, 1))
    if dominance >= SIMPLEX_CALIBRATED_SEARCH_DOMINANT_RATIO:
        return min(nY, int(SIMPLEX_CALIBRATED_SEARCH_DOMINANT_TARGET_BATCH))

    available_bytes, _ = _scratch_allowance(budget_bytes, base_bytes)

    # Use the search-dominated batch as the first fixed-point iterate, then set
    # `ny` so the gathered target tile stays near the calibrated working set.
    sample_batch_search = max(1, min(S, available_bytes // max(int(search_per_sample), 1)))
    tile_unit_bytes = max(cbytes * nX * K * Ey * sample_batch_search, 1)
    target_by = float(SIMPLEX_CALIBRATED_TARGET_TILE_BYTES) / float(tile_unit_bytes)
    return _round_pow2_clamped(
        target_by,
        min_value=min(target_batch_min, nY),
        max_value=min(SIMPLEX_CALIBRATED_TARGET_BATCH_MAX, nY),
    )


def _simplex_per_sample_bytes(search_per_sample: int, reduce_per_sample: int) -> int:
    search_bytes = float(search_per_sample) * float(SIMPLEX_SEARCH_SAFETY_FACTOR)
    reduce_bytes = float(reduce_per_sample) * float(SIMPLEX_REDUCE_SAFETY_FACTOR)
    return int(max(search_bytes, reduce_bytes))


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
        "base_over_budget": bool(base_bytes >= budget_bytes),
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
    target_batch_size=None,
    extra_base_bytes: int = 0,
):
    num_ts_X, L, max_E_X = X_lib.shape
    num_ts_Y, _, max_EY = Y_lib_s.shape
    S = X_sample.shape[1]
    nX, nY, Ey, K = int(num_ts_X), int(num_ts_Y), int(max_EY), int(nbrs_num_max)
    device_type = X_lib.device.type

    cbytes = _dtype_bytes(compute_dtype)
    dbytes = _dtype_bytes(dtype)
    ibytes = 8  # int64 indices
    budget_bytes = int(budget_gb * (1024 ** 3) * 0.90)

    base_bytes = _simplex_base_bytes(
        num_ts_X=nX,
        library_size=L,
        max_E_X=max_E_X,
        total_samples=S,
        num_ts_Y=nY,
        max_E_Y=Ey,
        dtype=dtype,
        compute_dtype=compute_dtype,
        extra_base_bytes=extra_base_bytes,
    )
    if target_batch_size == "auto":
        if device_type == "cuda":
            # The CPU-calibrated target tiling policy is too conservative on CUDA:
            # for medium and even fairly large symmetric matrices, repeatedly
            # splitting the target axis pays a much larger kernel-launch and gather
            # overhead than the saved working-set memory. Prefer a single target
            # block on GPU and let the outer query batch size absorb memory pressure.
            y_batch = nY
        else:
            y_batch = _calibrated_simplex_target_batch_size(
                num_ts_X=nX,
                num_ts_Y=nY,
                total_samples=S,
                library_size=L,
                max_EY=Ey,
                nbrs_num_max=K,
                dtype=dtype,
                compute_dtype=compute_dtype,
                budget_bytes=budget_bytes,
                max_E_X=max_E_X,
                extra_base_bytes=extra_base_bytes,
            )
    else:
        y_batch = resolve_simplex_target_batch_size(nY, target_batch_size)

    search_per_sample = (
        cbytes * (nX * L + nX * K) +
        (dbytes + ibytes) * (nX * K)
    )
    reduce_per_sample = (
        cbytes * (nX * K * y_batch * Ey + nX * y_batch * Ey) +
        dbytes * (nX * y_batch * Ey)
    )
    per_sample_bytes = _simplex_per_sample_bytes(search_per_sample, reduce_per_sample)

    B = _resolve_batch_size(int(S), budget_bytes, base_bytes, per_sample_bytes)
    return B, {
        "base_bytes": int(base_bytes),
        "per_sample_bytes": int(per_sample_bytes),
        "budget_bytes": int(budget_bytes),
        "estimated_peak_bytes": int(base_bytes + B * max(per_sample_bytes, 0)),
        "target_batch_size": int(y_batch),
        "base_over_budget": bool(base_bytes >= budget_bytes),
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
