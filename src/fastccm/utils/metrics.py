# fastccm/utils/metrics.py
from __future__ import annotations
from typing import Callable, Dict, Optional
import torch

# All metrics take A,B with shape [S, D, Y, X] and return [D, Y, X]
Metric = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _double_center(D: torch.Tensor) -> torch.Tensor:
    # D: [S, S, N] where N = D*Y*X (vectorized across channels)
    mr = D.mean(dim=1, keepdim=True)            # row means [S,1,N]
    mc = D.mean(dim=0, keepdim=True)            # col means [1,S,N]
    ma = D.mean(dim=(0, 1), keepdim=True)       # grand mean [1,1,N]
    return D - mr - mc + ma


def batch_corr(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # Pearson r across sample axis, keep [D,Y,X]
    eps_t = torch.tensor(eps, dtype=A.dtype, device=A.device)
    muA = A.mean(dim=0, keepdim=True)
    muB = B.mean(dim=0, keepdim=True)
    num = ((A - muA) * (B - muB)).sum(dim=0)
    den = torch.sqrt(((A - muA).pow(2)).sum(dim=0) * ((B - muB).pow(2)).sum(dim=0) + eps_t)
    return num / den


def batch_mse(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return (A - B).pow(2).mean(dim=0)

 
def batch_mae(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Mean Absolute Error (MAE) across samples.
    A, B: [S, D, Y, X]  ->  returns [D, Y, X]
    """
    return (A - B).abs().mean(dim=0)


def batch_rmse(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    eps_t = torch.tensor(eps, dtype=A.dtype, device=A.device)
    return torch.sqrt(batch_mse(A, B) + eps_t)


def batch_neg_nrmse(A: torch.Tensor, B: torch.Tensor,
                                 T: float = 0.5, eps: float = 1e-12) -> torch.Tensor:
    """
    Multivariate version: normalize RMSE across both samples (S) and features (D)
    for each spatial location (y,x).
    A,B: [S, D, Y, X]
    Returns: [D, Y, X] (value is identical along D for each (y,x))
    """
    T_t  = torch.tensor(T,  dtype=A.dtype, device=A.device)
    eps_t = torch.tensor(eps, dtype=A.dtype, device=A.device)
    # RMSE over (S, D) per (y, x)
    mse  = (A - B).pow(2).mean(dim=(0, 1))              # [Y, X]
    rmse = torch.sqrt(mse + eps_t)                         # [Y, X]

    # Baseline: RMSE(mean over S, over D), i.e., std of B over (S, D)
    muB  = B.mean(dim=(0, 1), keepdim=True)             # [1, 1, Y, X]
    varB = (B - muB).pow(2).mean(dim=(0, 1))            # [Y, X]
    rmse_base = torch.sqrt(varB + eps_t)                  # [Y, X]

    neg_nrmse = torch.exp(- ((1.0 / T_t) * torch.pow(rmse / (rmse_base + eps_t), 2)))  # [Y, X]
    return neg_nrmse.unsqueeze(0).to(dtype=A.dtype)        # [1, Y, X]

def batch_dcor(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Multivariate distance correlation across samples using ALL features D,
    computed separately for each (y, x).

    Inputs:
        A, B: [S, D, Y, X]  (their dtype controls compute precision)
    Returns:
        dCor broadcast to [D, Y, X] (same value along D for each (y, x))
    """
    wd = A.dtype
    eps_t = torch.tensor(eps, dtype=wd, device=A.device)

    S, D, Y, X = A.shape

    # reshape to [Y*X, S, D]
    A2 = A.permute(2, 3, 0, 1).reshape(-1, S, D).to(wd)
    B2 = B.permute(2, 3, 0, 1).reshape(-1, S, D).to(wd)

    # pairwise distances
    DA = torch.cdist(A2, A2, p=2, compute_mode="use_mm_for_euclid_dist")
    DB = torch.cdist(B2, B2, p=2, compute_mode="use_mm_for_euclid_dist")

    # double-centering per block
    def _dc(M):
        mr = M.mean(dim=-1, keepdim=True)
        mc = M.mean(dim=-2, keepdim=True)
        ma = M.mean(dim=(-1, -2), keepdim=True)
        return M - mr - mc + ma

    A_dc, B_dc = _dc(DA), _dc(DB)

    # dCov / sqrt(dVarA * dVarB) per (y,x)
    dCov   = (A_dc * B_dc).mean(dim=(-1, -2))
    dVar_A = (A_dc.pow(2)).mean(dim=(-1, -2))
    dVar_B = (B_dc.pow(2)).mean(dim=(-1, -2))
    dCor   = dCov / (torch.sqrt(dVar_A * dVar_B) + eps_t)        

    # reshape back and broadcast
    dCor_yx = dCor.reshape(Y, X).to(dtype=A.dtype)
    return dCor_yx.unsqueeze(0).expand(D, Y, X)

# ---- registry ----
_METRICS: Dict[str, Metric] = {
    "corr":      batch_corr,
    "mse":       batch_mse,
    "mae":       batch_mae,
    "rmse":      batch_rmse,
    "neg_nrmse": batch_neg_nrmse,
    "dcorr":     batch_dcor,
}

def get_metric(name: str) -> Metric:
    if name not in _METRICS:
        raise ValueError(f"Unknown metric: {name}. Available: {list(_METRICS)}")
    return _METRICS[name]


def get_streaming_metric_kind(metric_fn) -> Optional[str]:
    name = getattr(metric_fn, "__name__", "")
    return {
        "batch_corr": "corr",
        "batch_mse": "mse",
        "batch_rmse": "rmse",
        "batch_mae": "mae",
        "batch_neg_nrmse": "neg_nrmse",
    }.get(name)


def stream_metric_state_init(kind: str, D, Y, X, *, device, dtype):
    shape_dyx = (D, Y, X)
    z_dyx = torch.zeros(shape_dyx, device=device, dtype=dtype)
    state = {"n": 0, "D": int(D), "dtype": dtype, "device": device}

    if kind == "corr":
        state.update({
            "sumA": z_dyx.clone(),
            "sumB": z_dyx.clone(),
            "sumAA": z_dyx.clone(),
            "sumBB": z_dyx.clone(),
            "sumAB": z_dyx.clone(),
        })
        return state
    if kind in ("mse", "rmse"):
        state["sum_sq_err"] = z_dyx
        return state
    if kind == "mae":
        state["sum_abs_err"] = z_dyx
        return state
    if kind == "neg_nrmse":
        shape_yx = (Y, X)
        z_yx = torch.zeros(shape_yx, device=device, dtype=dtype)
        state.update({
            "sum_sq_err_sd": z_yx.clone(),  # over S and D
            "sumB_sd": z_yx.clone(),        # over S and D
            "sumBB_sd": z_yx.clone(),       # over S and D
        })
        return state
    raise ValueError(f"Unsupported streaming metric kind: {kind}")


def stream_metric_state_update(kind: str, state, A_blk, B_blk, *, y_start: int = 0, count_samples: bool = True):
    if count_samples:
        state["n"] += int(A_blk.shape[0])
    y_stop = int(y_start + A_blk.shape[2])
    dyx = (slice(None), slice(int(y_start), y_stop), slice(None))
    yx = (slice(int(y_start), y_stop), slice(None))

    if kind == "corr":
        state["sumA"][dyx] += A_blk.sum(dim=0)
        state["sumB"][dyx] += B_blk.sum(dim=0)
        state["sumAA"][dyx] += (A_blk * A_blk).sum(dim=0)
        state["sumBB"][dyx] += (B_blk * B_blk).sum(dim=0)
        state["sumAB"][dyx] += (A_blk * B_blk).sum(dim=0)
        return
    if kind in ("mse", "rmse"):
        d = A_blk - B_blk
        state["sum_sq_err"][dyx] += (d * d).sum(dim=0)
        return
    if kind == "mae":
        state["sum_abs_err"][dyx] += (A_blk - B_blk).abs().sum(dim=0)
        return
    if kind == "neg_nrmse":
        d = A_blk - B_blk
        state["sum_sq_err_sd"][yx] += (d * d).sum(dim=(0, 1))
        state["sumB_sd"][yx] += B_blk.sum(dim=(0, 1))
        state["sumBB_sd"][yx] += (B_blk * B_blk).sum(dim=(0, 1))
        return
    raise ValueError(f"Unsupported streaming metric kind: {kind}")


def stream_metric_state_finalize(kind: str, state, *, eps=1e-12, neg_nrmse_T=0.5):
    n = max(int(state["n"]), 1)
    D = max(int(state["D"]), 1)
    device = state["device"]
    dtype = state["dtype"]
    n_t = torch.tensor(float(n), device=device, dtype=dtype)
    eps_t = torch.tensor(eps, device=device, dtype=dtype)

    if kind == "corr":
        num = state["sumAB"] - (state["sumA"] * state["sumB"] / n_t)
        denA = state["sumAA"] - (state["sumA"] * state["sumA"] / n_t)
        denB = state["sumBB"] - (state["sumB"] * state["sumB"] / n_t)
        return num / torch.sqrt(denA * denB + eps_t)
    if kind == "mse":
        return state["sum_sq_err"] / n_t
    if kind == "rmse":
        return torch.sqrt((state["sum_sq_err"] / n_t) + eps_t)
    if kind == "mae":
        return state["sum_abs_err"] / n_t
    if kind == "neg_nrmse":
        cnt_t = torch.tensor(float(n * D), device=device, dtype=dtype)
        mse = state["sum_sq_err_sd"] / cnt_t
        rmse = torch.sqrt(mse + eps_t)
        muB = state["sumB_sd"] / cnt_t
        varB = (state["sumBB_sd"] / cnt_t) - (muB * muB)
        rmse_base = torch.sqrt(varB.clamp_min(0.0) + eps_t)
        T_t = torch.tensor(neg_nrmse_T, device=device, dtype=dtype)
        out = torch.exp(-((1.0 / T_t) * torch.pow(rmse / (rmse_base + eps_t), 2)))
        return out.unsqueeze(0).to(dtype=dtype)
    raise ValueError(f"Unsupported streaming metric kind: {kind}")
