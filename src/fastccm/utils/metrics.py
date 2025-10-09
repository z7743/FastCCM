# fastccm/utils/metrics.py
from __future__ import annotations
from typing import Callable, Dict
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