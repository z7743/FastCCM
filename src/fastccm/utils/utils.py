# utils/utils.py
import torch
import numpy as np
from sklearn.decomposition import PCA

def embed(ts, E, tau):
    """
    Embed a time series into a delay embedding space.
    (n_samples, n_variables) -> (n_variables, n_samples, E)
    (n_samples, ) -> (1, n_samples, E)

    Args:
        ts (list or np.ndarray): Shape (n_samples,) or (n_samples, n_variables).
        E (int): Embedding dimension.
        tau (int): Time delay (stride).

    Returns:
        np.ndarray: Shape (n_variables, n_embedded_samples, E).
    """
    if not isinstance(ts, (list, np.ndarray)):
        raise TypeError("ts must be a list or numpy.ndarray with shape (n_samples,) or (n_samples, n_variables).")

    x = np.asarray(ts)

    # Normalize to 2D: (n_samples, n_variables)
    if x.ndim == 1:
        x = x[:, None]
    elif x.ndim != 2:
        raise ValueError("ts must have shape (n_samples,) or (n_samples, n_variables).")

    # Basic argument checks
    if not (isinstance(E, (int, np.integer)) and E >= 1):
        raise ValueError("E must be an integer >= 1.")
    if not (isinstance(tau, (int, np.integer)) and tau >= 1):
        raise ValueError("tau must be an integer >= 1.")

    td = get_td_embedding_np(x, E, tau)

    return td.transpose((2, 0, 1))

def get_td_embeddings(ts, opt_E, opt_tau):
    ts_num = len(ts)
    tdembs = []
    for i in range(ts_num):
        tdembs += [get_td_embedding_np(ts[i][:,None],opt_E[i],opt_tau[i])[:,:,0]]
    return tdembs

def get_td_embedding_torch(ts, dim, stride, return_pred=False, tp=0):
    tdemb = ts.unfold(0,(dim-1) * stride + 1,1)[...,::stride]
    tdemb = torch.swapaxes(tdemb,-1,-2)
    if return_pred:
        return tdemb[:tdemb.shape[0]-tp], ts[(dim-1) * stride + tp:]
    else:
        return tdemb
    
def get_td_embedding_np(time_series, dim, stride, return_pred=False, tp=0):
    num_points, num_dims = time_series.shape
    # Calculate the size of the unfolding window
    window_size = (dim - 1) * stride + 1
    
    # Ensure the time series is long enough for the unfolding
    if num_points < window_size:
        raise ValueError("Time series is too short for the given dimensions and stride.")
    
    # Create an array to hold the unfolded data
    # Calculate shape and strides for as_strided
    shape = (num_points - window_size + 1, dim, num_dims)
    strides = (time_series.strides[0], stride * time_series.strides[0], time_series.strides[1])
    tdemb = np.lib.stride_tricks.as_strided(time_series, shape=shape, strides=strides)
    
    if return_pred:
        # Return the embedded data excluding the last 'tp' points, and the prediction points starting from a specific index
        return tdemb[:tdemb.shape[0]-tp], time_series[(dim - 1) * stride + tp:]
    else:
        return tdemb
    

def get_td_embedding_specified(time_series, delays):
    """
    [ChatGPT written]
    Embeds a time series using specified time delays by truncating the beginning of the series.

    Parameters:
    time_series (array-like): The input time series data.
    delays (array-like): An array of time delays.

    Returns:
    np.ndarray: A matrix where each column is the time series delayed by the respective delay.
    """
    n = len(time_series)
    max_delay = max(delays)
    
    # Ensure the time series is long enough for the maximum delay
    if max_delay >= n:
        raise ValueError("Maximum delay exceeds the length of the time series.")

    # Create an embedding matrix with shape (n - max_delay, len(delays))
    embedded = np.empty((n - max_delay, len(delays)))
    
    for i, delay in enumerate(delays):
        embedded[:, i] = time_series[max_delay - delay:n - delay]

    return embedded


def calculate_correlation_dimension(embedded_data, radii=None):
    """
    Grassberger–Procaccia correlation dimension (standard):
      - counts unique unordered pairs (i<j)
      - excludes the diagonal (no self-pairs)

    Parameters
    ----------
    embedded_data : array-like or torch.Tensor, shape (N, M)
        Point cloud (e.g., time-delay embedded series).
    radii : torch.Tensor of shape (R,), optional
        Radii (distance units). If None, a robust log-spaced grid is chosen from pairwise-distance quantiles.

    Returns
    -------
    float
        Estimated slope of log C(r) vs log r (correlation dimension).
    """
    # --- setup ---
    X0 = torch.as_tensor(embedded_data)
    device = X0.device
    X = X0.to(device=device, dtype=torch.float32,
              copy=(X0.device != device or X0.dtype != torch.float32)).contiguous()

    if X.ndim != 2:
        raise ValueError("embedded_data must have shape (N, M).")
    N, M = X.shape
    if N < 3:
        raise ValueError("Need at least 3 points.")

    # internal chunk sizes 
    ROW_BS = 2048
    FEAT_BS = 1024

    # --- helpers: compute distances in feature chunks (no (N,N,M) tensors) ---
    def row_sqnorms(A: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(A.shape[0], dtype=torch.float64, device=device)
        for k in range(0, A.shape[1], FEAT_BS):
            Ak = A[:, k:k+FEAT_BS].to(torch.float64)
            out += (Ak * Ak).sum(dim=1)
        return out  # (a,)

    def pairwise_sqdist(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A2 = row_sqnorms(A)[:, None]          # (a,1)
        B2 = row_sqnorms(B)[None, :]          # (1,b)
        D2 = A2 + B2
        for k in range(0, A.shape[1], FEAT_BS):
            Ak = A[:, k:k+FEAT_BS].to(torch.float64)
            Bk = B[:, k:k+FEAT_BS].to(torch.float64)
            D2 -= 2.0 * (Ak @ Bk.T)           # (a,b)
        D2.clamp_(min=0)
        return D2  # (a,b) float64

    # --- choose radii if not provided: sample unordered pair distances for quantiles ---
    if radii is None:
        s = min(4096, N)
        idx = torch.randperm(N, device=device)[:s]
        S = X[idx]

        sampled = []
        total = 0
        target = min(2_000_000, (s * (s - 1)) // 2)
        inner = min(ROW_BS, s)

        for i in range(0, s, inner):
            Ai = S[i:i+inner]
            for j in range(i, s, inner):
                Bj = S[j:j+inner]
                D2 = pairwise_sqdist(Ai, Bj)
                if i == j:
                    # keep upper triangle only (exclude diagonal)
                    a = D2.shape[0]
                    if a > 1:
                        tri = torch.triu_indices(a, a, offset=1, device=device)
                        d = D2[tri[0], tri[1]]
                    else:
                        d = D2.new_empty((0,), dtype=D2.dtype)
                else:
                    d = D2.flatten()
                sampled.append(d)
                total += d.numel()
                if total >= target:
                    break
            if total >= target:
                break

        samp = torch.cat(sampled) if sampled else torch.tensor([], device=device, dtype=torch.float64)
        if samp.numel() == 0:
            # fallback: rough scale from norms
            norms = row_sqnorms(S).sqrt()
            med = norms.median() if norms.numel() else torch.tensor(1.0, device=device, dtype=torch.float64)
            q1, q2 = med * 1e-3, med * 1e+3
        else:
            d = samp.sqrt()
            q1, q2 = torch.quantile(d, 0.001), torch.quantile(d, 0.999)
            if not torch.isfinite(q1) or not torch.isfinite(q2) or q1 <= 0 or q2 <= q1:
                med = torch.median(d)
                q1, q2 = med * 1e-3, med * 1e+3

        a = torch.log2(q1) - 2.0
        b = torch.log2(q2) + 2.0
        radii = torch.logspace(a.item(), b.item(), steps=100, base=2.0, device=device, dtype=torch.float32)

    radii = radii.to(device=device, dtype=torch.float32)
    R = radii.numel()
    radii2 = (radii.to(torch.float64) ** 2)

    # --- count unordered pairs (i<j), exclude diagonal ---
    counts = torch.zeros(R, dtype=torch.int64, device=device)
    for i in range(0, N, ROW_BS):
        i_end = min(i + ROW_BS, N)
        Ai = X[i:i_end]
        for j in range(i, N, ROW_BS):
            j_end = min(j + ROW_BS, N)
            Bj = X[j:j_end]
            D2 = pairwise_sqdist(Ai, Bj)

            if i == j:
                # exclude self-pairs, later halve to keep i<j once
                n = min(D2.shape[0], D2.shape[1])
                if n > 0:
                    diag = torch.arange(n, device=device)
                    D2[diag, diag] = float('inf')

            # loop radii to avoid allocating (a,b,R)
            for r_idx in range(R):
                c = (D2 < radii2[r_idx]).sum(dtype=torch.int64)
                if i == j:
                    c = c // 2
                counts[r_idx] += c

    total_pairs = (N * (N - 1)) // 2
    if total_pairs == 0:
        raise RuntimeError("No eligible pairs.")

    C_r = counts.to(torch.float64) / float(total_pairs)

    # --- fit slope on log-log ---
    valid = C_r > 0
    if valid.sum().item() < 2:
        raise RuntimeError("Not enough valid correlation-sum points for a fit.")

    log_r = torch.log(radii[valid].to(torch.float64))
    log_C = torch.log(C_r[valid])

    # Prefer a symmetric interior band; fall back to 20–80% in log_r
    thr = (log_C.max() - log_C.min()) / 20.0
    mask = (log_C > (log_C.min() + thr)) & (log_C < (log_C.max() - thr))
    if mask.sum().item() < 5:
        order = torch.argsort(log_r)
        k1 = int(0.20 * order.numel())
        k2 = int(0.80 * order.numel())
        sel = order[k1:k2]
        log_r_fit, log_C_fit = log_r[sel], log_C[sel]
    else:
        log_r_fit, log_C_fit = log_r[mask], log_C[mask]

    Xmat = torch.stack([log_r_fit, torch.ones_like(log_r_fit)], dim=1)
    beta = torch.linalg.pinv(Xmat) @ log_C_fit.unsqueeze(1)
    return float(beta[0].item())

def calculate_rank_for_variance(data_matrix, variance_threshold=0.95):
    # Perform PCA
    pca = PCA()
    pca.fit(data_matrix)
    
    # Calculate the cumulative explained variance ratio
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    
    # Find the number of components that explain at least the desired variance
    rank = np.searchsorted(cumulative_variance_ratio, variance_threshold) + 1
    
    return rank

