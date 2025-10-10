# ccm.py
import gc
import warnings
import os
import torch
from .utils.metrics import get_metric

#torch.set_num_threads(os.cpu_count())  
#torch.set_num_interop_threads(1)

def _resolve_dtype(x):
    if isinstance(x, torch.dtype):
        return x
    if x is None:
        return None
    key = str(x).strip().lower()
    key = {
        "float": "float32", "double": "float64", "half": "float16",
        "bf16": "bfloat16", "f16": "float16", "f32": "float32", "f64": "float64",
        "fp16": "float16", "fp32": "float32", "fp64": "float64",
    }.get(key, key)
    dt = getattr(torch, key, None)
    if isinstance(dt, torch.dtype):
        return dt
    raise ValueError(f"Unknown dtype: {x!r}")

class PairwiseCCM:
    
    def __init__(self,device = "cpu", dtype="float32", compute_dtype=None):
        """
        Create a PairwiseCCM instance.

        Parameters:
            device (str): The computation device ('cpu' or 'cuda') to use for all calculations.
            dtype (torch.dtype or str): Numeric dtype for internal tensors
                ('float32' | 'float16' | 'bfloat16' | 'float64' or torch.dtype). Default: float32.
            compute_dtype (torch.dtype or str or None): math-ops dtype. If None, uses dtype. 
        """
        self.device = device
        self.dtype = _resolve_dtype(dtype) or torch.float32
        self.compute_dtype = _resolve_dtype(compute_dtype) or self.dtype

        # (Optional) sanity: ensure float type
        if not (self.dtype.is_floating_point and self.compute_dtype.is_floating_point):
            raise ValueError("dtype and compute_dtype must be floating dtypes.")

    def compute(self, *args, **kwargs):
        warnings.warn("PairwiseCCM.compute → PairwiseCCM.score_matrix", DeprecationWarning)
        return self.score_matrix(*args, **kwargs)
    
    def predict(self, *args, **kwargs):
        warnings.warn("PairwiseCCM.predict → PairwiseCCM.predict_matrix", DeprecationWarning)
        return self.predict_matrix(*args, **kwargs)

    def score_matrix(
            self, 
            X_emb, 
            Y_emb = None, 
            library_size = None, 
            sample_size = None, 
            exclusion_window = 0, 
            tp = 0, 
            method = "simplex", 
            seed = None, 
            metric = "corr",
            batch_size=2048,
            **kwargs
    ):
        """
        Compute the pairwise CCM matrix between lists of delay-embedded time series.

        Parameters
        ----------
        X_emb : list[np.ndarray]
            Source embeddings. Each entry is a 2D array of shape (T, E_x) where
            T is time and E_x is the embedding dimension for that series.
        Y_emb : list[np.ndarray] or None, optional
            Target embeddings. Same structure as X_emb. If None, uses X_emb.
        library_size : int or "auto" or None, optional
            Size of the library (number of candidate neighbor points) drawn from each
            series. If None, uses the maximum common length across series. If "auto",
            uses min(max_common_len // 2, 700).
        sample_size : int or "auto" or None, optional
            Number of query/evaluation points where predictions are scored.
            If None, uses the maximum common length across series. If "auto",
            uses min(max_common_len // 6, 250).
        exclusion_window : int or None, optional
            Temporal exclusion radius in samples. When an int r is provided,
            neighbors with |t_neighbor − t_query| ≤ r are excluded (self excluded when r=0).
            When None, no temporal exclusion is applied (self can be included).
        tp : int, optional
            Prediction horizon (lead). Predicts Y[t + tp] from X[t]. Default: 0.
        method : {"simplex", "smap"}, optional
            Local regressor: k-NN weighted average ("simplex") or locally weighted
            linear ("smap").
        metric : {"corr","mse","mae","rmse","neg_nrmse","dcorr"} or Callable, optional
            Scoring function applied to (prediction, target). If a string, one of the
            built-ins. If a callable, it must accept (A, B) with shapes
            [S, E_y, n_Y, n_X] and return [E_y, n_Y, n_X].
            Default: "corr".

        Other Parameters
        ----------------
        nbrs_num : int, optional
            Number of neighbors for "simplex". Default is E_x + 1 per source series.
        theta : float, optional
            Local weighting strength for "smap". Default: 1.0.

        Returns
        -------
        np.ndarray
            Array of Pearson correlations with shape (E_y, n_Y, n_X), where E_y is
            the target embedding dimension for each Y series, n_Y is the number of
            target series, and n_X is the number of source series.

        Raises
        ------
        ValueError
            If `method` is not one of {"simplex", "smap"} or required options are invalid.
        """
        if Y_emb is None:
            Y_emb = X_emb

        r_AB = self.__ccm_core(
            mode="score",
            X_lib_list=X_emb,
            Y_lib_list=Y_emb,
            X_sample_list=X_emb,
            library_size=library_size,
            sample_size=sample_size,
            exclusion_window=exclusion_window,
            tp=tp,
            method=method,
            seed=seed,
            metric=metric,
            batch_size=batch_size,
            **kwargs
        )

        r_AB = r_AB.to("cpu").numpy()
        self._soft_clear()
        return r_AB

    def predict_matrix(
            self, 
            X_lib_emb, 
            Y_lib_emb = None, 
            X_pred_emb = None, 
            library_size = None, 
            exclusion_window = 0, 
            tp = 0, 
            method = "simplex", 
            seed = None,
            metric = "corr",
            batch_size=2048,
            **kwargs
    ):
        """
        Predict target embeddings at given query points using a CCM model fit on a library.

        Parameters
        ----------
        X_lib_emb : list[np.ndarray]
            Source library embeddings. Each entry is (T_lib, E_x).
        Y_lib_emb : list[np.ndarray] or None, optional
            Target library embeddings. Same structure as X_lib_emb. If None, uses X_lib_emb.
        X_pred_emb : list[np.ndarray] or None, optional
            Source query embeddings at which predictions are evaluated. Each entry is
            (T_pred, E_x). If None, uses X_lib_emb.
        library_size : int or "auto" or None, optional
            Number of library points drawn from each series. If None, uses the maximum
            common library length. If "auto", uses min(max_lib_len // 2, 700).
        exclusion_window : int or None, optional
            Temporal exclusion radius in samples. When an int r is provided,
            neighbors with |t_neighbor − t_query| ≤ r are excluded (self excluded when r=0).
            When None, no temporal exclusion is applied (self can be included).
        tp : int, optional
            Prediction horizon (lead). Predicts Y[t + tp] from X[t]. Default: 0.
        method : {"simplex", "smap"}, optional
            Local regressor: "simplex" or "smap".
        metric : {"corr","mse","mae","rmse","neg_nrmse","dcorr"} or Callable, optional
            Scoring function applied to (prediction, target). If a string, one of the
            built-ins. If a callable, it must accept (A, B) with shapes
            [S, E_y, n_Y, n_X] and return [E_y, n_Y, n_X].
            Default: "corr".

        Other Parameters
        ----------------
        nbrs_num : int, optional
            Number of neighbors for "simplex". Default is E_x + 1 per source series.
        theta : float, optional
            Local weighting strength for "smap". Default: 1.0.

        Returns
        -------
        np.ndarray
            Predicted target embeddings with shape (T_pred, E_y, n_Y, n_X), where
            T_pred is the number of query points, E_y is the target embedding
            dimension, n_Y is number of targets, and n_X is number of sources.

        Raises
        ------
        ValueError
            If `method` is not one of {"simplex", "smap"} or required options are invalid.
        """
        if Y_lib_emb is None:
            Y_lib_emb = X_lib_emb
        if X_pred_emb is None:
            X_pred_emb = X_lib_emb

        A = self.__ccm_core(
            mode="predict",
            X_lib_list=X_lib_emb,
            Y_lib_list=Y_lib_emb,
            X_sample_list=X_pred_emb,
            library_size=library_size,
            sample_size=None, 
            exclusion_window=exclusion_window,
            tp=tp,
            method=method,
            seed=seed,
            metric=metric,
            batch_size=batch_size,
            **kwargs
        )

        A = A.to("cpu").numpy()
        self._soft_clear()
        return A

    @torch.inference_mode()
    def __ccm_core(
        self,
        mode,                      # "score" or "predict"
        X_lib_list,                # list[np.ndarray]
        Y_lib_list,                # list[np.ndarray]
        X_sample_list=None,        # list[np.ndarray] | None (required for "predict")
        library_size=None,
        sample_size=None,
        exclusion_window=0,
        tp=0,
        method="simplex",
        seed=None,
        metric="corr",
        batch_size=None,
        **kwargs
    ):
        metric_fn = get_metric(metric)
        # ---------- 1) dims / lengths ----------
        num_ts_X = len(X_lib_list)
        num_ts_Y = len(Y_lib_list)

        max_E_X = torch.tensor([X_lib_list[i].shape[-1] for i in range(num_ts_X)], device=self.device).max().item()
        max_E_Y = torch.tensor([Y_lib_list[i].shape[-1] for i in range(num_ts_Y)], device=self.device).max().item()

        if mode == "score":
            min_len = torch.tensor(
                [Y_lib_list[i].shape[0] for i in range(num_ts_Y)] +
                [X_lib_list[i].shape[0] for i in range(num_ts_X)],
                device=self.device
            ).min().item()
            if min_len - tp <= 0:
                raise ValueError("Not enough points after applying tp.")
        else:
            if X_sample_list is None:
                raise ValueError("X_sample_list is required in 'predict' mode.")
            min_len_lib = torch.tensor(
                [Y_lib_list[i].shape[0] for i in range(num_ts_Y)] +
                [X_lib_list[i].shape[0] for i in range(num_ts_X)],
                device=self.device
            ).min().item()
            min_len_pred = torch.tensor([X_sample_list[i].shape[0] for i in range(num_ts_X)], device=self.device).min().item()
            if min_len_lib - tp <= 0 or min_len_pred <= 0:
                raise ValueError("Not enough points for library or prediction.")

        # ---------- 2) method params ----------
        if method == "simplex":
            if "nbrs_num" in kwargs:
                nbrs_num = kwargs["nbrs_num"]
                nbrs_num = torch.tensor([nbrs_num] * num_ts_X, device=self.device) if isinstance(nbrs_num, int) \
                        else torch.tensor(nbrs_num, device=self.device)
            else:
                nbrs_num = torch.tensor([X_lib_list[i].shape[-1] + 1 for i in range(num_ts_X)], device=self.device)
        elif method == "smap":
            ridge = kwargs.get("ridge", 0.0)
            theta = kwargs.get("theta", 1.0)
        else:
            raise ValueError("Invalid method. Supported methods are 'simplex' and 'smap'.")

        # ---------- 3) size resolution ----------
        if mode == "score":
            # Defaults/auto computed from min_len (not min_len - tp)
            if library_size is None:
                library_size_res = min_len
            elif library_size == "auto":
                library_size_res = min(min_len // 2, 700)
            else:
                library_size_res = int(library_size)

            if sample_size is None:
                sample_size_res = min_len
            elif sample_size == "auto":
                sample_size_res = min(min_len // 6, 250)
            else:
                sample_size_res = int(sample_size)
        else:  # predict
            if library_size is None:
                library_size_res = min_len_lib
            elif library_size == "auto":
                library_size_res = min(min_len_lib // 2, 700)
            else:
                library_size_res = int(library_size)

        # ---------- 4) indices ----------
        gen_lib = gen_smpl = None
        if seed is not None:
            base = int(seed)
            gen_lib  = torch.Generator(device=self.device).manual_seed(base)
            gen_smpl = torch.Generator(device=self.device).manual_seed(base + 1)
            
        if mode == "score":
            # Indices are still drawn from the valid (min_len - tp) window, like your original
            lib_indices  = self.__get_random_indices(min_len - tp, library_size_res, gen_lib)
            smpl_indices = self.__get_random_indices(min_len - tp, sample_size_res, gen_smpl)
        else:
            lib_indices  = self.__get_random_indices(min_len_lib - tp, library_size_res, gen_lib)
            smpl_indices = torch.arange(min_len_pred, device=self.device)  # same as original

        # ---------- 5) sampling ----------
        if mode == "score":
            X_lib    = self.__get_random_sample(X_lib_list, min_len, lib_indices,  num_ts_X, max_E_X)
            X_sample = self.__get_random_sample(X_lib_list, min_len, smpl_indices, num_ts_X, max_E_X)
            Y_lib_s  = self.__get_random_sample(Y_lib_list, min_len, lib_indices + tp,  num_ts_Y, max_E_Y)
            Y_smp_s  = self.__get_random_sample(Y_lib_list, min_len, smpl_indices + tp, num_ts_Y, max_E_Y)
        else:
            X_lib    = self.__get_random_sample(X_lib_list,   min_len_lib,  lib_indices,      num_ts_X, max_E_X)
            X_sample = self.__get_random_sample(X_sample_list,min_len_pred, smpl_indices,     num_ts_X, max_E_X)
            Y_lib_s  = self.__get_random_sample(Y_lib_list,   min_len_lib,  lib_indices + tp, num_ts_Y, max_E_Y)
            Y_smp_s  = None

        
        # ---------- 6) method call ----------
        if method == "simplex":
            nbrs_num_max = nbrs_num.max().item()
            if batch_size == "auto":
                batch_size = self.__auto_batch_size_simplex(
                    X_lib, X_sample, Y_lib_s, nbrs_num_max, budget_gb=8.0
                )
                #print("Using batch_size =", batch_size)
            if batch_size is not None and batch_size <= 0:
                raise ValueError("batch_size must be positive, 'auto', or None.")

            out = self.__simplex_prediction(
                lib_indices, smpl_indices,
                X_lib, X_sample, Y_lib_s, Y_smp_s,
                exclusion_window, nbrs_num, metric_fn=metric_fn,
                return_pred=(Y_smp_s is None), sample_batch_size=batch_size,
            )


        else:
            if batch_size == "auto":
                batch_size = self.__auto_batch_size_smap(X_lib, X_sample, Y_lib_s, budget_gb=8.0)
                #print("Using batch_size =", batch_size)
            if batch_size is not None and batch_size <= 0:
                raise ValueError("batch_size must be positive, 'auto', or None.")


            out = self.__smap_prediction(
                lib_indices, smpl_indices,
                X_lib, X_sample, Y_lib_s, Y_smp_s,
                exclusion_window, theta, metric_fn=metric_fn,
                return_pred=(Y_smp_s is None),
                sample_batch_size=batch_size,
                ridge=ridge
            )


        return out

    @torch.inference_mode()
    def __simplex_prediction(self, lib_indices, smpl_indices,
                              X_lib, X_sample, Y_lib_shifted, Y_sample_shifted, 
                              exclusion_rad, nbrs_num, metric_fn, return_pred=False, sample_batch_size=None):
        num_ts_X = X_lib.shape[0]
        num_ts_Y = Y_lib_shifted.shape[0]
        max_E_Y = Y_lib_shifted.shape[2]
        subsample_size = X_sample.shape[1]

        if (sample_batch_size is None) or (sample_batch_size >= subsample_size):
            sample_batch_size = subsample_size

        nbrs_num_max = nbrs_num.max().item()
        #nbrs_mask = (torch.arange(nbrs_num_max).unsqueeze(0) < nbrs_num.unsqueeze(1))
        A = torch.empty((subsample_size, max_E_Y, num_ts_Y, num_ts_X), device=self.device, dtype=self.dtype)
        for s0 in range(0, subsample_size, sample_batch_size):
            s1 = min(subsample_size, s0 + sample_batch_size)

            try:
                weights, indices = self.__get_nbrs_indices_with_weights(
                    X_lib, X_sample[:, s0:s1, :],
                    nbrs_num, nbrs_num_max, lib_indices, smpl_indices[s0:s1],
                    exclusion_rad
                ) 
            except RuntimeError as e:
                if self._is_oom(e):
                    self._hard_clear()
                raise

            I = indices.reshape(num_ts_X, -1).T

            weights_c = torch.permute(weights, (1, 2, 0))[:, :, None, None] \
                            .expand(-1, -1, max_E_Y, num_ts_Y, -1).to(self.compute_dtype)

            
            Y_lib_shifted_indexed = torch.permute(Y_lib_shifted[:, I], (1, 3, 0, 2)).to(self.compute_dtype)
            A_blk = Y_lib_shifted_indexed.reshape(-1, nbrs_num_max, max_E_Y, num_ts_Y, num_ts_X)
            A_blk = (A_blk * weights_c).sum(axis=1)  # (B, E_y, n_Y, n_X)

            A[s0:s1] = A_blk.to(self.dtype)

            del weights, indices, I, weights_c, Y_lib_shifted_indexed, A_blk

        if (Y_sample_shifted is None) and return_pred:
            return A

        B = torch.permute(Y_sample_shifted, (1, 2, 0)).to(self.compute_dtype)[:, :, :, None] \
            .expand(Y_sample_shifted.shape[1], max_E_Y, num_ts_Y, num_ts_X)

        r_AB = metric_fn(A.to(self.compute_dtype), B)

        if return_pred:
            return (r_AB, A)
        else:
            return r_AB

    @torch.inference_mode()
    def __smap_prediction(self, lib_indices, smpl_indices, X_lib, X_sample, Y_lib_shifted, Y_sample_shifted,
                      exclusion_rad, theta, metric_fn, return_pred=False,
                      sample_batch_size=None, ridge=0.0):
        num_ts_X = X_lib.shape[0]
        num_ts_Y = Y_lib_shifted.shape[0]
        max_E_X  = X_lib.shape[2]
        max_E_Y  = Y_lib_shifted.shape[2]
        subsample_size = X_sample.shape[1]
        subset_size    = X_lib.shape[1]

        A_all = torch.empty((subsample_size, max_E_Y, num_ts_Y, num_ts_X), device=self.device, dtype=self.dtype)

        if sample_batch_size is None or sample_batch_size >= subsample_size:
            sample_batch_size = subsample_size

        for s0 in range(0, subsample_size, sample_batch_size):
            s1 = min(subsample_size, s0 + sample_batch_size)
            B  = s1 - s0

            # Slice the queries 
            X_sample_b = X_sample[:, s0:s1, :]  # (num_ts_X, B, max_E_X)

            try:
                weights = self.__get_local_weights(
                    lib=X_lib, sublib=X_sample_b,
                    subset_idx=lib_indices, sample_idx=smpl_indices[s0:s1],
                    exclusion_rad=exclusion_rad, theta=theta
                ).to(self.compute_dtype)

                W = weights.unsqueeze(1).expand(num_ts_X, num_ts_Y, B, subset_size) \
                                    .reshape(num_ts_X * num_ts_Y * B, subset_size, 1)

                X = X_lib.to(self.compute_dtype).unsqueeze(1).unsqueeze(1).expand(num_ts_X, num_ts_Y, B, subset_size, max_E_X) \
                                    .reshape(num_ts_X * num_ts_Y * B, subset_size, max_E_X)

                Y = Y_lib_shifted.to(self.compute_dtype).unsqueeze(1).unsqueeze(0).expand(num_ts_X, num_ts_Y, B, subset_size, max_E_Y) \
                                            .reshape(num_ts_X * num_ts_Y * B, subset_size, max_E_Y)

                X_intercept = torch.cat([torch.ones((num_ts_X * num_ts_Y * B, subset_size, 1), device=self.device, dtype=self.compute_dtype), X], dim=2)

                X_intercept_weighted = X_intercept * W
                Y_weighted = Y * W

                XTWX = torch.bmm(X_intercept_weighted.transpose(1, 2), X_intercept_weighted)
                if ridge and ridge > 0.0:
                    I = torch.eye(max_E_X + 1, device=self.device, dtype=self.compute_dtype).expand_as(XTWX)
                    XTWX = XTWX + ridge * I
                XTWy = torch.bmm(X_intercept_weighted.transpose(1, 2), Y_weighted)

                L = torch.linalg.cholesky(XTWX)         
                beta = torch.cholesky_solve(XTWy, L)
                #beta = torch.bmm(torch.pinverse(XTWX), XTWy) 

                X_ = X_sample_b.to(self.compute_dtype).unsqueeze(1).expand(num_ts_X, num_ts_Y, B, max_E_X) \
                                            .reshape(num_ts_X * num_ts_Y * B, max_E_X)
                X_ = torch.cat([torch.ones((num_ts_X * num_ts_Y * B, 1), device=self.device, dtype=self.compute_dtype), X_], dim=1) \
                    .reshape(num_ts_X * num_ts_Y * B, 1, max_E_X + 1)

                A = torch.bmm(X_, beta).reshape(num_ts_X, num_ts_Y, B, max_E_Y)
                A = torch.permute(A, (2, 3, 1, 0)).to(self.dtype)  # (B, max_E_Y, num_ts_Y, num_ts_X)

                A_all[s0:s1] = A  
                
                del weights, W, X, Y, X_intercept, X_intercept_weighted, Y_weighted, XTWX, XTWy, beta, X_, A
            except RuntimeError as e:
                if self._is_oom(e):
                    self._hard_clear()
                raise

        if (Y_sample_shifted is None) and return_pred:
            return A_all

        B_full = torch.permute(Y_sample_shifted, (1, 2, 0)).unsqueeze(-1).expand(subsample_size, max_E_Y, num_ts_Y, num_ts_X).to(self.compute_dtype)
        r_AB = metric_fn(A_all.to(self.compute_dtype), B_full)

        if return_pred:
            return (r_AB, A_all)
        else:
            return r_AB
        

    def __get_random_indices(self, num_points, sample_len, generator=None):
        idxs_X = torch.argsort(torch.rand(num_points, device=self.device, generator=generator))[0:sample_len]

        return idxs_X


    def __get_random_sample(self, X, min_len, indices, dim, max_E):
        X_buf = torch.zeros((dim, indices.shape[0], max_E),device=self.device, dtype=self.dtype)

        for i in range(dim):
            Xi = torch.tensor(X[i][-min_len:],device=self.device, dtype=self.dtype)
            X_buf[i, :, :X[i].shape[-1]] = Xi[indices]

        return X_buf

    def __get_nbrs_indices_with_weights(
        self, lib, sample, n_nbrs, n_nbrs_max, lib_idx, sample_idx, exclusion_rad
    ):
        try:
            dist = self._cdist(sample, lib)  # (num_ts_X, S_blk, L)
        except RuntimeError as e:
            if self._is_oom(e):
                self._hard_clear()
            raise

        if exclusion_rad is None:
            near_dist, indices = torch.topk(dist, 1 + n_nbrs_max, largest=False)
            indices   = indices[:, :, :n_nbrs_max]
            near_dist = near_dist[:, :, :n_nbrs_max]
        else:
            k_cand = min(lib.shape[1], 1 + n_nbrs_max + 2 * exclusion_rad)
            near_dist_all, indices_all = torch.topk(dist, k_cand, largest=False)

            mask = ~(
                (lib_idx[indices_all] <= (sample_idx[:, None] + exclusion_rad)) &
                (lib_idx[indices_all] >= (sample_idx[:, None] - exclusion_rad))
            )

            selector = (mask.cumsum(dim=2) <= n_nbrs_max) & mask

            indices   = indices_all[selector].view(dist.shape[0], dist.shape[1], n_nbrs_max)
            near_dist = near_dist_all[selector].view(dist.shape[0], dist.shape[1], n_nbrs_max)

        return self.__weights_from_dists(near_dist, indices, n_nbrs, n_nbrs_max)
    
    def __weights_from_dists(self, near_dist, indices, n_nbrs, n_nbrs_max):
        eps = torch.finfo(near_dist.dtype).eps
        d0 = torch.clamp(near_dist[:, :, :1], min=eps)
        w  = torch.exp(-near_dist / d0)
        w  = torch.where(torch.isfinite(near_dist), w, torch.zeros_like(w))

        keep = (torch.arange(n_nbrs_max, device=w.device).unsqueeze(0) < n_nbrs.unsqueeze(1))
        w = w * keep[:, None, :].to(w.dtype)

        sumw = w.sum(dim=2, keepdim=True)
        zero = sumw <= eps
        if zero.any():
            raise RuntimeError(
                "All neighbors excluded by `exclusion_window` for some queries. "
                "Reduce `exclusion_window`, increase `library_size`, or ensure the "
                "library contains valid neighbors."
            )

        w = w / sumw.clamp_min(eps)
        return w.to(self.dtype), indices


    def __get_local_weights(self, lib, sublib, subset_idx, sample_idx, exclusion_rad, theta):
        dist = self._cdist(sublib, lib)
        if theta == None:
            weights = torch.exp(-(dist))
        else:
            denom = dist.mean(dim=2, keepdim=True).clamp_min(1e-12)  # (n_X, S, 1)
            weights = torch.exp(-(theta * dist / denom))

        #if exclusion_rad > 0:
        if exclusion_rad is not None:
            exclusion_matrix = (torch.abs(subset_idx[None] - sample_idx[:,None]) > exclusion_rad)
            weights = weights * exclusion_matrix.to(weights.dtype)
    
        return weights.to(self.dtype)
      
    def _cdist(self, a, b):
        comp = self.compute_dtype
        if self.device.startswith("cpu") and comp == torch.float16:
            comp = torch.float32
        try:
            a = a.to(dtype=comp)
            b = b.to(dtype=comp)
            return torch.cdist(a, b, p=2, compute_mode="use_mm_for_euclid_dist")
        except RuntimeError as e:
            if self._is_oom(e):
                self._hard_clear()
            raise


    def _is_oom(self, e: Exception) -> bool:
        msg = str(e).lower()
        return any(s in msg for s in (
            "cuda out of memory", "out of memory", "failed to allocate",
            "can't allocate memory", "cublas status alloc failed", "mps allocation failed"
        ))

    def _soft_clear(self):
        gc.collect()
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _hard_clear(self):
        gc.collect()
        if self.device.startswith("cuda") and torch.cuda.is_available():
            try:
                torch.cuda.synchronize(self.device)
            except Exception:
                pass
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            try:
                torch.cuda.reset_peak_memory_stats(self.device)
            except Exception:
                pass

    def __auto_batch_size_smap(self, X_lib, X_sample, Y_lib_s, budget_gb=8.0):

        num_ts_X, L, max_E_X = X_lib.shape
        num_ts_Y, _, max_E_Y = Y_lib_s.shape
        S = X_sample.shape[1]
        P = num_ts_X * num_ts_Y

        cbytes = torch.tensor([], dtype=self.compute_dtype).element_size()
        dbytes = torch.tensor([], dtype=self.dtype).element_size()

        # 1) distances + local weights (nX * L) each
        dist_w_per_sample = (cbytes + dbytes) * (num_ts_X * L)

        # 2) Expanded row stacks used in weighted normal equations
        #    W, X, Y, X_intercept, X_intercept_weighted, Y_weighted  (all ~ P * L * ...)
        rows_per_sample = cbytes * (
            P * L * (1         # W
                    + max_E_X # X
                    + max_E_Y # Y
                    + (max_E_X + 1) # X_intercept
                    + (max_E_X + 1) # X_intercept_weighted
                    + max_E_Y)      # Y_weighted
        )

        # 3) Normal equations and solve buffers per sample
        ne_per_sample = cbytes * (
            P * ((max_E_X + 1) * (max_E_X + 1)   # XTWX
                + (max_E_X + 1) * max_E_Y      # XTWy / beta
                + (max_E_X + 1))               # X_ (query design)
        )

        # (A_all is preallocated; not included in per-sample peak)
        per_sample_bytes = int(dist_w_per_sample + rows_per_sample + ne_per_sample)

        # Budget with some headroom
        budget_bytes = int(budget_gb * (1024 ** 3) * 0.90)

        if per_sample_bytes <= 0:
            return min(int(S), 1)

        B = budget_bytes // per_sample_bytes
        if B < 1:
            B = 1
        if B > int(S):
            B = int(S)
        return int(B)
    
    def __auto_batch_size_simplex(self, X_lib, X_sample, Y_lib_s, nbrs_num_max, budget_gb=8.0):
        """
        Estimate samples chunk size B for simplex using actual tensors.
        Conservative: counts cdist, KNN book-keeping, your 5D weights_c & gathered Y,
        the elementwise product, and the block output. Targets an ~budget_gb cap.
        """
        # Shapes
        num_ts_X, L, _      = X_lib.shape            # (nX, L, Ex)
        num_ts_Y, _, max_EY = Y_lib_s.shape          # (nY, L, Ey)
        S                   = X_sample.shape[1]
        nX, nY, Ey, K       = int(num_ts_X), int(num_ts_Y), int(max_EY), int(nbrs_num_max)

        # dtype sizes
        cbytes = torch.tensor([], dtype=self.compute_dtype).element_size()  # compute dtype
        dbytes = torch.tensor([], dtype=self.dtype).element_size()          # model/output dtype
        ibytes = 8  # int64 indices

        # Per-sample (B=1) estimates:

        # 1) full pairwise distances: (nX, 1, L)
        cdist_per_sample = cbytes * (nX * L)

        # 2) topk KNN bookkeeping per sample (weights + indices): (nX, 1, K)
        knn_per_sample = (dbytes + ibytes) * (nX * K)

        # 3) your core 5D tensors for one sample:
        #    weights_c:     (1, K, Ey, nY, nX)
        #    gathered Y:    (1, K, Ey, nY, nX)
        #    product/sum:   same shape in compute dtype (we count one more copy)
        core_5d_per_sample = cbytes * (3 * K * Ey * nY * nX)

        # 4) output slice per sample: (Ey, nY, nX) in self.dtype
        out_per_sample = dbytes * (Ey * nY * nX)

        per_sample_bytes = int(cdist_per_sample + knn_per_sample + core_5d_per_sample + out_per_sample)

        # Budget (leave headroom)
        budget_bytes = int(budget_gb * (1024 ** 3) * 0.90)

        if per_sample_bytes <= 0:
            return min(int(S), 1)

        B = budget_bytes // per_sample_bytes
        if B < 1:
            B = 1
        if B > int(S):
            B = int(S)
        return int(B)