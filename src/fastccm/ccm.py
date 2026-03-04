# ccm.py
import warnings
import os
import torch
from .utils.metrics import (
    get_metric,
    get_streaming_metric_kind,
    stream_metric_state_init,
    stream_metric_state_update,
    stream_metric_state_finalize,
)
from .utils.runtime import (
    is_oom_error,
    soft_clear,
    hard_clear,
    format_bytes,
    tic,
    toc_ms,
    time_block,
    timings_summary,
    auto_batch_size_smap,
    auto_batch_size_simplex,
    batch_starts,
)
from .utils.logger import setup_logger
import math
import logging
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
    """
    Pairwise Convergent Cross Mapping (CCM) in PyTorch.

    Public API
    ----------
    • score_matrix(...)   -> CCM scores for all target/source pairs.
    • predict_matrix(...) -> Predicted target embeddings for all pairs.

    Notes
    -----
    • Inputs are Python lists of array-like 2D objects (NumPy arrays or Torch
      tensors), one per time series, with shape (T, E) = (time, embedding_dim).
    • When series in the input lists have different lengths, the implementation 
      **left-truncates** each series to the common minimum length and 
      **end-aligns** them. The **last time point is assumed to match at time t**, 
      and earlier samples are dropped.
    • When running on CPU with compute_dtype=float16, distances are promoted to
      float32 to maintain numerical stability.
    """

    def __init__(self,device = "cpu", dtype="float32", compute_dtype=None,
                 verbose = 0, log_file = None, memory_budget_gb=2.0):
        """
        Create a PairwiseCCM instance.

        Parameters
        ----------
        device : {"cpu", "cuda", "cuda:0", ...}, default "cpu"
            Device on which tensors will be allocated and computations performed.
            Use a specific CUDA device string to select a GPU.
        dtype : torch.dtype or str, default "float32"
            Storage dtype for internal tensors and outputs. Accepts torch dtypes or
            common strings like {"float16","float32","float64","bfloat16"} and aliases
            {"f16","f32","f64","fp16","fp32","fp64","bf16"}.
        compute_dtype : torch.dtype or str or None, default None
            Math-ops dtype used for heavy linear algebra (e.g., distances, solves).
            If None, uses the same value as `dtype`. 
        memory_budget_gb : float, default 2.0
            Memory budget (GB) used by automatic batching (`batch_size="auto"`).
            Larger values increase batch size and speed, but use more memory.
        """
        self.device = device
        self.dtype = _resolve_dtype(dtype) or torch.float32
        self.compute_dtype = _resolve_dtype(compute_dtype) or self.dtype
        self.memory_budget_gb = float(memory_budget_gb)

        # (Optional) sanity: ensure float type
        if not (self.dtype.is_floating_point and self.compute_dtype.is_floating_point):
            raise ValueError("dtype and compute_dtype must be floating dtypes.")
        if self.memory_budget_gb <= 0:
            raise ValueError("memory_budget_gb must be positive.")
        
        self.logger = setup_logger(__name__, verbose=verbose, log_file=log_file)

    def compute(self, *args, **kwargs):
        """
        DEPRECATED: Use `score_matrix(...)` instead.

        This method forwards all arguments to `score_matrix(...)` and emits a
        DeprecationWarning. See `score_matrix` for full parameter and return details.
        """
        warnings.warn("PairwiseCCM.compute → PairwiseCCM.score_matrix", DeprecationWarning)
        return self.score_matrix(*args, **kwargs)
    
    def predict(self, *args, **kwargs):
        """
        DEPRECATED: Use `predict_matrix(...)` instead.

        This method forwards all arguments to `predict_matrix(...)` and emits a
        DeprecationWarning. See `predict_matrix` for full parameter and return details.
        """
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
            batch_size="auto",
            clean_after=True,
            **kwargs
    ):
        """
        Compute pairwise CCM scores for all embedding pairs.
        
        Parameters
        ----------
        X_emb : list[array-like]
            Source embeddings, one per series. Each item is 2D with shape (T_x, E_x),
            where T_x is time steps and E_x is the embedding dimension for that series.
            Accepts NumPy arrays or torch.Tensors; data are copied to `device`.
        Y_emb : list[array-like] or None, default None
            Target embeddings, same structure as X_emb. If None, uses X_emb (i.e.,
            computes all-against-all within one set).
        library_size : int or {"auto", None}, default None
            Number of library points drawn from each series:
            • None  -> use the maximum common length across series.
            • "auto"-> min(max_common_len // 2, 700).
            • int   -> use the provided count (clipped internally to valid range).
            Library indices are sampled uniformly at random (reproducible with `seed`).
        sample_size : int or {"auto", None}, default None
            Number of query (evaluation) points drawn from each series:
            • None  -> use the maximum common length across series.
            • "auto"-> min(max_common_len // 6, 250).
            • int   -> use the provided count (clipped internally to valid range).
            Sample indices are drawn uniformly at random (reproducible with `seed`).
        exclusion_window : int or None, default 0
            Temporal exclusion radius (in samples). When an integer r is provided,
            neighbors with |t_neighbor − t_query| ≤ r are excluded (including self when r>=0).
            Use None to disable temporal exclusion entirely (self-neighbor allowed).
        tp : int, optional
            Prediction horizon. Predict Y[t + tp] from X[t]. Default: 0.
        method : {"simplex", "smap"}, default "simplex"
            Local model used for neighbor-based prediction at each query point:
            • "simplex": k-NN exponential weights (Sugihara's simplex method).
            `nbrs_num` may be provided in `kwargs`; default is E_x + 1 per source series.
            • "smap"   : Locally weighted linear regression with parameter `theta`
            (default 1.0) and optional `ridge` (default 0.0) in `kwargs`.
        seed : int or None, default None
            Seed for deterministic sampling of library and sample indices.
        metric : {"corr","mse","mae","rmse","neg_nrmse","dcorr"} or Callable, default "corr"
            Scoring applied to (prediction, target).
        batch_size : int or {"auto", None}, default "auto"
            Number of query points processed per chunk. If "auto", a heuristic estimates
            a safe chunk size using `memory_budget_gb` from the class constructor.
            If None,
            processes all at once (may be memory heavy).
        clean_after : bool, default False
            If True, run a cleanup after returning (calls Python GC and clears
            PyTorch/CUDA caching allocators). Keep False inside loops for
            performance.

        Other Parameters
        ----------------
        nbrs_num : int or list[int], optional (simplex only)
            Number of neighbors per source series. If int, the same k is used for all.
            Default is E_x + 1 for each source series.
        theta : float, optional (smap only)
            Local weighting strength; larger values induce steeper locality.
            Default is 1.0.
        ridge : float, optional (smap only)
            Non-negative ridge penalty added to the local linear regression. Default 0.0.

        Returns
        -------
        np.ndarray
            CCM scores with shape (E_y, n_Y, n_X), where:
            • E_y : maximum embedding dimension across targets,
            • n_Y : number of target series,
            • n_X : number of source series.

        Raises
        ------
        ValueError
            If `method` is invalid or there are not enough points after applying `tp`.
        RuntimeError
            If all neighbors are excluded by `exclusion_window` for some queries.
            Out-of-memory errors are surfaced after internal cache clearing.
        """
        if Y_emb is None:
            Y_emb = X_emb

        self.logger.info(
            "score_matrix started (n_x=%d, n_y=%d, method=%s, tp=%d, metric=%s)",
            len(X_emb), len(Y_emb), method, tp, metric
        )
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
        self.logger.info("score_matrix completed with output shape %s", r_AB.shape)
        if clean_after:
            soft_clear(self.logger, self.device)
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
            batch_size="auto",
            clean_after=True,
            **kwargs
    ):
        """
        Predict target embeddings at given query points for all (target, source) pairs.

        Parameters
        ----------
        X_lib_emb : list[array-like]
            Source library embeddings, one per series. Each item is 2D with shape
            (T_lib, E_x). Accepts NumPy arrays or torch.Tensors.
        Y_lib_emb : list[array-like] or None, default None
            Target library embeddings, same structure as X_lib_emb. If None, uses X_lib_emb.
            Targets are aligned using Y[t + tp] so ensure `tp` leaves enough points.
        X_pred_emb : list[array-like] or None, default None
            Source query embeddings at which to evaluate predictions, one per series,
            shaped (T_pred, E_x). If None, uses X_lib_emb (predict-on-library).
        library_size : int or {"auto", None}, default None
            Number of library points drawn from each series:
            • None  -> use the maximum common library length across series.
            • "auto"-> min(max_lib_len // 2, 700).
            • int   -> use the provided count (clipped internally to valid range).
            Library indices are sampled uniformly at random (reproducible with `seed`).
        exclusion_window : int or None, default 0
            Temporal exclusion radius (in samples). When an integer r is provided,
            neighbors with |t_neighbor − t_query| ≤ r are excluded (including self when r>=0).
            Use None to disable temporal exclusion entirely (self-neighbor allowed).
        tp : int, optional
            Prediction horizon. Predict Y[t + tp] from X[t]. Default: 0.
        method : {"simplex", "smap"}, default "simplex"
            Local model used for neighbor-based prediction at each query point:
            • "simplex": k-NN exponential weights (Sugihara's simplex method).
            `nbrs_num` may be provided in `kwargs`; default is E_x + 1 per source series.
            • "smap"   : Locally weighted linear regression with parameter `theta`
            (default 1.0) and optional `ridge` (default 0.0) in `kwargs`.
        seed : int or None, default None
            Seed for deterministic sampling of library and sample indices.
        metric : {"corr","mse","mae","rmse","neg_nrmse","dcorr"} or Callable, default "corr"
            Scoring applied to (prediction, target).
        batch_size : int or {"auto", None}, default "auto"
            Number of query points processed per chunk. If "auto", a heuristic estimates
            a safe chunk size using `memory_budget_gb` from the class constructor.
            If None,
            processes all at once (may be memory heavy).
        clean_after : bool, default False
            If True, run a cleanup after returning (calls Python GC and clears
            PyTorch/CUDA caching allocators). Keep False inside loops for
            performance.

        Other Parameters
        ----------------
        nbrs_num : int or list[int], optional (simplex only)
            Number of neighbors per source series. If int, the same k is used for all.
            Default is E_x + 1 for each source series.
        theta : float, optional (smap only)
            Local weighting strength; larger values induce steeper locality.
            Default is 1.0.
        ridge : float, optional (smap only)
            Non-negative ridge penalty added to the local linear regression. Default 0.0.

        Returns
        -------
        np.ndarray
            Predicted target embeddings with shape (T_pred, E_y, n_Y, n_X), where:
            • T_pred : number of query points per source series,
            • E_y    : maximum embedding dimension across targets,
            • n_Y    : number of target series,
            • n_X    : number of source series.

        Raises
        ------
        ValueError
            If `method` is invalid or there are not enough points after applying `tp`.
        RuntimeError
            If all neighbors are excluded by `exclusion_window` for some queries.
            Out-of-memory errors are surfaced after internal cache clearing.
        """
        if Y_lib_emb is None:
            Y_lib_emb = X_lib_emb
        if X_pred_emb is None:
            X_pred_emb = X_lib_emb

        self.logger.info(
            "predict_matrix started (n_x=%d, n_y=%d, n_pred=%d, method=%s, tp=%d)",
            len(X_lib_emb), len(Y_lib_emb), len(X_pred_emb), method, tp
        )
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
        self.logger.info("predict_matrix completed with output shape %s", A.shape)
        if clean_after:
            soft_clear(self.logger, self.device)
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
        self.logger.debug(
            "__ccm_core(mode=%s, method=%s, library_size=%s, sample_size=%s, exclusion_window=%s, batch_size=%s)",
            mode, method, library_size, sample_size, exclusion_window, batch_size
        )
        # ---------- 1) dims / lengths ----------
        num_ts_X = len(X_lib_list)
        num_ts_Y = len(Y_lib_list)

        max_E_X = torch.tensor([X_lib_list[i].shape[-1] for i in range(num_ts_X)], device=self.device).max().item()
        max_E_Y = torch.tensor([Y_lib_list[i].shape[-1] for i in range(num_ts_Y)], device=self.device).max().item()

        if mode == "score":
            min_len = min(
                min(y.shape[0] for y in Y_lib_list),
                min(x.shape[0] for x in X_lib_list)
            )
            self.logger.info(
                "Embedding common_len=%d max_dim=%d",
                int(min_len), int(max(max_E_X, max_E_Y))
            )
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

            min_len_lib = min(
                min(y.shape[0] for y in Y_lib_list),
                min(x.shape[0] for x in X_lib_list)
            )
            min_len_pred = min(x.shape[0] for x in X_sample_list)
            self.logger.info(
                "Embedding common_lib_len=%d common_pred_len=%d max_dim=%d",
                int(min_len_lib), int(min_len_pred), int(max(max_E_X, max_E_Y))
            )

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
            library_size_mode = "explicit"
            if library_size is None:
                library_size_res = min_len
                library_size_mode = "none->min_len"
            elif library_size == "auto":
                library_size_res = min(min_len // 2, 700)
                library_size_mode = "auto"
            else:
                library_size_res = int(library_size)

            sample_size_mode = "explicit"
            if sample_size is None:
                sample_size_res = min_len
                sample_size_mode = "none->min_len"
            elif sample_size == "auto":
                sample_size_res = min(min_len // 6, 250)
                sample_size_mode = "auto"
            else:
                sample_size_res = int(sample_size)

            self.logger.info(
                "library_size=%d (%s, input=%s) sample_size=%d (%s, input=%s) tp=%d",
                int(library_size_res), library_size_mode, str(library_size),
                int(sample_size_res), sample_size_mode, str(sample_size),
                int(tp)
            )
        else:  # predict
            library_size_mode = "explicit"
            if library_size is None:
                library_size_res = min_len_lib
                library_size_mode = "none->min_len_lib"
            elif library_size == "auto":
                library_size_res = min(min_len_lib // 2, 700)
                library_size_mode = "auto"
            else:
                library_size_res = int(library_size)
            self.logger.info(
                "library_size=%d (%s, input=%s) sample_size=%d (predict-uses-all-queries) tp=%d",
                int(library_size_res), library_size_mode, str(library_size),
                int(min_len_pred), int(min_len_lib), int(tp)
            )
            pred_shape = (int(min_len_pred), int(max_E_Y), int(num_ts_Y), int(num_ts_X))
            pred_bytes = int(
                pred_shape[0] * pred_shape[1] * pred_shape[2] * pred_shape[3]
                * torch.tensor([], dtype=self.dtype).element_size()
            )
            self.logger.warning(
                "Predict final tensor allocation shape=%s dtype=%s approx=%s on device=%s (before CPU/NumPy transfer).",
                str(pred_shape),
                str(self.dtype),
                format_bytes(pred_bytes),
                str(self.device),
            )

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
            auto_batch = (batch_size == "auto")
            nbrs_num_max = nbrs_num.max().item()
            total_samples = int(X_sample.shape[1])
            if auto_batch:
                batch_size, batch_auto_meta = auto_batch_size_simplex(
                    X_lib, X_sample, Y_lib_s, nbrs_num_max, dtype=self.dtype, compute_dtype=self.compute_dtype, budget_gb=self.memory_budget_gb
                )
            else:
                _, batch_auto_meta = auto_batch_size_simplex(
                    X_lib, X_sample, Y_lib_s, nbrs_num_max, dtype=self.dtype, compute_dtype=self.compute_dtype, budget_gb=self.memory_budget_gb
                )
            if batch_size is not None and batch_size <= 0:
                raise ValueError("batch_size must be positive, 'auto', or None.")
            self.logger.info(
                "Batching policy=%s total_samples=%d batch_size=%d num_batches=%d split=%s per_sample_est=%s budget=%s selected_batch_peak_est=%s",
                "auto" if auto_batch else ("all-at-once" if batch_size is None else "manual"),
                total_samples,
                total_samples if batch_size is None else int(batch_size),
                max(1, int(math.ceil(total_samples / max(total_samples if batch_size is None else int(batch_size), 1)))),
                str((total_samples if batch_size is None else int(batch_size)) < total_samples),
                format_bytes(batch_auto_meta["per_sample_bytes"]),
                format_bytes(batch_auto_meta["budget_bytes"]),
                format_bytes((total_samples if batch_size is None else int(batch_size)) * max(batch_auto_meta["per_sample_bytes"], 0)),
            )

            out = self.__simplex_prediction(
                lib_indices, smpl_indices,
                X_lib, X_sample, Y_lib_s, Y_smp_s,
                exclusion_window, nbrs_num, metric_fn=metric_fn,
                return_pred=(Y_smp_s is None), sample_batch_size=batch_size,
                nbrs_num_max=int(nbrs_num_max),
            )


        else:
            auto_batch = (batch_size == "auto")
            total_samples = int(X_sample.shape[1])
            if auto_batch:
                batch_size, batch_auto_meta = auto_batch_size_smap(
                    X_lib, X_sample, Y_lib_s, dtype=self.dtype, compute_dtype=self.compute_dtype, budget_gb=self.memory_budget_gb
                )
            else:
                _, batch_auto_meta = auto_batch_size_smap(
                    X_lib, X_sample, Y_lib_s, dtype=self.dtype, compute_dtype=self.compute_dtype, budget_gb=self.memory_budget_gb
                )
            if batch_size is not None and batch_size <= 0:
                raise ValueError("batch_size must be positive, 'auto', or None.")
            self.logger.info(
                "Batching policy=%s total_samples=%d batch_size=%d num_batches=%d split=%s per_sample_est=%s budget=%s selected_batch_peak_est=%s",
                "auto" if auto_batch else ("all-at-once" if batch_size is None else "manual"),
                total_samples,
                total_samples if batch_size is None else int(batch_size),
                max(1, int(math.ceil(total_samples / max(total_samples if batch_size is None else int(batch_size), 1)))),
                str((total_samples if batch_size is None else int(batch_size)) < total_samples),
                format_bytes(batch_auto_meta["per_sample_bytes"]),
                format_bytes(batch_auto_meta["budget_bytes"]),
                format_bytes((total_samples if batch_size is None else int(batch_size)) * max(batch_auto_meta["per_sample_bytes"], 0)),
            )


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
                              exclusion_rad, nbrs_num, metric_fn, return_pred=False, sample_batch_size=None,
                              nbrs_num_max=None):
        num_ts_X = X_lib.shape[0]
        num_ts_Y = Y_lib_shifted.shape[0]
        max_E_Y = Y_lib_shifted.shape[2]
        subsample_size = X_sample.shape[1]

        if (sample_batch_size is None) or (sample_batch_size >= subsample_size):
            sample_batch_size = subsample_size

        if nbrs_num_max is None:
            nbrs_num_max = int(nbrs_num.max().item())
        self.logger.debug(
            "Entering simplex backend (queries=%d, library_points=%d, num_sources=%d, num_targets=%d, Ey=%d, nbrs_max=%d, batch_size=%d, num_batches=%d, exclusion=%s)",
            int(subsample_size),
            int(X_lib.shape[1]),
            int(num_ts_X),
            int(num_ts_Y),
            int(max_E_Y),
            int(nbrs_num_max),
            int(sample_batch_size),
            int(math.ceil(subsample_size / sample_batch_size)),
            str(exclusion_rad),
        )
        stream_kind = get_streaming_metric_kind(metric_fn) if (not return_pred) else None
        stream_state = None
        if stream_kind is not None:
            stream_state = stream_metric_state_init(
                stream_kind, max_E_Y, num_ts_Y, num_ts_X, device="cpu", dtype=self.compute_dtype
            )

        # Keep full output off accelerator in score mode so device memory tracks batch size.
        out_device = self.device if ((Y_sample_shifted is None) and return_pred) else "cpu"
        #nbrs_mask = (torch.arange(nbrs_num_max).unsqueeze(0) < nbrs_num.unsqueeze(1))
        A = None if stream_kind is not None else torch.empty((subsample_size, max_E_Y, num_ts_Y, num_ts_X), device=out_device, dtype=self.dtype)
        for s0 in batch_starts(self.logger, subsample_size, sample_batch_size, "simplex batches"):
            s1 = min(subsample_size, s0 + sample_batch_size)
            self.logger.debug(
                "Simplex batch [%d:%d) started (batch_queries=%d)",
                int(s0), int(s1), int(s1 - s0)
            )
            timings = {}
            t_batch = tic(self.logger, self.device) if self._debug_enabled() else None

            try:
                with time_block(self.logger, self.device, timings, "neighbors"):
                    X_sample_b = X_sample[:, s0:s1, :].to(device=self.device, dtype=self.dtype, copy=False)
                    weights, indices = self.__get_nbrs_indices_with_weights(
                        X_lib, X_sample_b,
                        nbrs_num, nbrs_num_max, lib_indices, smpl_indices[s0:s1],
                        exclusion_rad
                    )
            except RuntimeError as e:
                if is_oom_error(e):
                    self.logger.warning("OOM in simplex batch [%d:%d): %s", int(s0), int(s1), str(e))
                    hard_clear(self.logger, self.device)
                raise

            I = indices

            weights_c = weights.to(self.compute_dtype)

            with time_block(self.logger, self.device, timings, "gather"):
                Y_idx = Y_lib_shifted.to(self.compute_dtype).permute(1, 0, 2)[I]
                Y_lib_shifted_indexed = Y_idx.reshape(num_ts_X, s1 - s0, nbrs_num_max, num_ts_Y, max_E_Y)

            with time_block(self.logger, self.device, timings, "weighted_avg"):
                A_blk = torch.einsum("xbk,xbkye->xbye", weights_c, Y_lib_shifted_indexed)
                A_blk = A_blk.permute(1, 3, 2, 0).contiguous()  # (B, E_y, n_Y, n_X)

            if stream_kind is not None:
                with time_block(self.logger, self.device, timings, "metric"):
                    B_blk = torch.permute(Y_sample_shifted[:, s0:s1, :], (1, 2, 0)).to(device="cpu", dtype=self.compute_dtype)[:, :, :, None] \
                        .expand(s1 - s0, max_E_Y, num_ts_Y, num_ts_X)
                    stream_metric_state_update(
                        stream_kind,
                        stream_state,
                        A_blk.to(device="cpu", dtype=self.compute_dtype),
                        B_blk,
                    )
                    del B_blk
            else:
                with time_block(self.logger, self.device, timings, "store"):
                    A[s0:s1] = A_blk.to(out_device, dtype=self.dtype)

            if self._debug_enabled():
                timings["total"] = toc_ms(self.logger, self.device, t_batch)
                self.logger.debug(
                    "Simplex batch [%d:%d) timings: %s",
                    int(s0),
                    int(s1),
                    timings_summary(
                        timings,
                        order=["neighbors", "gather", "weighted_avg", "metric", "store", "total"],
                    ),
                )

            del weights, indices, I, weights_c, Y_idx, Y_lib_shifted_indexed, A_blk

        if stream_kind is not None:
            return stream_metric_state_finalize(stream_kind, stream_state)

        if (Y_sample_shifted is None) and return_pred:
            return A

        self.logger.debug("Computing simplex score metric")
        B = torch.permute(Y_sample_shifted, (1, 2, 0)).to(device=A.device, dtype=self.compute_dtype)[:, :, :, None] \
            .expand(Y_sample_shifted.shape[1], max_E_Y, num_ts_Y, num_ts_X)

        r_AB = metric_fn(A.to(dtype=self.compute_dtype), B)

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
        stream_kind = get_streaming_metric_kind(metric_fn) if (not return_pred) else None
        stream_state = None
        if stream_kind is not None:
            stream_state = stream_metric_state_init(
                stream_kind, max_E_Y, num_ts_Y, num_ts_X, device="cpu", dtype=self.compute_dtype
            )

        # Keep full output off accelerator in score mode so device memory tracks batch size.
        out_device = self.device if ((Y_sample_shifted is None) and return_pred) else "cpu"
        A_all = None if stream_kind is not None else torch.empty((subsample_size, max_E_Y, num_ts_Y, num_ts_X), device=out_device, dtype=self.dtype)

        if sample_batch_size is None or sample_batch_size >= subsample_size:
            sample_batch_size = subsample_size

        self.logger.debug(
            "Entering smap backend (queries=%d, library_points=%d, num_sources=%d, num_targets=%d, Ex=%d, Ey=%d, batch_size=%d, num_batches=%d, exclusion=%s, theta=%s, ridge=%s)",
            int(subsample_size),
            int(subset_size),
            int(num_ts_X),
            int(num_ts_Y),
            int(max_E_X),
            int(max_E_Y),
            int(sample_batch_size),
            int(math.ceil(subsample_size / sample_batch_size)),
            str(exclusion_rad),
            str(theta),
            str(ridge),
        )

        Xc = X_lib.to(self.compute_dtype)                 # (nX, L, Ex)
        Yc = Y_lib_shifted.to(self.compute_dtype)         # (nY, L, Ey)
        onesL = torch.ones((num_ts_X, subset_size, 1), device=self.device, dtype=self.compute_dtype)
        Xint = torch.cat([onesL, Xc], dim=2)             # (nX, L, Ex1)
        I = None
        if ridge and ridge > 0.0:
            I = torch.eye(max_E_X + 1, device=self.device, dtype=self.compute_dtype)[None, None]

        for s0 in batch_starts(self.logger, subsample_size, sample_batch_size, "smap batches"):
            s1 = min(subsample_size, s0 + sample_batch_size)
            B  = s1 - s0
            self.logger.debug(
                "SMAP batch [%d:%d) started (batch_queries=%d)",
                int(s0), int(s1), int(s1 - s0)
            )
            timings = {}
            t_batch = tic(self.logger, self.device) if self._debug_enabled() else None

            # Slice the queries 
            with time_block(self.logger, self.device, timings, "slice"):
                X_sample_b = X_sample[:, s0:s1, :].to(device=self.device, dtype=self.dtype, copy=False)  # (num_ts_X, B, max_E_X)

            try:
                with time_block(self.logger, self.device, timings, "local_weights"):
                    weights = self.__get_local_weights(
                        lib=X_lib, sublib=X_sample_b,
                        subset_idx=lib_indices, sample_idx=smpl_indices[s0:s1],
                        exclusion_rad=exclusion_rad, theta=theta
                    ).to(self.compute_dtype)

                with time_block(self.logger, self.device, timings, "square"):
                    weights.square_()  # (nX, B, L) in-place; avoid extra w2 allocation

                # XTWX: (nX, B, Ex1, Ex1)
                with time_block(self.logger, self.device, timings, "XTWX"):
                    XTWX = torch.einsum("xli,xbl,xlj->xbij", Xint, weights, Xint)
                    if I is not None:
                        XTWX = XTWX + ridge * I

                # XTWy: (nX, B, Ex1, nY, Ey) -> flatten to (nX, B, Ex1, nY*Ey)
                with time_block(self.logger, self.device, timings, "XTWy"):
                    XTWy = torch.einsum("xli,xbl,yle->xbiye", Xint, weights, Yc).reshape(
                        num_ts_X, B, max_E_X + 1, num_ts_Y * max_E_Y
                    )

                with time_block(self.logger, self.device, timings, "cholesky"):
                    Lchol = torch.linalg.cholesky(XTWX)                 # (nX, B, Ex1, Ex1)
                with time_block(self.logger, self.device, timings, "solve"):
                    beta  = torch.cholesky_solve(XTWy, Lchol)           # (nX, B, Ex1, nY*Ey)

                with time_block(self.logger, self.device, timings, "query_design"):
                    Xq = X_sample_b.to(self.compute_dtype)              # (nX, B, Ex)
                    Xq = torch.cat(
                        [torch.ones((num_ts_X, B, 1), device=self.device, dtype=self.compute_dtype), Xq],
                        dim=2
                    )                                                   # (nX, B, Ex1)

                with time_block(self.logger, self.device, timings, "predict"):
                    pred_flat = torch.matmul(Xq.unsqueeze(2), beta).squeeze(2)  # (nX, B, nY*Ey)

                A = pred_flat.view(num_ts_X, B, num_ts_Y, max_E_Y).permute(1, 3, 2, 0)  # (B,Ey,nY,nX)
                if stream_kind is not None:
                    with time_block(self.logger, self.device, timings, "metric"):
                        B_blk = torch.permute(Y_sample_shifted[:, s0:s1, :], (1, 2, 0)).to(device="cpu", dtype=self.compute_dtype)[:, :, :, None] \
                            .expand(B, max_E_Y, num_ts_Y, num_ts_X)
                        stream_metric_state_update(
                            stream_kind,
                            stream_state,
                            A.to(device="cpu", dtype=self.compute_dtype),
                            B_blk,
                        )
                        del B_blk
                else:
                    with time_block(self.logger, self.device, timings, "store"):
                        A_all[s0:s1] = A.to(out_device, dtype=self.dtype)

                if self._debug_enabled():
                    timings["total"] = toc_ms(self.logger, self.device, t_batch)
                    self.logger.debug(
                        "SMAP batch [%d:%d) timings: %s",
                        int(s0),
                        int(s1),
                        timings_summary(
                            timings,
                            order=["slice", "local_weights", "square", "cast_xy", "design", "XTWX", "XTWy",
                                   "cholesky", "solve", "query_design", "predict", "metric", "store", "total"],
                        ),
                    )

                del weights, XTWX, XTWy, Lchol, beta, Xq, pred_flat, A
            except RuntimeError as e:
                if is_oom_error(e):
                    self.logger.warning("OOM in SMAP batch [%d:%d): %s", int(s0), int(s1), str(e))
                    hard_clear(self.logger, self.device)
                raise

        if stream_kind is not None:
            return stream_metric_state_finalize(stream_kind, stream_state)

        if (Y_sample_shifted is None) and return_pred:
            return A_all

        self.logger.debug("Computing SMAP score metric")
        B_full = torch.permute(Y_sample_shifted, (1, 2, 0)).unsqueeze(-1).expand(
            subsample_size, max_E_Y, num_ts_Y, num_ts_X
        ).to(device=A_all.device, dtype=self.compute_dtype)
        r_AB = metric_fn(A_all.to(dtype=self.compute_dtype), B_full)

        if return_pred:
            return (r_AB, A_all)
        else:
            return r_AB

    def __get_random_indices(self, num_points, sample_len, generator=None):
        #idxs_X = torch.argsort(torch.rand(num_points, device=self.device, generator=generator))[0:sample_len]

        return torch.randperm(num_points, device=self.device, generator=generator)[:sample_len]


    def __get_random_sample(self, X, min_len, indices, dim, max_E):
        X_buf = torch.zeros((dim, indices.shape[0], max_E),device=self.device, dtype=self.dtype)

        for i in range(dim):
            Xi_src = X[i]
            if isinstance(Xi_src, torch.Tensor):
                Xi = Xi_src.to(device=self.device, dtype=self.dtype, copy=False)[-min_len:]
            else:
                Xi = torch.as_tensor(Xi_src[-min_len:], device=self.device, dtype=self.dtype)
            X_buf[i, :, :X[i].shape[-1]] = Xi[indices]

        return X_buf

    def __get_nbrs_indices_with_weights(
        self, lib, sample, n_nbrs, n_nbrs_max, lib_idx, sample_idx, exclusion_rad
    ):
        timings = {}
        try:
            with time_block(self.logger, self.device, timings, "cdist"):
                dist = self._cdist(sample, lib)  # (num_ts_X, S_blk, L)
        except RuntimeError as e:
            if is_oom_error(e):
                hard_clear(self.logger, self.device)
            raise

        if exclusion_rad is None:
            with time_block(self.logger, self.device, timings, "select"):
                near_dist, indices = torch.topk(dist, 1 + n_nbrs_max, largest=False, sorted=False)
                indices   = indices[:, :, :n_nbrs_max]
                near_dist = near_dist[:, :, :n_nbrs_max]
        else:
            with time_block(self.logger, self.device, timings, "select"):
                allowed = (
                    (lib_idx[None, :] > (sample_idx[:, None] + exclusion_rad)) |
                    (lib_idx[None, :] < (sample_idx[:, None] - exclusion_rad))
                )  # (S_blk, L)
                dist = dist.masked_fill(~allowed.unsqueeze(0), float("inf"))
                near_dist, indices = torch.topk(dist, n_nbrs_max, largest=False, sorted=False)

        with time_block(self.logger, self.device, timings, "weights"):
            weights, indices = self.__weights_from_dists(near_dist, indices, n_nbrs, n_nbrs_max)

        if self._debug_enabled():
            timings["total"] = sum(v for v in timings.values())
            self.logger.debug("Neighbor search timings: %s", timings_summary(timings, ["cdist", "select", "weights", "total"]))
        return weights, indices
    
    def __weights_from_dists(self, near_dist, indices, n_nbrs, n_nbrs_max):
        timings = {}
        eps = torch.finfo(near_dist.dtype).eps
        with time_block(self.logger, self.device, timings, "exp"):
            d0 = near_dist.min(dim=2, keepdim=True).values.clamp_min(eps)
            w  = torch.exp(-near_dist / d0)
            w  = torch.where(torch.isfinite(near_dist), w, torch.zeros_like(w))

        with time_block(self.logger, self.device, timings, "mask"):
            keep = (torch.arange(n_nbrs_max, device=w.device).unsqueeze(0) < n_nbrs.unsqueeze(1))
            w = w * keep[:, None, :].to(w.dtype)

        with time_block(self.logger, self.device, timings, "normalize"):
            sumw = w.sum(dim=2, keepdim=True)
            zero = sumw <= eps
            if zero.any():
                raise RuntimeError(
                    "All neighbors excluded by `exclusion_window` for some queries. "
                    "Reduce `exclusion_window`, increase `library_size`, or ensure the "
                    "library contains valid neighbors."
                )
            w = w / sumw.clamp_min(eps)
            out = w.to(self.dtype)

        if self._debug_enabled():
            timings["total"] = sum(v for v in timings.values())
            self.logger.debug("Neighbor weight timings: %s", timings_summary(timings, ["exp", "mask", "normalize", "total"]))
        return out, indices


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
            a = self.__to_tensor(a, dtype=comp)
            b = self.__to_tensor(b, dtype=comp)
            return torch.cdist(a, b, p=2, compute_mode="use_mm_for_euclid_dist")
        except RuntimeError as e:
            if is_oom_error(e):
                hard_clear(self.logger, self.device)
            raise

    def __to_tensor(self, arr, *, dtype=None, device=None):
        dtype  = self.dtype  if dtype  is None else dtype
        device = self.device if device is None else device
        if isinstance(arr, torch.Tensor):
            return arr.to(device=device, dtype=dtype, copy=False)
        return torch.as_tensor(arr, device=device, dtype=dtype)

    def _debug_enabled(self):
        return self.logger.isEnabledFor(logging.DEBUG)
