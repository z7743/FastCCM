# ccm.py
import gc
import warnings
import os
import torch
from .utils.metrics import get_metric
import math
import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime
#torch.set_num_threads(os.cpu_count())  
#torch.set_num_interop_threads(1)


class _MicrosecondFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")

def _setup_logger(name: str, verbose: int = 0, log_file: str | None = None) -> logging.Logger:
    """
    verbose:
      0 -> WARNING
      1 -> INFO
      2+-> DEBUG

    If log_file is None -> log to terminal (stderr).
    Else -> log to file.
    """
    logger = logging.getLogger(name)

    level = logging.WARNING if verbose <= 0 else (logging.INFO if verbose == 1 else logging.DEBUG)
    logger.setLevel(level)

    # Avoid duplicate handlers across multiple PairwiseCCM instances
    if not getattr(logger, "_pairwiseccm_configured", False):
        logger.propagate = False

        fmt = _MicrosecondFormatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S.%f",
        )

        handler = logging.StreamHandler(sys.stderr) if log_file is None else logging.FileHandler(log_file)
        handler.setLevel(level)
        handler.setFormatter(fmt)
        logger.addHandler(handler)

        logger._pairwiseccm_configured = True
    else:
        # Update handler levels when verbose changes
        for h in logger.handlers:
            h.setLevel(level)

    return logger

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
                 verbose = 0, log_file = None, ):
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
        """
        self.device = device
        self.dtype = _resolve_dtype(dtype) or torch.float32
        self.compute_dtype = _resolve_dtype(compute_dtype) or self.dtype

        # (Optional) sanity: ensure float type
        if not (self.dtype.is_floating_point and self.compute_dtype.is_floating_point):
            raise ValueError("dtype and compute_dtype must be floating dtypes.")
        
        self.logger = _setup_logger(__name__, verbose=verbose, log_file=log_file)

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
            a safe chunk size targeting ~8 GB peak usage with some headroom. If None,
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
            a safe chunk size targeting ~8 GB peak usage with some headroom. If None,
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
        self.logger.debug(
            "__ccm_core(mode=%s, method=%s, library_size=%s, sample_size=%s, exclusion_window=%s, batch_size=%s)",
            mode, method, library_size, sample_size, exclusion_window, batch_size
        )
        self.logger.debug("Core step 1/6: resolving dimensions and aligned lengths")
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

            if min_len_lib - tp <= 0 or min_len_pred <= 0:
                raise ValueError("Not enough points for library or prediction.")

        self.logger.debug("Core step 2/6: parsing method-specific parameters")
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

        self.logger.debug("Core step 3/6: resolving effective library/sample sizes")
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
                "resolved sizes[score] library_size=%d (%s, input=%s) sample_size=%d (%s, input=%s) base_min_len=%d tp=%d",
                int(library_size_res), library_size_mode, str(library_size),
                int(sample_size_res), sample_size_mode, str(sample_size),
                int(min_len), int(tp)
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
                "resolved sizes[predict] library_size=%d (%s, input=%s) sample_size=%d (predict-uses-all-queries) base_min_len_lib=%d tp=%d",
                int(library_size_res), library_size_mode, str(library_size),
                int(min_len_pred), int(min_len_lib), int(tp)
            )

        self.logger.debug("Core step 4/6: generating library/sample indices")
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

        self.logger.debug("Core step 5/6: materializing sampled tensors")
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

        self.logger.debug("Core step 6/6: running prediction backend (%s)", method)
        # ---------- 6) method call ----------
        if method == "simplex":
            auto_batch = (batch_size == "auto")
            nbrs_num_max = nbrs_num.max().item()
            total_samples = int(X_sample.shape[1])
            if auto_batch:
                batch_size, batch_auto_meta = self.__auto_batch_size_simplex(
                    X_lib, X_sample, Y_lib_s, nbrs_num_max, budget_gb=8.0
                )
            else:
                _, batch_auto_meta = self.__auto_batch_size_simplex(
                    X_lib, X_sample, Y_lib_s, nbrs_num_max, budget_gb=8.0
                )
            if batch_size is not None and batch_size <= 0:
                raise ValueError("batch_size must be positive, 'auto', or None.")
            self.logger.info(
                "batching[simplex] policy=%s total_samples=%d batch_size=%d num_batches=%d split=%s per_sample_est=%s budget=%s selected_batch_peak_est=%s",
                "auto" if auto_batch else ("all-at-once" if batch_size is None else "manual"),
                total_samples,
                total_samples if batch_size is None else int(batch_size),
                max(1, int(math.ceil(total_samples / max(total_samples if batch_size is None else int(batch_size), 1)))),
                str((total_samples if batch_size is None else int(batch_size)) < total_samples),
                self._format_bytes(batch_auto_meta["per_sample_bytes"]),
                self._format_bytes(batch_auto_meta["budget_bytes"]),
                self._format_bytes((total_samples if batch_size is None else int(batch_size)) * max(batch_auto_meta["per_sample_bytes"], 0)),
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
                batch_size, batch_auto_meta = self.__auto_batch_size_smap(X_lib, X_sample, Y_lib_s, budget_gb=8.0)
            else:
                _, batch_auto_meta = self.__auto_batch_size_smap(X_lib, X_sample, Y_lib_s, budget_gb=8.0)
            if batch_size is not None and batch_size <= 0:
                raise ValueError("batch_size must be positive, 'auto', or None.")
            self.logger.info(
                "batching[smap] policy=%s total_samples=%d batch_size=%d num_batches=%d split=%s per_sample_est=%s budget=%s selected_batch_peak_est=%s",
                "auto" if auto_batch else ("all-at-once" if batch_size is None else "manual"),
                total_samples,
                total_samples if batch_size is None else int(batch_size),
                max(1, int(math.ceil(total_samples / max(total_samples if batch_size is None else int(batch_size), 1)))),
                str((total_samples if batch_size is None else int(batch_size)) < total_samples),
                self._format_bytes(batch_auto_meta["per_sample_bytes"]),
                self._format_bytes(batch_auto_meta["budget_bytes"]),
                self._format_bytes((total_samples if batch_size is None else int(batch_size)) * max(batch_auto_meta["per_sample_bytes"], 0)),
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
        #nbrs_mask = (torch.arange(nbrs_num_max).unsqueeze(0) < nbrs_num.unsqueeze(1))
        A = torch.empty((subsample_size, max_E_Y, num_ts_Y, num_ts_X), device=self.device, dtype=self.dtype)
        for s0 in range(0, subsample_size, sample_batch_size):
            s1 = min(subsample_size, s0 + sample_batch_size)
            self.logger.debug(
                "Simplex batch [%d:%d) started (batch_queries=%d)",
                int(s0), int(s1), int(s1 - s0)
            )
            timings = {}
            t_batch = self._tic() if self._debug_enabled() else None

            try:
                with self._time_block(timings, "neighbors"):
                    weights, indices = self.__get_nbrs_indices_with_weights(
                        X_lib, X_sample[:, s0:s1, :],
                        nbrs_num, nbrs_num_max, lib_indices, smpl_indices[s0:s1],
                        exclusion_rad
                    )
            except RuntimeError as e:
                if self._is_oom(e):
                    self.logger.warning("OOM in simplex batch [%d:%d): %s", int(s0), int(s1), str(e))
                    self._hard_clear()
                raise

            I = indices

            weights_c = weights.to(self.compute_dtype)

            with self._time_block(timings, "gather"):
                Y_idx = Y_lib_shifted.to(self.compute_dtype).permute(1, 0, 2)[I]
                Y_lib_shifted_indexed = Y_idx.reshape(num_ts_X, s1 - s0, nbrs_num_max, num_ts_Y, max_E_Y)

            with self._time_block(timings, "weighted_avg"):
                A_blk = torch.einsum("xbk,xbkye->xbye", weights_c, Y_lib_shifted_indexed)
                A_blk = A_blk.permute(1, 3, 2, 0).contiguous()  # (B, E_y, n_Y, n_X)

            with self._time_block(timings, "store"):
                A[s0:s1] = A_blk.to(self.dtype)

            if self._debug_enabled():
                timings["total"] = self._toc_ms(t_batch)
                self.logger.debug(
                    "Simplex batch [%d:%d) timings: %s",
                    int(s0),
                    int(s1),
                    self._timings_summary(
                        timings,
                        order=["neighbors", "gather", "weighted_avg", "store", "total"],
                    ),
                )

            del weights, indices, I, weights_c, Y_idx, Y_lib_shifted_indexed, A_blk

        if (Y_sample_shifted is None) and return_pred:
            return A

        self.logger.debug("Computing simplex score metric")
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
        for s0 in range(0, subsample_size, sample_batch_size):
            s1 = min(subsample_size, s0 + sample_batch_size)
            B  = s1 - s0
            self.logger.debug(
                "SMAP batch [%d:%d) started (batch_queries=%d)",
                int(s0), int(s1), int(s1 - s0)
            )
            timings = {}
            t_batch = self._tic() if self._debug_enabled() else None

            # Slice the queries 
            with self._time_block(timings, "slice"):
                X_sample_b = X_sample[:, s0:s1, :]  # (num_ts_X, B, max_E_X)

            try:
                with self._time_block(timings, "local_weights"):
                    weights = self.__get_local_weights(
                        lib=X_lib, sublib=X_sample_b,
                        subset_idx=lib_indices, sample_idx=smpl_indices[s0:s1],
                        exclusion_rad=exclusion_rad, theta=theta
                    ).to(self.compute_dtype)

                with self._time_block(timings, "square"):
                    w2 = weights * weights  # (nX, B, L)

                with self._time_block(timings, "cast_xy"):
                    Xc = X_lib.to(self.compute_dtype)         # (nX, L, Ex)
                    Yc = Y_lib_shifted.to(self.compute_dtype) # (nY, L, Ey)

                with self._time_block(timings, "design"):
                    onesL = torch.ones((num_ts_X, subset_size, 1), device=self.device, dtype=self.compute_dtype)
                    Xint  = torch.cat([onesL, Xc], dim=2)     # (nX, L, Ex1)

                # XTWX: (nX, B, Ex1, Ex1)
                with self._time_block(timings, "XTWX"):
                    XTWX = torch.einsum("xli,xbl,xlj->xbij", Xint, w2, Xint)
                    if ridge and ridge > 0.0:
                        I = torch.eye(max_E_X + 1, device=self.device, dtype=self.compute_dtype)[None, None]
                        XTWX = XTWX + ridge * I

                # XTWy: (nX, B, Ex1, nY, Ey) -> flatten to (nX, B, Ex1, nY*Ey)
                with self._time_block(timings, "XTWy"):
                    XTWy = torch.einsum("xli,xbl,yle->xbiye", Xint, w2, Yc).reshape(
                        num_ts_X, B, max_E_X + 1, num_ts_Y * max_E_Y
                    )

                with self._time_block(timings, "cholesky"):
                    Lchol = torch.linalg.cholesky(XTWX)                 # (nX, B, Ex1, Ex1)
                with self._time_block(timings, "solve"):
                    beta  = torch.cholesky_solve(XTWy, Lchol)           # (nX, B, Ex1, nY*Ey)

                with self._time_block(timings, "query_design"):
                    Xq = X_sample_b.to(self.compute_dtype)              # (nX, B, Ex)
                    Xq = torch.cat(
                        [torch.ones((num_ts_X, B, 1), device=self.device, dtype=self.compute_dtype), Xq],
                        dim=2
                    )                                                   # (nX, B, Ex1)

                with self._time_block(timings, "predict"):
                    pred_flat = torch.matmul(Xq.unsqueeze(2), beta).squeeze(2)  # (nX, B, nY*Ey)

                with self._time_block(timings, "store"):
                    A = pred_flat.view(num_ts_X, B, num_ts_Y, max_E_Y).permute(1, 3, 2, 0)  # (B,Ey,nY,nX)
                    A_all[s0:s1] = A.to(self.dtype)

                if self._debug_enabled():
                    timings["total"] = self._toc_ms(t_batch)
                    self.logger.debug(
                        "SMAP batch [%d:%d) timings: %s",
                        int(s0),
                        int(s1),
                        self._timings_summary(
                            timings,
                            order=["slice", "local_weights", "square", "cast_xy", "design", "XTWX", "XTWy",
                                   "cholesky", "solve", "query_design", "predict", "store", "total"],
                        ),
                    )

                del w2, Xc, Yc, onesL, Xint, XTWX, XTWy, Lchol, beta, Xq, pred_flat, A
            except RuntimeError as e:
                if self._is_oom(e):
                    self.logger.warning("OOM in SMAP batch [%d:%d): %s", int(s0), int(s1), str(e))
                    self._hard_clear()
                raise

        if (Y_sample_shifted is None) and return_pred:
            return A_all

        self.logger.debug("Computing SMAP score metric")
        B_full = torch.permute(Y_sample_shifted, (1, 2, 0)).unsqueeze(-1).expand(subsample_size, max_E_Y, num_ts_Y, num_ts_X).to(self.compute_dtype)
        r_AB = metric_fn(A_all.to(self.compute_dtype), B_full)

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
            with self._time_block(timings, "cdist"):
                dist = self._cdist(sample, lib)  # (num_ts_X, S_blk, L)
        except RuntimeError as e:
            if self._is_oom(e):
                self._hard_clear()
            raise

        if exclusion_rad is None:
            with self._time_block(timings, "select"):
                near_dist, indices = torch.topk(dist, 1 + n_nbrs_max, largest=False, sorted=False)
                indices   = indices[:, :, :n_nbrs_max]
                near_dist = near_dist[:, :, :n_nbrs_max]
        else:
            with self._time_block(timings, "select"):
                allowed = (
                    (lib_idx[None, :] > (sample_idx[:, None] + exclusion_rad)) |
                    (lib_idx[None, :] < (sample_idx[:, None] - exclusion_rad))
                )  # (S_blk, L)
                dist = dist.masked_fill(~allowed.unsqueeze(0), float("inf"))
                near_dist, indices = torch.topk(dist, n_nbrs_max, largest=False, sorted=False)

        with self._time_block(timings, "weights"):
            weights, indices = self.__weights_from_dists(near_dist, indices, n_nbrs, n_nbrs_max)

        if self._debug_enabled():
            timings["total"] = sum(v for v in timings.values())
            self.logger.debug("Neighbor search timings: %s", self._timings_summary(timings, ["cdist", "select", "weights", "total"]))
        return weights, indices
    
    def __weights_from_dists(self, near_dist, indices, n_nbrs, n_nbrs_max):
        timings = {}
        eps = torch.finfo(near_dist.dtype).eps
        with self._time_block(timings, "exp"):
            d0 = near_dist.min(dim=2, keepdim=True).values.clamp_min(eps)
            w  = torch.exp(-near_dist / d0)
            w  = torch.where(torch.isfinite(near_dist), w, torch.zeros_like(w))

        with self._time_block(timings, "mask"):
            keep = (torch.arange(n_nbrs_max, device=w.device).unsqueeze(0) < n_nbrs.unsqueeze(1))
            w = w * keep[:, None, :].to(w.dtype)

        with self._time_block(timings, "normalize"):
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
            self.logger.debug("Neighbor weight timings: %s", self._timings_summary(timings, ["exp", "mask", "normalize", "total"]))
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
        self.logger.info("Running soft memory cleanup")
        gc.collect()
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("Soft memory cleanup finished")

    def _hard_clear(self):
        self.logger.warning("Running hard memory cleanup after OOM")
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
        self.logger.warning("Hard memory cleanup finished")

    def __auto_batch_size_smap(self, X_lib, X_sample, Y_lib_s, budget_gb=8.0):

        num_ts_X, L, max_E_X = X_lib.shape
        num_ts_Y, _, max_E_Y = Y_lib_s.shape
        S = X_sample.shape[1]

        cbytes = torch.tensor([], dtype=self.compute_dtype).element_size()
        dbytes = torch.tensor([], dtype=self.dtype).element_size()
        ex1 = int(max_E_X + 1)
        nX = int(num_ts_X)
        nY = int(num_ts_Y)
        Ey = int(max_E_Y)
        L = int(L)

        # Per-query memory model (B=1) based on tensors materialized in __smap_prediction.
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

        # Conservative headroom for temporary kernels and allocator overhead.
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
        cbytes = torch.tensor([], dtype=self.compute_dtype).element_size()  
        dbytes = torch.tensor([], dtype=self.dtype).element_size()        
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
    
    def __to_tensor(self, arr, *, dtype=None, device=None):
        dtype  = self.dtype  if dtype  is None else dtype
        device = self.device if device is None else device
        if isinstance(arr, torch.Tensor):
            return arr.to(device=device, dtype=dtype, copy=False)
        return torch.as_tensor(arr, device=device, dtype=dtype)

    def _format_bytes(self, n):
        if n is None:
            return "n/a"
        n = float(n)
        units = ["B", "KB", "MB", "GB", "TB", "PB"]
        i = 0
        while n >= 1024.0 and i < len(units) - 1:
            n /= 1024.0
            i += 1
        return f"{n:.2f}{units[i]}"

    def _tic(self):
        if self.logger.isEnabledFor(logging.DEBUG) and self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def _toc_ms(self, start):
        if self.logger.isEnabledFor(logging.DEBUG) and self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        return (time.perf_counter() - start) * 1000.0

    def _format_ms(self, ms):
        return f"{ms:.3f}ms"

    def _debug_enabled(self):
        return self.logger.isEnabledFor(logging.DEBUG)

    @contextmanager
    def _time_block(self, timings: dict, key: str):
        if not self._debug_enabled():
            yield
            return
        t0 = self._tic()
        try:
            yield
        finally:
            timings[key] = self._toc_ms(t0)

    def _timings_summary(self, timings: dict, order=None):
        keys = order if order is not None else timings.keys()
        return " ".join(
            f"{k}={self._format_ms(timings[k])}" for k in keys if k in timings
        )
