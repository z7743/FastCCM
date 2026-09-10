# ccm.py
import warnings
import os
import numpy as np
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
    smap_xtwx_precompute_bytes,
    smap_xtwy_precompute_bytes,
    batch_starts,
    resolve_simplex_target_batch_size,
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


class _NeighborLibrary:
    """
    The library side of `torch.cdist(..., "use_mm_for_euclid_dist")`, hoisted out
    of the query-batch loop.

    That mode evaluates ||q-p|| as a single matmul of augmented operands,
    [-2q, ||q||^2, 1] . [p, 1, ||p||^2]. The right-hand operand depends only on
    the library, so it is built once per call instead of once per batch.
    """

    __slots__ = ("augmented", "num_points")

    def __init__(self, augmented, num_points):
        self.augmented = augmented
        self.num_points = int(num_points)

class PairwiseCCM:
    """
    Pairwise Convergent Cross Mapping (CCM) in PyTorch.

    Public API
    ----------
    • score_matrix(...)   -> CCM scores for all target/source pairs.
    • predict_matrix(...) -> Predicted target embeddings for all pairs.
    • moran_matrix(...)   -> Moran's I of target vectors over source k-NN graphs.

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
                 verbose = 0, log_file = None, memory_budget_gb=1.0):
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
        memory_budget_gb : float, default 1.0
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

        # Reusable flat buffers for the k-NN search, released by `clean_after`.
        # `cdist` has no `out=`, so it allocates its (n_X, S, L) result every
        # batch; on CPU the first-touch page faults on that block cost more than
        # the matmul that fills it.
        self._nbr_workspace = {}

    def _predict_warning_threshold_bytes(self) -> int:
        budget_bytes = max(1, int(self.memory_budget_gb * (1024 ** 3)))
        return max(1, min(64 * 1024 * 1024, budget_bytes // 8))

    def _log_predict_output_allocation(self, pred_shape, pred_bytes: int) -> None:
        level = (
            logging.WARNING
            if int(pred_bytes) >= self._predict_warning_threshold_bytes()
            else logging.INFO
        )
        self.logger.log(
            level,
            "Predict final tensor allocation shape=%s dtype=%s approx=%s on device=%s (before CPU/NumPy transfer).",
            str(pred_shape),
            str(self.dtype),
            format_bytes(pred_bytes),
            str(self.device),
        )

    def _log_base_over_budget(self, batch_auto_meta) -> None:
        """
        Say so when the budget was unreachable before batching began.

        The base is the sampled libraries and the copies taken of them: it is
        resident for the whole call whatever batch size is chosen, so the policy
        keeps a working batch and overshoots rather than serialising for memory
        it cannot save.
        """
        if not batch_auto_meta.get("base_over_budget"):
            return
        self.logger.warning(
            "Resident tensors (%s) already exceed memory_budget_gb=%s (%s); "
            "batching cannot reclaim them, so the run will overshoot the budget. "
            "Raise memory_budget_gb, or split the target axis across calls.",
            format_bytes(batch_auto_meta["base_bytes"]),
            str(self.memory_budget_gb),
            format_bytes(batch_auto_meta["budget_bytes"]),
        )

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
            subtract_global = False,
            batch_size="auto",
            clean_after=False,
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
        subtract_global : bool, default False
            If True, fit a global linear model for each source/target pair using the
            sampled library points and subtract its metric score from the local CCM
            score. The returned tensor is therefore `local_score - global_score`.
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
        target_batch_size : {int, None, "auto"}, optional (simplex only)
            Number of target series (`n_Y`) processed per simplex reduction chunk.
            `None` keeps the original all-targets-at-once path, while
            `"auto"` applies a device-aware policy: unsplit on CUDA, calibrated
            chunking on CPU.
            When this argument is omitted, the auto policy is used by default.
        theta : float, optional (smap only)
            Local weighting strength; larger values induce steeper locality.
            Default is 1.0.
        ridge : float, optional (smap only)
            Non-negative ridge penalty added to the local linear regression. Default 0.0.
        xtwx_precompute : bool, optional (smap only)
            Whether to precompute the per-library outer-product features used to form
            SMAP normal matrices (`X^T W X`). Default is True.
        xtwy_precompute : bool, optional (smap only)
            Whether to precompute the per-library cross terms used to form SMAP
            right-hand sides (`X^T W y`). This is usually most beneficial for
            single time series or small source/target matrices. Default is False.

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
            subtract_global=subtract_global,
            batch_size=batch_size,
            **kwargs
        )

        r_AB = r_AB.to("cpu").numpy()
        self.logger.info("score_matrix completed with output shape %s", r_AB.shape)
        if clean_after:
            self._release_nbr_workspace()
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
            subtract_global = False,
            batch_size="auto",
            clean_after=False,
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
        subtract_global : bool, default False
            If True, fit a global linear model for each source/target pair using the
            sampled library points and subtract its raw prediction from the local CCM
            prediction. The returned tensor is therefore
            `local_prediction - global_prediction`.
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
        target_batch_size : {int, None, "auto"}, optional (simplex only)
            Number of target series (`n_Y`) processed per simplex reduction chunk.
            `None` keeps the original all-targets-at-once path, while
            `"auto"` applies a device-aware policy: unsplit on CUDA, calibrated
            chunking on CPU.
            When this argument is omitted, the auto policy is used by default.
        theta : float, optional (smap only)
            Local weighting strength; larger values induce steeper locality.
            Default is 1.0.
        ridge : float, optional (smap only)
            Non-negative ridge penalty added to the local linear regression. Default 0.0.
        xtwx_precompute : bool, optional (smap only)
            Whether to precompute the per-library outer-product features used to form
            SMAP normal matrices (`X^T W X`). Default is True.
        xtwy_precompute : bool, optional (smap only)
            Whether to precompute the per-library cross terms used to form SMAP
            right-hand sides (`X^T W y`). This is usually most beneficial for
            single time series or small source/target matrices. Default is False.

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
            subtract_global=subtract_global,
            batch_size=batch_size,
            **kwargs
        )

        A = A.to("cpu").numpy()
        self.logger.info("predict_matrix completed with output shape %s", A.shape)
        if clean_after:
            self._release_nbr_workspace()
            soft_clear(self.logger, self.device)
        return A

    def moran_matrix(
            self,
            X_emb,
            Y_emb=None,
            library_size=None,
            sample_size=None,
            exclusion_window=0,
            tp=0,
            seed=None,
            nbrs_num=None,
            sparse="auto",
            batch_size="auto",
            clean_after=True,
    ):
        """
        Compute Moran's I for target embeddings over source simplex-neighbor graphs.

        For each source embedding in `X_emb`, this builds a square graph on one
        sampled node set using the same indices for the simplex library and
        sample. The graph weights are the usual simplex exponential k-NN weights.
        Each target vector in `Y_emb` is then evaluated directly on that graph:

            I = (n / W.sum()) * (y_c.T @ W @ y_c) / (y_c.T @ y_c)

        Parameters
        ----------
        X_emb : list[array-like]
            Source embeddings, one per series, each with shape (T_x, E_x).
        Y_emb : list[array-like] or None, default None
            Target embeddings. If None, uses `X_emb`. If a target has multiple
            columns, Moran's I is computed independently for each column.
        library_size : int or {"auto", None}, default None
            Number of nodes in each square graph. Uses the same defaults as
            `score_matrix` over the valid `tp` window: None uses all valid
            points, and "auto" uses min(valid_points // 2, 700).
        sample_size : int or {"auto", None}, default None
            Alias for `library_size` for callers mirroring `score_matrix`.
            Because the Moran graph is square, both sizes must match when both
            are supplied.
        exclusion_window : int or None, default 0
            Temporal exclusion radius passed through to the simplex neighbor
            search. The default excludes self-neighbors.
        tp : int, default 0
            Optional target shift: graph nodes are built from X[t], while target
            values are drawn from Y[t + tp].
        seed : int or None, default None
            Seed for deterministic node sampling.
        nbrs_num : int or list[int], optional
            Number of simplex neighbors per source series. Defaults to E_x + 1.
        sparse : {bool, "auto"}, default "auto"
            Whether to build each graph as a sparse COO tensor. "auto" uses a
            dense graph for modest node counts and sparse storage for larger
            graphs.
        batch_size : int or {"auto", None}, default "auto"
            Number of graph rows processed per neighbor-search chunk.
        clean_after : bool, default True
            If True, run a light memory cleanup after returning.

        Returns
        -------
        dict[str, np.ndarray]
            Arrays with shape (E_y, n_Y, n_X): "I", "z", "p", "expected",
            and "var". The p-value is the upper-tail normal approximation
            `0.5 * erfc(z / sqrt(2))` using the standard Moran randomization
            variance with S0, S1, S2, and b2.
        """
        if Y_emb is None:
            Y_emb = X_emb
        size_spec = library_size
        if sample_size is not None:
            if library_size is None:
                size_spec = sample_size
            elif sample_size != library_size:
                raise ValueError(
                    "moran_matrix uses one square node set; sample_size must "
                    "match library_size or be None."
                )

        self.logger.info(
            "moran_matrix started (n_x=%d, n_y=%d, tp=%d, sparse=%s)",
            len(X_emb), len(Y_emb), int(tp), str(sparse)
        )

        result = self.__moran_core(
            X_emb,
            Y_emb,
            library_size=size_spec,
            exclusion_window=exclusion_window,
            tp=tp,
            seed=seed,
            nbrs_num=nbrs_num,
            sparse=sparse,
            batch_size=batch_size,
        )

        result_np = {key: value.to("cpu").numpy() for key, value in result.items()}
        self.logger.info("moran_matrix completed with output shape %s", result_np["I"].shape)
        if clean_after:
            self._release_nbr_workspace()
            soft_clear(self.logger, self.device)
        return result_np

    @torch.inference_mode()
    def __moran_core(
        self,
        X_emb,
        Y_emb,
        *,
        library_size=None,
        exclusion_window=0,
        tp=0,
        seed=None,
        nbrs_num=None,
        sparse="auto",
        batch_size="auto",
    ):
        # ---------- 1) dims / lengths ----------
        num_ts_X = len(X_emb)
        num_ts_Y = len(Y_emb)
        if num_ts_X == 0 or num_ts_Y == 0:
            raise ValueError("X_emb and Y_emb must be non-empty.")

        x_dims = tuple(int(X_emb[i].shape[-1]) for i in range(num_ts_X))
        y_dims = tuple(int(Y_emb[i].shape[-1]) for i in range(num_ts_Y))
        max_E_X = max(x_dims)
        max_E_Y = max(y_dims)
        min_len = min(
            min(y.shape[0] for y in Y_emb),
            min(x.shape[0] for x in X_emb)
        )
        tp = int(tp)
        if tp >= 0:
            valid_points = int(min_len - tp)
            x_offset = 0
            y_offset = tp
        else:
            valid_points = int(min_len + tp)
            x_offset = -tp
            y_offset = 0
        if valid_points <= 0:
            raise ValueError("Not enough points after applying tp.")

        # ---------- 2) graph size / node indices ----------
        library_size_mode = "explicit"
        if library_size is None:
            graph_size_res = int(valid_points)
            library_size_mode = "none->valid_points"
        elif library_size == "auto":
            graph_size_res = int(min(valid_points // 2, 700))
            library_size_mode = "auto"
        else:
            graph_size_res = int(library_size)
        graph_size_res = min(graph_size_res, int(valid_points))
        if graph_size_res <= 0:
            raise ValueError("library_size must resolve to a positive graph size.")

        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(int(seed))
        base_indices = self.__get_random_indices(valid_points, graph_size_res, gen)
        graph_size = int(base_indices.numel())
        if graph_size < 4:
            raise ValueError("Moran randomization variance requires at least 4 sampled nodes.")

        x_indices = base_indices + int(x_offset)
        y_indices = base_indices + int(y_offset)
        self.logger.info(
            "Moran graph_size=%d (%s, input=%s) common_len=%d valid_points=%d tp=%d",
            int(graph_size), library_size_mode, str(library_size),
            int(min_len), int(valid_points), int(tp)
        )

        # ---------- 3) simplex neighbor params ----------
        if nbrs_num is None:
            nbrs_num = torch.tensor(
                [dim + 1 for dim in x_dims],
                device=self.device,
                dtype=torch.long,
            )
        elif isinstance(nbrs_num, int):
            nbrs_num = torch.tensor(
                [nbrs_num] * num_ts_X,
                device=self.device,
                dtype=torch.long,
            )
        else:
            if len(nbrs_num) != num_ts_X:
                raise ValueError("nbrs_num must be an int or have one entry per source series.")
            nbrs_num = torch.tensor(nbrs_num, device=self.device, dtype=torch.long)
        if (nbrs_num <= 0).any():
            raise ValueError("nbrs_num values must be positive.")
        nbrs_num_max = int(nbrs_num.max().item())
        if exclusion_window is not None and nbrs_num_max >= graph_size:
            raise ValueError(
                "nbrs_num must be smaller than graph_size when temporal "
                "exclusion is enabled."
            )
        if nbrs_num_max > graph_size:
            raise ValueError(
                "nbrs_num cannot exceed the number of sampled graph nodes. "
                f"Got nbrs_num max {nbrs_num_max} and graph_size {graph_size}."
            )

        # ---------- 4) sampling ----------
        X_nodes = self.__get_random_sample(X_emb, min_len, x_indices, num_ts_X, max_E_X)
        Y_nodes = self.__get_random_sample(Y_emb, min_len, y_indices, num_ts_Y, max_E_Y)

        sample_batch_size = self.__resolve_moran_batch_size(
            X_nodes,
            graph_size=graph_size,
            nbrs_num_max=nbrs_num_max,
            batch_size=batch_size,
        )
        use_sparse = self.__resolve_moran_sparse(sparse, graph_size)
        self.logger.info(
            "Moran neighbor graph storage=%s batch_size=%d num_batches=%d nbrs_max=%d exclusion=%s",
            "sparse" if use_sparse else "dense",
            int(sample_batch_size),
            max(1, int(math.ceil(graph_size / max(sample_batch_size, 1)))),
            int(nbrs_num_max),
            str(exclusion_window),
        )

        # ---------- 5) neighbors ----------
        neighbor_indices = torch.empty(
            (num_ts_X, graph_size, nbrs_num_max),
            device=self.device,
            dtype=torch.long,
        )
        neighbor_weights = torch.empty(
            (num_ts_X, graph_size, nbrs_num_max),
            device=self.device,
            dtype=self.dtype,
        )
        lib_index = self._prepare_nbr_library(X_nodes)
        for s0 in batch_starts(self.logger, graph_size, sample_batch_size, "moran graph batches"):
            s1 = min(graph_size, s0 + sample_batch_size)
            weights, indices = self.__get_nbrs_indices_with_weights(
                X_nodes,
                X_nodes[:, s0:s1, :],
                nbrs_num,
                nbrs_num_max,
                x_indices,
                x_indices[s0:s1],
                exclusion_window,
                lib_index=lib_index,
            )
            neighbor_indices[:, s0:s1, :] = indices
            neighbor_weights[:, s0:s1, :] = weights

        # ---------- 6) Moran statistics ----------
        stat_dtype = self.__moran_compute_dtype()
        out_shape = (int(max_E_Y), int(num_ts_Y), int(num_ts_X))
        result = {
            "I": torch.full(out_shape, float("nan"), device=self.device, dtype=stat_dtype),
            "z": torch.full(out_shape, float("nan"), device=self.device, dtype=stat_dtype),
            "p": torch.full(out_shape, float("nan"), device=self.device, dtype=stat_dtype),
            "expected": torch.full(out_shape, float("nan"), device=self.device, dtype=stat_dtype),
            "var": torch.full(out_shape, float("nan"), device=self.device, dtype=stat_dtype),
        }
        Y_matrix = Y_nodes.to(dtype=stat_dtype).permute(1, 0, 2).reshape(
            graph_size, num_ts_Y * max_E_Y
        ).contiguous()

        for x_idx in range(num_ts_X):
            W = self.__build_moran_weight_matrix(
                neighbor_indices[x_idx],
                neighbor_weights[x_idx],
                dtype=stat_dtype,
                sparse=use_sparse,
            )
            stats = self.__moran_statistics_from_W(W, Y_matrix, sparse=use_sparse)
            for key, values in stats.items():
                result[key][:, :, x_idx] = values.reshape(num_ts_Y, max_E_Y).permute(1, 0)
            del W

        valid_target_dims = torch.zeros(
            (max_E_Y, num_ts_Y),
            device=self.device,
            dtype=torch.bool,
        )
        for y_idx, dim in enumerate(y_dims):
            valid_target_dims[:dim, y_idx] = True
        invalid = ~valid_target_dims[:, :, None]
        for key in result:
            result[key] = result[key].masked_fill(invalid, float("nan"))

        return result

    def __resolve_moran_batch_size(self, X_nodes, *, graph_size, nbrs_num_max, batch_size):
        if batch_size == "auto":
            Y_dummy = torch.empty((1, graph_size, 1), device=self.device, dtype=self.dtype)
            edge_bytes = int(
                X_nodes.shape[0] * graph_size * nbrs_num_max *
                (torch.tensor([], dtype=self.dtype).element_size() + 8)
            )
            resolved, _ = auto_batch_size_simplex(
                X_nodes,
                X_nodes,
                Y_dummy,
                nbrs_num_max,
                dtype=self.dtype,
                compute_dtype=self.compute_dtype,
                budget_gb=self.memory_budget_gb,
                target_batch_size=1,
                extra_base_bytes=edge_bytes,
            )
            return max(1, int(resolved))
        if batch_size is None:
            return int(graph_size)
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive, 'auto', or None.")
        return min(int(graph_size), batch_size)

    def __resolve_moran_sparse(self, sparse, graph_size):
        if sparse == "auto":
            stat_dtype = self.__moran_compute_dtype()
            dense_bytes = int(
                graph_size * graph_size *
                torch.tensor([], dtype=stat_dtype).element_size()
            )
            budget_bytes = int(self.memory_budget_gb * (1024 ** 3))
            threshold = max(16 * 1024 * 1024, budget_bytes // 4)
            return dense_bytes > threshold
        if isinstance(sparse, bool):
            return sparse
        raise ValueError("sparse must be True, False, or 'auto'.")

    def __moran_compute_dtype(self):
        if str(self.device).startswith("mps"):
            return torch.float32
        return torch.float64

    def __build_moran_weight_matrix(self, indices, weights, *, dtype, sparse):
        n = int(indices.shape[0])
        weights = weights.to(dtype=dtype)
        if sparse:
            rows = torch.arange(
                n,
                device=self.device,
                dtype=torch.long,
            ).repeat_interleave(indices.shape[1])
            cols = indices.reshape(-1).to(dtype=torch.long)
            vals = weights.reshape(-1)
            keep = torch.isfinite(vals) & (vals != 0)
            return torch.sparse_coo_tensor(
                torch.stack((rows[keep], cols[keep]), dim=0),
                vals[keep],
                size=(n, n),
                device=self.device,
                dtype=dtype,
            ).coalesce()

        W = torch.zeros((n, n), device=self.device, dtype=dtype)
        W.scatter_add_(1, indices.to(dtype=torch.long), weights)
        return W

    def __moran_statistics_from_W(self, W, Y_matrix, *, sparse):
        n = int(Y_matrix.shape[0])
        dtype = Y_matrix.dtype
        n_t = torch.tensor(float(n), device=self.device, dtype=dtype)
        y_c = Y_matrix - Y_matrix.mean(dim=0, keepdim=True)

        if sparse:
            W = W.coalesce()
            edge_idx = W.indices()
            edge_vals = W.values()
            S0 = edge_vals.sum()
            row_sum = torch.zeros(n, device=self.device, dtype=dtype)
            col_sum = torch.zeros(n, device=self.device, dtype=dtype)
            if edge_vals.numel() > 0:
                row_sum.scatter_add_(0, edge_idx[0], edge_vals)
                col_sum.scatter_add_(0, edge_idx[1], edge_vals)
            sym = (W + W.transpose(0, 1)).coalesce()
            S1 = 0.5 * sym.values().pow(2).sum()
            S2 = (row_sum + col_sum).pow(2).sum()
            Wy = torch.sparse.mm(W, y_c)
        else:
            S0 = W.sum()
            sym = W + W.transpose(0, 1)
            S1 = 0.5 * sym.pow(2).sum()
            row_sum = W.sum(dim=1)
            col_sum = W.sum(dim=0)
            S2 = (row_sum + col_sum).pow(2).sum()
            Wy = W @ y_c

        den = y_c.pow(2).sum(dim=0)
        yWy = (y_c * Wy).sum(dim=0)
        eps = torch.finfo(dtype).eps
        E_scalar = -1.0 / (n_t - 1.0)
        expected = torch.empty_like(den).fill_(E_scalar)

        valid = (S0.abs() > eps) & (den > eps)
        I = torch.full_like(den, float("nan"))
        if bool(valid.any().item()):
            I = torch.where(valid, (n_t / S0) * (yWy / den), I)

        b2 = torch.full_like(den, float("nan"))
        b2 = torch.where(den > eps, n_t * y_c.pow(4).sum(dim=0) / den.pow(2), b2)

        A = n_t * (((n_t * n_t - 3.0 * n_t + 3.0) * S1) - (n_t * S2) + (3.0 * S0 * S0))
        B = ((n_t * n_t - n_t) * S1) - (2.0 * n_t * S2) + (6.0 * S0 * S0)
        denom = (n_t - 1.0) * (n_t - 2.0) * (n_t - 3.0) * S0 * S0
        var = (A - b2 * B) / denom - (E_scalar * E_scalar)
        var = torch.where((var < 0.0) & (var > -1e-12), torch.zeros_like(var), var)
        var = torch.where(
            valid & (denom.abs() > eps) & (var >= 0.0),
            var,
            torch.full_like(var, float("nan")),
        )

        z = torch.full_like(den, float("nan"))
        z_valid = valid & (var > 0.0)
        z = torch.where(z_valid, (I - E_scalar) / torch.sqrt(var), z)
        p = torch.full_like(den, float("nan"))
        p = torch.where(z_valid, 0.5 * torch.erfc(z / math.sqrt(2.0)), p)

        return {
            "I": I,
            "z": z,
            "p": p,
            "expected": expected,
            "var": var,
        }

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
        subtract_global=False,
        batch_size=None,
        **kwargs
    ):
        metric_fn = get_metric(metric)
        self.logger.debug(
            "__ccm_core(mode=%s, method=%s, library_size=%s, sample_size=%s, exclusion_window=%s, batch_size=%s, subtract_global=%s)",
            mode, method, library_size, sample_size, exclusion_window, batch_size, subtract_global
        )
        # ---------- 1) dims / lengths ----------
        num_ts_X = len(X_lib_list)
        num_ts_Y = len(Y_lib_list)
        x_dims = tuple(int(X_lib_list[i].shape[-1]) for i in range(num_ts_X))

        max_E_X = torch.tensor([X_lib_list[i].shape[-1] for i in range(num_ts_X)], device=self.device).max().item()
        max_E_Y = torch.tensor([Y_lib_list[i].shape[-1] for i in range(num_ts_Y)], device=self.device).max().item()

        predict_output_bytes = 0
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
            target_batch_size_provided = "target_batch_size" in kwargs
            target_batch_size = kwargs.get("target_batch_size")
            if isinstance(target_batch_size, str):
                if target_batch_size != "auto":
                    raise ValueError("target_batch_size must be a positive int, None, or 'auto'.")
            elif target_batch_size is not None:
                target_batch_size = int(target_batch_size)
                if target_batch_size <= 0:
                    raise ValueError("target_batch_size must be positive, None, or 'auto'.")
            if not target_batch_size_provided:
                target_batch_size = "auto"
            global_ridge = 0.0
        elif method == "smap":
            ridge = kwargs.get("ridge", 0.0)
            theta = kwargs.get("theta", 1.0)
            xtwx_precompute = kwargs.get("xtwx_precompute", True)
            xtwy_precompute = kwargs.get("xtwy_precompute", False)
            if not isinstance(xtwx_precompute, bool):
                raise ValueError("xtwx_precompute must be a bool.")
            if not isinstance(xtwy_precompute, bool):
                raise ValueError("xtwy_precompute must be a bool.")
            global_ridge = float(ridge)
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
            predict_output_bytes = pred_bytes
            self._log_predict_output_allocation(pred_shape, pred_bytes)

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
        return_pred = (Y_smp_s is None)
        if method == "simplex":
            auto_batch = (batch_size == "auto")
            nbrs_num_max = nbrs_num.max().item()
            total_samples = int(X_sample.shape[1])
            simplex_extra_base_bytes = 0
            if mode == "score":
                simplex_extra_base_bytes = int(Y_smp_s.numel() * torch.tensor([], dtype=self.dtype).element_size())
            else:
                simplex_extra_base_bytes = int(predict_output_bytes)
            if auto_batch:
                batch_size, batch_auto_meta = auto_batch_size_simplex(
                    X_lib, X_sample, Y_lib_s, nbrs_num_max,
                    dtype=self.dtype,
                    compute_dtype=self.compute_dtype,
                    budget_gb=self.memory_budget_gb,
                    target_batch_size=target_batch_size,
                    extra_base_bytes=simplex_extra_base_bytes,
                )
            else:
                _, batch_auto_meta = auto_batch_size_simplex(
                    X_lib, X_sample, Y_lib_s, nbrs_num_max,
                    dtype=self.dtype,
                    compute_dtype=self.compute_dtype,
                    budget_gb=self.memory_budget_gb,
                    target_batch_size=target_batch_size,
                    extra_base_bytes=simplex_extra_base_bytes,
                )
            if batch_size is not None and batch_size <= 0:
                raise ValueError("batch_size must be positive, 'auto', or None.")
            selected_batch_size = total_samples if batch_size is None else int(batch_size)
            selected_target_batch_size = int(batch_auto_meta["target_batch_size"])
            selected_peak_bytes = batch_auto_meta["estimated_peak_bytes"] if auto_batch else (
                batch_auto_meta["base_bytes"] + selected_batch_size * max(batch_auto_meta["per_sample_bytes"], 0)
            )
            self.logger.info(
                "Batching policy=%s total_samples=%d batch_size=%d num_batches=%d split=%s target_batch_size=%d target_split=%s base_est=%s per_sample_est=%s budget=%s selected_batch_peak_est=%s",
                "auto" if auto_batch else ("all-at-once" if batch_size is None else "manual"),
                total_samples,
                selected_batch_size,
                max(1, int(math.ceil(total_samples / max(selected_batch_size, 1)))),
                str(selected_batch_size < total_samples),
                selected_target_batch_size,
                str(selected_target_batch_size < int(Y_lib_s.shape[0])),
                format_bytes(batch_auto_meta["base_bytes"]),
                format_bytes(batch_auto_meta["per_sample_bytes"]),
                format_bytes(batch_auto_meta["budget_bytes"]),
                format_bytes(selected_peak_bytes),
            )
            self._log_base_over_budget(batch_auto_meta)
            out = self.__simplex_prediction(
                lib_indices, smpl_indices,
                X_lib, X_sample, Y_lib_s, Y_smp_s,
                exclusion_window, nbrs_num, metric_fn=metric_fn,
                return_pred=return_pred, sample_batch_size=batch_size,
                nbrs_num_max=int(nbrs_num_max),
                target_batch_size=selected_target_batch_size,
            )


        else:
            auto_batch = (batch_size == "auto")
            total_samples = int(X_sample.shape[1])
            if auto_batch:
                batch_size, batch_auto_meta = auto_batch_size_smap(
                    X_lib, X_sample, Y_lib_s,
                    dtype=self.dtype,
                    compute_dtype=self.compute_dtype,
                    budget_gb=self.memory_budget_gb,
                    xtwx_precompute=xtwx_precompute,
                    xtwy_precompute=xtwy_precompute,
                )
            else:
                _, batch_auto_meta = auto_batch_size_smap(
                    X_lib, X_sample, Y_lib_s,
                    dtype=self.dtype,
                    compute_dtype=self.compute_dtype,
                    budget_gb=self.memory_budget_gb,
                    xtwx_precompute=xtwx_precompute,
                    xtwy_precompute=xtwy_precompute,
                )
            if batch_size is not None and batch_size <= 0:
                raise ValueError("batch_size must be positive, 'auto', or None.")
            selected_batch_size = total_samples if batch_size is None else int(batch_size)
            selected_peak_bytes = batch_auto_meta["estimated_peak_bytes"] if auto_batch else (
                batch_auto_meta["base_bytes"] + selected_batch_size * max(batch_auto_meta["per_sample_bytes"], 0)
            )
            self.logger.info(
                "Batching policy=%s total_samples=%d batch_size=%d num_batches=%d split=%s base_est=%s per_sample_est=%s budget=%s selected_batch_peak_est=%s",
                "auto" if auto_batch else ("all-at-once" if batch_size is None else "manual"),
                total_samples,
                selected_batch_size,
                max(1, int(math.ceil(total_samples / max(selected_batch_size, 1)))),
                str(selected_batch_size < total_samples),
                format_bytes(batch_auto_meta["base_bytes"]),
                format_bytes(batch_auto_meta["per_sample_bytes"]),
                format_bytes(batch_auto_meta["budget_bytes"]),
                format_bytes(selected_peak_bytes),
            )
            self._log_base_over_budget(batch_auto_meta)
            self.logger.info(
                "SMAP config theta=%s ridge=%s xtwx_precompute=%s xtwy_precompute=%s",
                str(theta),
                str(ridge),
                str(xtwx_precompute),
                str(xtwy_precompute),
            )


            out = self.__smap_prediction(
                lib_indices, smpl_indices,
                X_lib, X_sample, Y_lib_s, Y_smp_s,
                exclusion_window, theta, metric_fn=metric_fn,
                return_pred=return_pred,
                sample_batch_size=batch_size,
                ridge=ridge,
                xtwx_precompute=xtwx_precompute,
                xtwy_precompute=xtwy_precompute,
            )


        if subtract_global:
            self.logger.info("Subtracting global linear baseline from %s output", mode)
            global_out = self.__global_linear_output(
                X_lib,
                X_sample,
                Y_lib_s,
                Y_smp_s,
                x_dims=x_dims,
                metric_fn=metric_fn,
                ridge=global_ridge,
                sample_batch_size=batch_size,
                return_pred=return_pred,
            )
            out = out - global_out.to(device=out.device, dtype=out.dtype)

        return out

    @torch.inference_mode()
    def __simplex_prediction(self, lib_indices, smpl_indices,
                              X_lib, X_sample, Y_lib_shifted, Y_sample_shifted, 
                              exclusion_rad, nbrs_num, metric_fn, return_pred=False, sample_batch_size=None,
                              nbrs_num_max=None, target_batch_size=None):
        num_ts_X = X_lib.shape[0]
        num_ts_Y = Y_lib_shifted.shape[0]
        max_E_Y = Y_lib_shifted.shape[2]
        subsample_size = X_sample.shape[1]

        if (sample_batch_size is None) or (sample_batch_size >= subsample_size):
            sample_batch_size = subsample_size
        target_batch_size = resolve_simplex_target_batch_size(num_ts_Y, target_batch_size)

        if nbrs_num_max is None:
            nbrs_num_max = int(nbrs_num.max().item())

        lib_index = self._prepare_nbr_library(X_lib)
        self.logger.debug(
            "Entering simplex backend (queries=%d, library_points=%d, num_sources=%d, num_targets=%d, Ey=%d, nbrs_max=%d, batch_size=%d, num_batches=%d, target_batch_size=%d, num_target_batches=%d, exclusion=%s)",
            int(subsample_size),
            int(X_lib.shape[1]),
            int(num_ts_X),
            int(num_ts_Y),
            int(max_E_Y),
            int(nbrs_num_max),
            int(sample_batch_size),
            int(math.ceil(subsample_size / sample_batch_size)),
            int(target_batch_size),
            int(math.ceil(num_ts_Y / target_batch_size)),
            str(exclusion_rad),
        )
        stream_kind = get_streaming_metric_kind(metric_fn) if (not return_pred) else None
        stream_state = None
        if stream_kind is not None:
            stream_state = stream_metric_state_init(
                stream_kind,
                max_E_Y,
                num_ts_Y,
                num_ts_X,
                device=self.device,
                dtype=self.compute_dtype,
                shared_target=True,
            )

        # Keep full output off accelerator in score mode so device memory tracks batch size.
        out_device = self.device if ((Y_sample_shifted is None) and return_pred) else "cpu"
        # Flatten target features once so per-batch gathers can use index_select on a
        # dense 2D table instead of slower advanced indexing over a 3D tensor.
        Y_lib_flat = Y_lib_shifted.to(self.compute_dtype).permute(1, 0, 2).reshape(
            X_lib.shape[1], num_ts_Y * max_E_Y
        ).contiguous()
        target_ranges = tuple(
            (y0, min(num_ts_Y, y0 + target_batch_size))
            for y0 in range(0, num_ts_Y, target_batch_size)
        )
        max_target_block_width = int(target_batch_size * max_E_Y)
        # Only the gather + `bmm` path needs the k-fold staging buffers.
        if self._use_fused_reduce(nbrs_num_max, max_target_block_width):
            gather_buf = None
            reduce_buf = None
            bag_offsets = torch.arange(
                0, num_ts_X * sample_batch_size * nbrs_num_max, nbrs_num_max,
                device=Y_lib_flat.device, dtype=torch.long,
            )
        else:
            bag_offsets = None
            gather_buf = torch.empty(
                (num_ts_X * sample_batch_size * nbrs_num_max, max_target_block_width),
                device=Y_lib_flat.device,
                dtype=self.compute_dtype,
            )
            reduce_buf = torch.empty(
                (num_ts_X * sample_batch_size, 1, max_target_block_width),
                device=Y_lib_flat.device,
                dtype=self.compute_dtype,
            )
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
                        exclusion_rad, lib_index=lib_index
                    )
            except RuntimeError as e:
                if is_oom_error(e):
                    self.logger.warning("OOM in simplex batch [%d:%d): %s", int(s0), int(s1), str(e))
                    hard_clear(self.logger, self.device)
                raise

            weights_c = weights.to(self.compute_dtype)
            batch_queries = s1 - s0
            flat_indices = indices.reshape(-1)
            weights_rows = weights_c.reshape(num_ts_X * batch_queries, 1, nbrs_num_max)
            weights_flat = weights_c.reshape(-1)
            flat_count = int(flat_indices.numel())
            batch_rows = int(num_ts_X * batch_queries)
            Y_sample_batch_view = None
            if stream_kind is not None:
                Y_sample_batch_view = Y_sample_shifted[:, s0:s1, :].permute(1, 2, 0).to(
                    device=self.device, dtype=self.compute_dtype
                ).contiguous()
            gather_ms = 0.0
            weighted_avg_ms = 0.0
            metric_ms = 0.0
            store_ms = 0.0
            for y0, y1 in target_ranges:
                y_width = (y1 - y0) * max_E_Y

                fused = bag_offsets is not None and self._use_fused_reduce(
                    nbrs_num_max, y_width
                )

                t_part = tic(self.logger, self.device) if self._debug_enabled() else None
                Y_src = Y_lib_flat[:, y0 * max_E_Y:y1 * max_E_Y]
                if fused:
                    Y_idx = None  # folded into the weighted sum below
                elif y_width == max_target_block_width:
                    Y_idx = torch.index_select(
                        Y_src,
                        0,
                        flat_indices,
                        out=gather_buf[:flat_count],
                    ).reshape(batch_rows, nbrs_num_max, y_width)
                else:
                    Y_idx = torch.index_select(
                        Y_src,
                        0,
                        flat_indices,
                    ).reshape(batch_rows, nbrs_num_max, y_width)
                if t_part is not None:
                    gather_ms += toc_ms(self.logger, self.device, t_part)

                t_part = tic(self.logger, self.device) if self._debug_enabled() else None
                if fused:
                    A_y = torch.nn.functional.embedding_bag(
                        flat_indices,
                        Y_src,
                        bag_offsets[:batch_rows],
                        mode="sum",
                        per_sample_weights=weights_flat,
                    ).reshape(num_ts_X, batch_queries, y1 - y0, max_E_Y)
                elif y_width == max_target_block_width:
                    A_y = torch.bmm(
                        weights_rows,
                        Y_idx,
                        out=reduce_buf[:batch_rows],
                    ).reshape(num_ts_X, batch_queries, y1 - y0, max_E_Y)
                else:
                    A_y = torch.bmm(weights_rows, Y_idx).reshape(
                        num_ts_X, batch_queries, y1 - y0, max_E_Y
                    )
                A_y = A_y.permute(1, 3, 2, 0).contiguous()  # (B, E_y, y_blk, n_X)
                if t_part is not None:
                    weighted_avg_ms += toc_ms(self.logger, self.device, t_part)

                if stream_kind is not None:
                    t_part = tic(self.logger, self.device) if self._debug_enabled() else None
                    B_y = Y_sample_batch_view[:, :, y0:y1].unsqueeze(-1).expand(
                        batch_queries, max_E_Y, y1 - y0, num_ts_X
                    )
                    stream_metric_state_update(
                        stream_kind,
                        stream_state,
                        A_y.to(device=self.device, dtype=self.compute_dtype),
                        B_y,
                        y_start=y0,
                        count_samples=(y0 == 0),
                    )
                    if t_part is not None:
                        metric_ms += toc_ms(self.logger, self.device, t_part)
                    del B_y
                else:
                    t_part = tic(self.logger, self.device) if self._debug_enabled() else None
                    A[s0:s1, :, y0:y1, :] = A_y.to(out_device, dtype=self.dtype)
                    if t_part is not None:
                        store_ms += toc_ms(self.logger, self.device, t_part)

                del Y_idx, A_y

            if self._debug_enabled():
                timings["gather"] = gather_ms
                timings["weighted_avg"] = weighted_avg_ms
                if stream_kind is not None:
                    timings["metric"] = metric_ms
                else:
                    timings["store"] = store_ms

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

            del weights, indices, weights_c, flat_indices, weights_rows, weights_flat, Y_sample_batch_view

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
                      sample_batch_size=None, ridge=0.0,
                      xtwx_precompute=True, xtwy_precompute=False):
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
                stream_kind, max_E_Y, num_ts_Y, num_ts_X, device=self.device, dtype=self.compute_dtype
            )

        # Keep full output off accelerator in score mode so device memory tracks batch size.
        out_device = self.device if ((Y_sample_shifted is None) and return_pred) else "cpu"
        A_all = None if stream_kind is not None else torch.empty((subsample_size, max_E_Y, num_ts_Y, num_ts_X), device=out_device, dtype=self.dtype)

        if sample_batch_size is None or sample_batch_size >= subsample_size:
            sample_batch_size = subsample_size

        self.logger.debug(
            "Entering smap backend (queries=%d, library_points=%d, num_sources=%d, num_targets=%d, Ex=%d, Ey=%d, batch_size=%d, num_batches=%d, exclusion=%s, theta=%s, ridge=%s, xtwx_precompute=%s, xtwy_precompute=%s)",
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
            str(xtwx_precompute),
            str(xtwy_precompute),
        )

        Xc = X_lib.to(self.compute_dtype)                 # (nX, L, Ex)
        Yc = Y_lib_shifted.to(self.compute_dtype)         # (nY, L, Ey)
        onesL = torch.ones((num_ts_X, subset_size, 1), device=self.device, dtype=self.compute_dtype)
        Xint = torch.cat([onesL, Xc], dim=2)             # (nX, L, Ex1)
        Xint_t = Xint.transpose(1, 2).contiguous()       # (nX, Ex1, L)
        Yc_flat = Yc.permute(1, 0, 2).reshape(subset_size, num_ts_Y * max_E_Y).contiguous()
        ex1 = int(max_E_X + 1)

        XTWX_features = None
        if xtwx_precompute:
            self.logger.debug(
                "Building XTWX precompute features (~%s)",
                format_bytes(smap_xtwx_precompute_bytes(X_lib, compute_dtype=self.compute_dtype)),
            )
            XTWX_features = torch.matmul(Xint.unsqueeze(-1), Xint.unsqueeze(-2)).reshape(
                num_ts_X, subset_size, ex1 * ex1
            )
        XTWy_features = None
        if xtwy_precompute:
            self.logger.debug(
                "Building XTWy precompute features (~%s)",
                format_bytes(smap_xtwy_precompute_bytes(X_lib, Y_lib_shifted, compute_dtype=self.compute_dtype)),
            )
            XTWy_features = torch.mul(
                Xint.unsqueeze(-1),
                Yc_flat.unsqueeze(0).unsqueeze(2),
            ).reshape(num_ts_X, subset_size, ex1 * num_ts_Y * max_E_Y)
        I = None
        if ridge and ridge > 0.0:
            I = torch.eye(ex1, device=self.device, dtype=self.compute_dtype)[None, None]
        weighted_design_buf = None
        tail_batch_size = subsample_size % sample_batch_size
        weighted_design_tail_buf = None
        if not xtwy_precompute:
            weighted_design_buf = torch.empty(
                (num_ts_X, sample_batch_size, ex1, subset_size),
                device=self.device,
                dtype=self.compute_dtype,
            )
        if (not xtwy_precompute) and tail_batch_size:
            weighted_design_tail_buf = torch.empty(
                (num_ts_X, tail_batch_size, ex1, subset_size),
                device=self.device,
                dtype=self.compute_dtype,
            )

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
                    )

                with time_block(self.logger, self.device, timings, "square"):
                    weights.square_()  # (nX, B, L) in-place; avoid extra w2 allocation

                # XTWX: (nX, B, Ex1, Ex1)
                with time_block(self.logger, self.device, timings, "XTWX"):
                    if xtwx_precompute:
                        XTWX = torch.matmul(weights, XTWX_features).reshape(num_ts_X, B, ex1, ex1)
                    else:
                        XTWX = torch.einsum("xli,xbl,xlj->xbij", Xint, weights, Xint)
                    if I is not None:
                        XTWX = XTWX + ridge * I

                # XTWy: (nX, B, Ex1, nY, Ey) -> flatten to (nX, B, Ex1, nY*Ey)
                with time_block(self.logger, self.device, timings, "XTWy"):
                    if xtwy_precompute:
                        XTWy = torch.matmul(weights, XTWy_features).reshape(num_ts_X, B, ex1, num_ts_Y * max_E_Y)
                    else:
                        weighted_design = weighted_design_buf if B == sample_batch_size else weighted_design_tail_buf
                        torch.mul(weights.unsqueeze(-2), Xint_t.unsqueeze(1), out=weighted_design)
                        XTWy = torch.matmul(weighted_design, Yc_flat)

                with time_block(self.logger, self.device, timings, "solve"):
                    beta = torch.linalg.solve(XTWX, XTWy)               # (nX, B, Ex1, nY*Ey)

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
                        B_blk = torch.permute(Y_sample_shifted[:, s0:s1, :], (1, 2, 0)).to(device=self.device, dtype=self.compute_dtype)[:, :, :, None] \
                            .expand(B, max_E_Y, num_ts_Y, num_ts_X)
                        stream_metric_state_update(
                            stream_kind,
                            stream_state,
                            A.to(device=self.device, dtype=self.compute_dtype),
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
                                   "solve", "query_design", "predict", "metric", "store", "total"],
                        ),
                    )

                del weights, XTWX, XTWy, beta, Xq, pred_flat, A
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
        if dim > 0 and self.__can_stack_sample_block(X, min_len, max_E):
            if isinstance(X, torch.Tensor):
                stacked = X[:, -min_len:, :].to(device=self.device, dtype=self.dtype, copy=False)
            elif isinstance(X, np.ndarray):
                stacked = torch.as_tensor(X[:, -min_len:, :], device=self.device, dtype=self.dtype)
            else:
                first = X[0]
                if isinstance(first, torch.Tensor):
                    stacked = torch.stack(
                        [Xi[-min_len:] for Xi in X],
                        dim=0,
                    ).to(device=self.device, dtype=self.dtype, copy=False)
                else:
                    stacked_np = np.stack(
                        [np.asarray(Xi[-min_len:]) for Xi in X],
                        axis=0,
                    )
                    stacked = torch.as_tensor(stacked_np, device=self.device, dtype=self.dtype)
            return torch.index_select(stacked, 1, indices)

        X_buf = torch.zeros((dim, indices.shape[0], max_E),device=self.device, dtype=self.dtype)

        for i in range(dim):
            Xi_src = X[i]
            if isinstance(Xi_src, torch.Tensor):
                Xi = Xi_src.to(device=self.device, dtype=self.dtype, copy=False)[-min_len:]
            else:
                Xi = torch.as_tensor(Xi_src[-min_len:], device=self.device, dtype=self.dtype)
            X_buf[i, :, :X[i].shape[-1]] = Xi[indices]

        return X_buf

    def __can_stack_sample_block(self, X, min_len, max_E):
        if len(X) == 0:
            return False
        if isinstance(X, torch.Tensor) or isinstance(X, np.ndarray):
            if X.ndim != 3:
                return False
            return (
                int(X.shape[0]) > 0 and
                int(X.shape[1]) >= int(min_len) and
                int(X.shape[2]) == int(max_E)
            )
        first = X[0]
        first_is_tensor = isinstance(first, torch.Tensor)
        first_width = int(first.shape[-1])
        if first_width != int(max_E):
            return False
        if int(first.shape[0]) < int(min_len):
            return False

        for Xi in X[1:]:
            if isinstance(Xi, torch.Tensor) != first_is_tensor:
                return False
            if int(Xi.shape[-1]) != first_width:
                return False
            if int(Xi.shape[0]) < int(min_len):
                return False
        return True

    def __global_linear_output(
        self,
        X_lib,
        X_sample,
        Y_lib_shifted,
        Y_sample_shifted,
        *,
        x_dims,
        metric_fn,
        ridge=0.0,
        sample_batch_size=None,
        return_pred=False,
    ):
        num_ts_X = X_lib.shape[0]
        num_ts_Y = Y_lib_shifted.shape[0]
        subset_size = X_lib.shape[1]
        subsample_size = X_sample.shape[1]
        max_E_Y = Y_lib_shifted.shape[2]
        compute_dtype = self.__linear_compute_dtype()

        if sample_batch_size is None or sample_batch_size >= subsample_size:
            sample_batch_size = subsample_size

        Y_flat = Y_lib_shifted.to(compute_dtype).permute(1, 0, 2).reshape(
            subset_size, num_ts_Y * max_E_Y
        ).contiguous()
        beta_by_source = []
        for x_idx, ex in enumerate(x_dims):
            X_design = self.__with_intercept(
                X_lib[x_idx, :, :ex].to(device=self.device, dtype=compute_dtype, copy=False)
            )
            beta_by_source.append(self.__solve_global_linear_beta(X_design, Y_flat, ridge=ridge))

        stream_kind = get_streaming_metric_kind(metric_fn) if (not return_pred) else None
        stream_state = None
        out_device = self.device if return_pred else "cpu"
        out = None
        if stream_kind is not None:
            stream_state = stream_metric_state_init(
                stream_kind, max_E_Y, num_ts_Y, num_ts_X, device=self.device, dtype=compute_dtype
            )
        else:
            out = torch.empty(
                (subsample_size, max_E_Y, num_ts_Y, num_ts_X),
                device=out_device,
                dtype=self.dtype,
            )

        for s0 in batch_starts(self.logger, subsample_size, sample_batch_size, "global linear batches"):
            s1 = min(subsample_size, s0 + sample_batch_size)
            batch_queries = s1 - s0
            A_blk = None
            if stream_kind is not None:
                A_blk = torch.empty(
                    (batch_queries, max_E_Y, num_ts_Y, num_ts_X),
                    device=self.device,
                    dtype=compute_dtype,
                )

            for x_idx, ex in enumerate(x_dims):
                Xq = self.__with_intercept(
                    X_sample[x_idx, s0:s1, :ex].to(device=self.device, dtype=compute_dtype, copy=False)
                )
                pred_flat = Xq @ beta_by_source[x_idx]
                pred = pred_flat.view(batch_queries, num_ts_Y, max_E_Y).permute(0, 2, 1).contiguous()
                if stream_kind is not None:
                    A_blk[:, :, :, x_idx] = pred.to(device=self.device, dtype=compute_dtype)
                else:
                    out[s0:s1, :, :, x_idx] = pred.to(device=out_device, dtype=self.dtype)

            if stream_kind is not None:
                B_blk = torch.permute(Y_sample_shifted[:, s0:s1, :], (1, 2, 0)).to(
                    device=self.device, dtype=compute_dtype
                )[:, :, :, None].expand(batch_queries, max_E_Y, num_ts_Y, num_ts_X)
                stream_metric_state_update(stream_kind, stream_state, A_blk, B_blk)

        if stream_kind is not None:
            return stream_metric_state_finalize(stream_kind, stream_state)

        if return_pred:
            return out

        B_full = torch.permute(Y_sample_shifted, (1, 2, 0)).unsqueeze(-1).expand(
            subsample_size, max_E_Y, num_ts_Y, num_ts_X
        ).to(device=out.device, dtype=compute_dtype)
        return metric_fn(out.to(dtype=compute_dtype), B_full)

    def __solve_global_linear_beta(self, X_design, Y_flat, ridge=0.0):
        XtX = X_design.transpose(0, 1) @ X_design
        XTy = X_design.transpose(0, 1) @ Y_flat
        eye = torch.eye(XtX.shape[0], device=XtX.device, dtype=XtX.dtype)
        base = max(float(ridge), 0.0)
        jitter = tuple(base + extra for extra in (1e-8, 1e-6, 1e-4))
        for lam in jitter:
            try:
                return torch.linalg.solve(XtX + lam * eye, XTy)
            except RuntimeError:
                continue
        return torch.linalg.pinv(X_design) @ Y_flat

    def __with_intercept(self, X):
        ones = torch.ones((X.shape[0], 1), device=X.device, dtype=X.dtype)
        return torch.cat([ones, X], dim=1)

    def __linear_compute_dtype(self):
        return self._promoted_compute_dtype()

    def _promoted_compute_dtype(self):
        """compute_dtype, with fp16 promoted on CPU where it is slow and unstable."""
        if self.device.startswith("cpu") and self.compute_dtype == torch.float16:
            return torch.float32
        return self.compute_dtype

    def _use_workspace_neighbors(self):
        """
        Whether to run the neighbor search through reusable buffers instead of
        letting `cdist`/`topk` allocate per batch.

        CPU avoids first-touch page faults on large result blocks; CUDA avoids
        rebuilding the library operand and reuses intermediate allocations across
        query batches. MPS retains the native path because its `out=` support
        differs from CPU and CUDA.
        """
        return self.device.startswith(("cpu", "cuda"))

    # `bmm` on (rows, 1, k) x (rows, k, N) leaves its packed kernel once the
    # per-row right operand stops fitting the blocking it uses -- measured at
    # k*N ~ 450 for k in 2..21 -- and past that point it is ~50x slower. Stay
    # under it with margin.
    _BMM_FAST_ELEMS = 384

    def _use_fused_reduce(self, nbrs_num_max, y_width):
        """
        Whether to run the simplex weighted average through `embedding_bag`
        instead of gather + `bmm`.

        The fused path folds the neighbor gather into the weighted sum, so it
        never materializes the (rows, k, y_width) block. Below the threshold
        `bmm` is still the faster of the two.

        CUDA has no width cliff, but skipping the k-fold intermediate pays
        there too, and the target axis is left unsplit on CUDA so the width
        comes from the target count rather than from E_y. MPS keeps `bmm`;
        it has not been measured.
        """
        if not self.device.startswith(("cpu", "cuda")):
            return False
        return int(nbrs_num_max) * int(y_width) > self._BMM_FAST_ELEMS

    def _release_nbr_workspace(self):
        self._nbr_workspace = {}

    def __workspace(self, name, shape, dtype):
        """Persistent flat buffer for `name`, viewed as `shape`."""
        numel = 1
        for dim in shape:
            numel *= int(dim)
        buf = self._nbr_workspace.get(name)
        if buf is None or buf.numel() < numel or buf.dtype != dtype:
            self._nbr_workspace.pop(name, None)
            try:
                buf = torch.empty(numel, device=self.device, dtype=dtype)
            except RuntimeError as e:
                if is_oom_error(e):
                    self._nbr_workspace = {}
                    hard_clear(self.logger, self.device)
                raise
            self._nbr_workspace[name] = buf
        return buf[:numel].view(*shape)

    def _prepare_nbr_library(self, lib):
        """Build the library-side augmented operand once per call."""
        if not self._use_workspace_neighbors():
            return None
        lib_c = self.__to_tensor(lib, dtype=self._promoted_compute_dtype())
        sq = lib_c.pow(2).sum(-1, True)
        pad = torch.ones_like(sq)
        return _NeighborLibrary(torch.cat([lib_c, pad, sq], -1), lib_c.shape[1])

    def __get_nbrs_indices_with_weights(
        self, lib, sample, n_nbrs, n_nbrs_max, lib_idx, sample_idx, exclusion_rad,
        lib_index=None,
    ):
        """
        Weights and library indices of each query's k nearest neighbors.

        Note: the returned tensors alias per-instance workspaces and stay valid
        only until the next call, which both callers respect by consuming them
        before advancing to the next query batch.
        """
        timings = {}
        try:
            with time_block(self.logger, self.device, timings, "cdist"):
                if lib_index is None and self._use_workspace_neighbors():
                    lib_index = self._prepare_nbr_library(lib)
                if lib_index is None:
                    dist = self._cdist(sample, lib)  # (num_ts_X, S_blk, L)
                else:
                    dist = self.__squared_euclidean_dist(sample, lib_index)
        except RuntimeError as e:
            if is_oom_error(e):
                hard_clear(self.logger, self.device)
            raise

        with time_block(self.logger, self.device, timings, "select"):
            if exclusion_rad is not None and not self.__exclude_narrow(dist, lib_idx, sample_idx, exclusion_rad):
                # Keep the broadcast intermediates boolean. Subtracting int64
                # indices would materialize an 8-byte (samples x library) tensor.
                allowed = (
                    (lib_idx[None, :] > (sample_idx[:, None] + exclusion_rad)) |
                    (lib_idx[None, :] < (sample_idx[:, None] - exclusion_rad))
                )
                dist.masked_fill_(~allowed.unsqueeze(0), float("inf"))
            if lib_index is None:
                near_dist, indices = torch.topk(dist, n_nbrs_max, largest=False, sorted=False)
            else:
                chunk = self._prefilter_chunk(n_nbrs_max, dist.shape[2])
                if chunk:
                    near_dist, indices = self.__prefilter_topk(dist, n_nbrs_max, chunk)
                else:
                    sel = (dist.shape[0], dist.shape[1], n_nbrs_max)
                    near_dist = self.__workspace("near_dist", sel, dist.dtype)
                    indices = self.__workspace("indices", sel, torch.long)
                    torch.topk(dist, n_nbrs_max, dim=2, largest=False, sorted=False,
                               out=(near_dist, indices))
                near_dist.clamp_min_(0).sqrt_()

        with time_block(self.logger, self.device, timings, "weights"):
            weights, indices = self.__weights_from_dists(near_dist, indices, n_nbrs, n_nbrs_max)

        if self._debug_enabled():
            timings["total"] = sum(v for v in timings.values())
            self.logger.debug("Neighbor search timings: %s", timings_summary(timings, ["cdist", "select", "weights", "total"]))
        return weights, indices

    # Random-access cost of the scatter path is roughly one cache line per
    # touched element, so it only beats a streaming pass over the block while
    # the window stays much narrower than the library.
    _EXCLUSION_SCATTER_RATIO = 8

    def __exclude_narrow(self, dist, lib_idx, sample_idx, exclusion_rad):
        """
        Mark the excluded library columns of each query without touching the
        rest of the distance block.

        Only the (2r+1) time steps around a query can be excluded, so the
        columns are found through a time-index -> library-column table instead
        of comparing every query against every library point. Returns False
        when the window is wide enough that a full masked_fill_ is cheaper.
        """
        rad = int(exclusion_rad)
        num_lib = int(dist.shape[2])
        width = 2 * rad + 1
        if rad < 0 or width * self._EXCLUSION_SCATTER_RATIO >= num_lib:
            return False

        num_points = int(lib_idx.max()) + 1
        # Padded on both sides so the query-centred window never needs a bounds
        # check: absent time steps stay at -1, and query indices past the end of
        # the library (prediction mode) clamp onto the guaranteed -1 tail.
        table = torch.full((num_points + 2 * rad + 1,), -1, dtype=torch.long, device=dist.device)
        table[rad:rad + num_points].scatter_(
            0, lib_idx, torch.arange(num_lib, device=dist.device)
        )
        window = torch.arange(width, device=dist.device)
        cols = table[(sample_idx[:, None] + window[None, :]).clamp_(max=table.shape[0] - 1)]
        present = cols >= 0

        # Queries whose window holds no library point must be left alone; for the
        # rest, empty slots are folded onto a column that is excluded anyway so
        # the duplicated scatter writes stay harmless.
        first = cols.masked_fill(~present, num_lib).amin(dim=1)  # (S,)
        has_any = first < num_lib
        cols = torch.where(present, cols, (first * has_any)[:, None])

        idx = cols.unsqueeze(0).expand(dist.shape[0], -1, -1)
        if bool(has_any.all()):
            dist.scatter_(2, idx, float("inf"))
        else:
            keep = dist.gather(2, idx)
            inf = torch.tensor(float("inf"), device=dist.device, dtype=dist.dtype)
            dist.scatter_(2, idx, torch.where(has_any[None, :, None], inf, keep))
        return True

    # `topk` over the library axis is compute-bound rather than bandwidth-bound:
    # on CUDA it runs ~9-22x slower than a plain reduction over the same block.
    # Ranking chunk minima first replaces most of that scan with `amin`.
    _PREFILTER_CHUNK = 128
    _PREFILTER_MIN_LIB = 4096

    def _prefilter_chunk(self, n_nbrs_max, num_lib):
        """
        Chunk width for the prefiltered neighbor selection, or 0 for `topk`.

        The rescan covers k of the chunks, so the saving only survives while
        those are a small share of the block -- below `_PREFILTER_MIN_LIB` the
        candidate gather costs more than the skipped scan saves. CPU keeps
        `topk`, where the two run much closer together and this loses.
        """
        if not self.device.startswith("cuda"):
            return 0
        if int(num_lib) < self._PREFILTER_MIN_LIB:
            return 0
        if int(num_lib) // self._PREFILTER_CHUNK < 2 * int(n_nbrs_max):
            return 0
        return self._PREFILTER_CHUNK

    def __prefilter_topk(self, dist, k, chunk):
        """
        Exact k smallest per row, found through chunk minima.

        Each of the k smallest values sits in a chunk whose minimum is itself
        among the k smallest chunk minima -- otherwise more than k values would
        undercut it -- so rescanning only those k chunks loses nothing. Ties
        break arbitrarily, as they do in `topk`.
        """
        n_x, n_q, num_lib = dist.shape
        n_chunks = num_lib // chunk
        head = dist[..., : n_chunks * chunk].view(n_x, n_q, n_chunks, chunk)

        candidates = head.amin(-1).topk(k, dim=2, largest=False, sorted=False).indices
        scanned = head.gather(
            2, candidates[..., None].expand(n_x, n_q, k, chunk)
        ).reshape(n_x, n_q, k * chunk)
        values, flat = scanned.topk(k, dim=2, largest=False, sorted=False)
        cols = candidates.gather(
            2, flat.div(chunk, rounding_mode="floor")
        ) * chunk + flat.remainder(chunk)

        rest = num_lib - n_chunks * chunk
        if rest:
            # Columns past the last whole chunk, merged the same way.
            tail_k = min(k, rest)
            tail_v, tail_i = dist[..., n_chunks * chunk:].topk(
                tail_k, dim=2, largest=False, sorted=False
            )
            values = torch.cat([values, tail_v], 2)
            cols = torch.cat([cols, tail_i + n_chunks * chunk], 2)
            values, pick = values.topk(k, dim=2, largest=False, sorted=False)
            cols = cols.gather(2, pick)
        return values, cols

    def __squared_euclidean_dist(self, sample, lib_index):
        """
        Squared `cdist(..., "use_mm_for_euclid_dist")` in a reusable buffer.

        Rank raw squared distances and apply clamp and square root only to the
        selected neighbors. Float32 cancellation can make near-zero squared
        distances slightly negative, so this may choose a different member of a
        zero-distance tie than the native path. Square-root rounding can likewise
        create ties after selection.
        """
        comp = self._promoted_compute_dtype()
        q = self.__to_tensor(sample, dtype=comp)
        sq = q.pow(2).sum(-1, True)
        pad = torch.ones_like(sq)
        augmented = torch.cat([q.mul(-2), sq, pad], -1)
        dist = self.__workspace(
            "dist", (q.shape[0], q.shape[1], lib_index.num_points), comp
        )
        torch.matmul(augmented, lib_index.augmented.mT, out=dist)
        return dist

    def __weights_from_dists(self, near_dist, indices, n_nbrs, n_nbrs_max):
        timings = {}
        eps = torch.finfo(near_dist.dtype).eps
        with time_block(self.logger, self.device, timings, "exp"):
            d0 = near_dist.amin(dim=2, keepdim=True).clamp_min(eps)
            w = near_dist.div(d0).neg_().exp_()
            w.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

        with time_block(self.logger, self.device, timings, "mask"):
            # Uniform k (the default, nbrs_num = E_x + 1 over equal-width sources)
            # keeps every column, so the mask is only built when it can drop one.
            if int(n_nbrs.min()) < n_nbrs_max:
                keep = (torch.arange(n_nbrs_max, device=w.device).unsqueeze(0) < n_nbrs.unsqueeze(1))
                w.mul_(keep[:, None, :].to(w.dtype))

        with time_block(self.logger, self.device, timings, "normalize"):
            sumw = w.sum(dim=2, keepdim=True)
            zero = sumw <= eps
            if zero.any():
                raise RuntimeError(
                    "All neighbors excluded by `exclusion_window` for some queries. "
                    "Reduce `exclusion_window`, increase `library_size`, or ensure the "
                    "library contains valid neighbors."
                )
            out = w.div_(sumw.clamp_min(eps)).to(self.dtype)

        if self._debug_enabled():
            timings["total"] = sum(v for v in timings.values())
            self.logger.debug("Neighbor weight timings: %s", timings_summary(timings, ["exp", "mask", "normalize", "total"]))
        return out, indices


    def __get_local_weights(self, lib, sublib, subset_idx, sample_idx, exclusion_rad, theta):
        dist = self._cdist(sublib, lib)
        if theta == None:
            weights = dist.neg_().exp_()
        else:
            denom = dist.mean(dim=2, keepdim=True).clamp_min(1e-12)  # (n_X, S, 1)
            weights = dist.mul_(-theta).div_(denom).exp_() 

        #if exclusion_rad > 0:
        if exclusion_rad is not None:
            #allowed = (torch.abs(subset_idx[None] - sample_idx[:,None]) > exclusion_rad)
            allowed = (
                    (subset_idx[None, :] > (sample_idx[:, None] + exclusion_rad)) |
                    (subset_idx[None, :] < (sample_idx[:, None] - exclusion_rad))
                ) 
            weights.masked_fill_(~allowed.unsqueeze(0), 0.0)

        return weights
      
    def _cdist(self, a, b):
        comp = self._promoted_compute_dtype()
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
