# ccm_utils.py
import warnings
import torch
import numpy as np
from fastccm import PairwiseCCM
from fastccm.utils.utils import get_td_embedding_np
import matplotlib.pyplot as plt
from matplotlib import cm


class Functions:
    def __init__(self, device="cpu",dtype="float32", compute_dtype="float32"):
        """
        Initializes the CCMFunction with a PairwiseCCM instance.

        Parameters:
            device (str): The computation device ('cpu' or 'cuda') to use for all calculations.
        """
        self.ccm = PairwiseCCM(device=device, dtype=dtype, compute_dtype=compute_dtype)

    def _resolve_sizes_compute_(self, X_emb, Y_emb, library_size, sample_size, tp):
        # Mirror PairwiseCCM.__ccm_core 'compute' defaults using GLOBAL min_len
        min_len = min(min(arr.shape[0] for arr in X_emb),
                      min(arr.shape[0] for arr in Y_emb))
        if library_size is None:
            L = min_len
        elif library_size == "auto":
            L = min(min_len // 2, 700)
        else:
            L = int(library_size)

        if sample_size is None:
            S = min_len
        elif sample_size == "auto":
            S = min(min_len // 6, 250)
        else:
            S = int(sample_size)
        return L, S

    def _resolve_sizes_predict_(self, X_lib_emb, Y_lib_emb, X_pred_emb, library_size, tp):
        # Mirror PairwiseCCM.__ccm_core 'predict' defaults using GLOBAL minima
        min_len_lib = min(min(arr.shape[0] for arr in X_lib_emb),
                          min(arr.shape[0] for arr in Y_lib_emb))
        if X_pred_emb is None:
            min_len_pred = min(arr.shape[0] for arr in X_lib_emb)
        else:
            min_len_pred = min(arr.shape[0] for arr in X_pred_emb)

        if library_size is None:
            L = min_len_lib
        elif library_size == "auto":
            L = min(min_len_lib // 2, 700)
        else:
            L = int(library_size)
        return L, min_len_pred

    def compute_blocked(self, *args, **kwargs):
        warnings.warn("Functions.compute_blocked → Functions.score_matrix_blocked", DeprecationWarning)
        return self.score_matrix_blocked(*args, **kwargs)

    def predict_blocked(self, *args, **kwargs):
        warnings.warn("Functions.predict_blocked → Functions.predict_matrix_blocked", DeprecationWarning)
        return self.predict_matrix_blocked(*args, **kwargs)

    def score_matrix_blocked(
        self,
        X_emb,
        Y_emb=None,
        *,
        x_block=64,
        y_block=64,
        library_size=None,
        sample_size=None,
        exclusion_window=0,
        tp=0,
        method="simplex",
        seed=None,
        metric="corr",
        **kwargs
    ):
        """
        Compute CCM in (n_X x n_Y) blocks to reduce peak memory.

        Returns
        -------
        np.ndarray  with shape (E_y_max, n_Y, n_X)
        """
        import numpy as np

        if Y_emb is None:
            Y_emb = X_emb

        nX, nY = len(X_emb), len(Y_emb)
        E_y_max = max(y.shape[-1] for y in Y_emb)

        # Preallocate and fill with NaN (in case some Y have smaller E)
        out = np.full((E_y_max, nY, nX), np.nan, dtype=np.float32)

        # Resolve sizes globally so each block uses identical sampling defaults
        L_res, S_res = self._resolve_sizes_compute_(X_emb, Y_emb, library_size, sample_size, tp)

        for y0 in range(0, nY, y_block):
            y1 = min(y0 + y_block, nY)
            for x0 in range(0, nX, x_block):
                x1 = min(x0 + x_block, nX)

                r_blk = self.ccm.score_matrix(
                    X_emb[x0:x1],
                    Y_emb[y0:y1],
                    library_size=L_res,
                    sample_size=S_res,
                    exclusion_window=exclusion_window,
                    tp=tp,
                    method=method,
                    seed=seed,
                    metric=metric,
                    **kwargs
                )  # shape: (E_y_blk, y_len, x_len)

                Ey_blk = r_blk.shape[0]
                out[:Ey_blk, y0:y1, x0:x1] = r_blk

        return out

    def predict_matrix_blocked(
        self,
        X_lib_emb,
        Y_lib_emb=None,
        X_pred_emb=None,
        *,
        x_block=64,
        y_block=64,
        library_size=None,
        exclusion_window=0,
        tp=0,
        method="simplex",
        seed=None,
        metric="corr",
        **kwargs
    ):
        """
        Predict in (n_X x n_Y) blocks. Forces a GLOBAL S_pred so output is a single tensor.

        Returns
        -------
        np.ndarray with shape (S_pred_global, E_y_max, n_Y, n_X)
        """
        import numpy as np

        if Y_lib_emb is None:
            Y_lib_emb = X_lib_emb

        nX, nY = len(X_lib_emb), len(Y_lib_emb)
        E_y_max = max(y.shape[-1] for y in Y_lib_emb)

        # Resolve global library size and a global S_pred
        L_res, S_pred = self._resolve_sizes_predict_(X_lib_emb, Y_lib_emb, X_pred_emb, library_size, tp)

        # Build a trimmed X_pred view so every block uses the same S_pred
        if X_pred_emb is None:
            X_pred_trimmed = [x[:S_pred] for x in X_lib_emb]
        else:
            X_pred_trimmed = [x[:S_pred] for x in X_pred_emb]

        out = np.full((S_pred, E_y_max, nY, nX), np.nan, dtype=np.float32)

        for y0 in range(0, nY, y_block):
            y1 = min(y0 + y_block, nY)
            for x0 in range(0, nX, x_block):
                x1 = min(x0 + x_block, nX)

                A_blk = self.ccm.predict_matrix(
                    X_lib_emb[x0:x1],
                    Y_lib_emb[y0:y1],
                    X_pred_trimmed[x0:x1],
                    library_size=L_res,
                    exclusion_window=exclusion_window,
                    tp=tp,
                    method=method,
                    seed=seed,
                    metric=metric,
                    **kwargs
                )  # shape: (S_pred, E_y_blk, y_len, x_len)

                Ey_blk = A_blk.shape[1]
                out[:, :Ey_blk, y0:y1, x0:x1] = A_blk

        return out

    def convergence_test(
            self, 
            X_emb, 
            Y_emb=None, 
            library_sizes="auto", 
            sample_size=None, 
            exclusion_window=0, 
            tp=0, 
            method="simplex",
            trials=10, 
            seed=None, 
            metric="corr",
            **kwargs
    ):
        """
        Run a CCM convergence test by sweeping the library size.

        Parameters:
            X_emb : list[np.ndarray]
                List of input time series data embeddings.
            Y_emb : list[np.ndarray] or None, optional
                List of target time series data embeddings. If None, defaults to X.
            library_sizes : list[int] | np.ndarray | "auto", optional
                List of subset sizes to test for convergence.
                If "auto", set as logarithmically spaced values between max(max common length//100, 10) and max common length.
            sample_size : int | "auto" | None, optional
                Number of random samples of embeddings to estimate prediction quality. 
                If None, set to max common length. If "auto", set to max common length//6.
            exclusion_window : int, optional
                Minimum temporal gap (in samples) around each query index. Neighbors with
                |t_neighbor − t_query| ≤ exclusion_window are excluded (self-match excluded
                when 0).
            tp : int, optional
                Prediction horizon. Predicts Y[t + tp] from X[t]. Default: 0.
            method : {"simplex", "smap"}, optional
                Local regressor: k-NN weighted average ("simplex") or locally weighted
                linear ("smap"). Default: "simplex".
            trials : int, optional
                Number of independent trials per library size. Default: 10.
            seed : int or None, optional
                Base seed for reproducible random sampling. For trial t, the seed passed
                to X→Y is (seed + t) and to Y→X is (seed + t + 10000). If None, sampling
                is non-deterministic.
            metric : {"corr","mse","rmse","neg_nrmse","dcorr"} or Callable, optional
                Scoring function applied to (prediction, target). If a string, one of the
                built-ins. If a callable, it must accept (A, B) with shapes
                [S, E_y, n_Y, n_X] and return [E_y, n_Y, n_X].
                Default: "corr".
            **kwargs: Additional parameters for CCM, including:
                - nbrs_num : int or list[int]
                    Number of neighbors for "simplex". Defaults to E + 1 internally.
                - theta : float (for "smap") 
                    Local weighting parameter for "smap". Defaults to 5 internally.

        Returns
        -------
        dict
            {
            "library_sizes": np.ndarray,     # same values as `library_sizes`
            "X_to_Y": np.ndarray,            # shape: (n_sizes, trials, E_y, n_Y, n_X)
            "Y_to_X": np.ndarray,            # shape: (n_sizes, trials, E_x, n_X, n_Y)
            }
        """
        if Y_emb is None:
            Y_emb = X_emb  # Default Y to X if not provided

        num_ts_X = len(X_emb)
        num_ts_Y = len(Y_emb)

        min_len = min(
            min(y.shape[0] for y in Y_emb),
            min(x.shape[0] for x in X_emb)
        )

        # Handle library_sizes
        if isinstance(library_sizes, str) and library_sizes == "auto":
            lo = max(min_len // 100, 10)
            hi = max(min_len, lo + 1)
            library_sizes = np.unique(
                np.logspace(np.log10(lo), np.log10(hi), num=10, dtype=int)
            ).tolist()
        elif not isinstance(library_sizes, (list, np.ndarray)):
            raise ValueError("library_sizes must be either 'auto', a list, or a numpy array.")

            
        res_X_to_Y = []
        res_Y_to_X = []

        # Running convergence test for each subset size
        for size in library_sizes:
            res_X_to_Y_size = []
            res_Y_to_X_size = []

            for t in range(trials):
                trial_seed_xy = None if seed is None else int(seed) + t
                trial_seed_yx = None if seed is None else int(seed) + t + 10000
                # Calculate CCM for X -> Y
                res_X_to_Y_size.append(self.ccm.score_matrix(
                    X_emb, Y_emb,
                    library_size=size, sample_size=sample_size,
                    exclusion_window=exclusion_window, tp=tp,
                    method=method,
                    seed=trial_seed_xy, metric=metric, clean_after=False, **kwargs
                ))

                # Calculate CCM for Y -> X
                res_Y_to_X_size.append(self.ccm.score_matrix(
                    Y_emb, X_emb,
                    library_size=size, sample_size=sample_size,
                    exclusion_window=exclusion_window, tp=tp,
                    method=method,
                    seed=trial_seed_yx, metric=metric, clean_after=False, **kwargs
                ))
                
            # Store results for current subset size
            res_X_to_Y.append(res_X_to_Y_size)
            res_Y_to_X.append(res_Y_to_X_size)

        # Convert lists to numpy arrays for easier analysis and visualization
        return {
            "library_sizes": np.array(library_sizes),
            "X_to_Y": np.array(res_X_to_Y),
            "Y_to_X": np.array(res_Y_to_X)
        }

    def prediction_interval_test(
            self, 
            x_emb, 
            y_emb, 
            library_size="auto", 
            sample_size="auto", 
            max_tp=1, 
            exclusion_window=0, 
            method="simplex", 
            seed=None, 
            metric="corr",
            **kwargs
    ):
        """
        Calculates CCM for different prediction intervals (tp).

        Parameters:
            x_emb : np.ndarray
                Input time series data embedding (source).
            y_emb : np.ndarray 
                Target time series data embedding (target).
            library_size : int | "auto" | None, optional 
                Number of library points. If None, uses max common length; if "auto",
                uses max_common_len // 2 (capped at 700 internally).
            sample_size : int | "auto" | None, optional
                Number of random samples of embeddings used to estimate prediction quality.
            max_tp : int, optional
                Maximum prediction interval. Evaluates tp in [0, max_tp]. Default: 1.
            exclusion_window : int, optional
                Theiler window (minimum temporal gap). Default: 0.
            method : {"simplex", "smap"}, optional 
                Local regressor to use. Default: "simplex".
            seed : int or None, optional
                Base seed for reproducible random sampling. For trial t, the seed passed
                to X→Y is (seed + t) and to Y→X is (seed + t + 10000). If None, sampling
                is non-deterministic.
            metric : {"corr","mse","rmse","neg_nrmse","dcorr"} or Callable, optional
                Scoring function applied to (prediction, target). If a string, one of the
                built-ins. If a callable, it must accept (A, B) with shapes
                [S, E_y, n_Y, n_X] and return [E_y, n_Y, n_X].
                Default: "corr".
            **kwargs :
                Passed through to PairwiseCCM.score_matrix. Useful keys:
                - nbrs_num : int or list[int]
                - theta    : float (for "smap")

        Returns
        -------
        dict
            {
            "tp_list": np.ndarray,           # shape: (max_tp + 1,)
            "X_to_Y": np.ndarray,            # output of PairwiseCCM.score_matrix for stacked targets
            }
        """

        X_ = x_emb[:-(max_tp)][None]
        Y_ = np.transpose(get_td_embedding_np(y_emb,max_tp+1,1),axes=(1,0,2))
        
        res = self.ccm.score_matrix(X_,Y_,library_size=library_size, sample_size=sample_size,
                    exclusion_window=exclusion_window, tp=0,
                    method=method, seed=seed, metric=metric, **kwargs
                )


        # Convert lists to numpy arrays for easier analysis and visualization
        return {
            "tp_list": np.arange(max_tp+1),
            "X_to_Y": np.array(res),
        }

    def find_optimal_embedding_params(
            self, 
            x, 
            y=None, 
            library_size="auto", 
            sample_size="auto", 
            exclusion_window=0, 
            E_range=np.arange(1,10,1), 
            tau_range=np.arange(1,10,1), 
            tp_max=1,  
            method="simplex", 
            trials=10, 
            seed=None,
            metric="corr",
            **kwargs
    ):
        """
        Grid-search (E, τ) for CCM by building delay embeddings of x for each (E, τ).

        Parameters:
            x : np.ndarray
                Source scalar time series (1D).
            y : np.ndarray or None, optional
                Target scalar time series (1D). If None, uses x.
            library_size : int | "auto" | None, optional
                Number of library points for CCM. See PairwiseCCM for defaults.
            sample_size : int | "auto" | None, optional
                Number of query points for scoring. See PairwiseCCM for defaults.
            exclusion_window : int, optional
                Theiler window (minimum temporal gap). Default: 0.
            E_range : array-like of int, optional
                Embedding dimensions to test. Default: np.arange(1, 10).
            tau_range : array-like of int, optional
                Time delays to test. Default: np.arange(1, 10).
            tp_max : int, optional
                Maximum prediction interval used to form Y targets. Targets correspond
                to tp=1..tp_max (tp=0 is not included here). Default: 1.
            method : {"simplex", "smap"}, optional
                Local regressor. Default: "simplex".
            trials : int, optional
                Number of repeated runs per (E, τ) to average results. Default: 10.
            seed : int or None, optional
                Base seed for reproducible random sampling. For trial t, the seed passed
                to X→Y is (seed + t) and to Y→X is (seed + t + 10000). If None, sampling
                is non-deterministic.
            metric : {"corr","mse","rmse","neg_nrmse","dcorr"} or Callable, optional
                Scoring function applied to (prediction, target). If a string, one of the
                built-ins. If a callable, it must accept (A, B) with shapes
                [S, E_y, n_Y, n_X] and return [E_y, n_Y, n_X].
                Default: "corr".
            **kwargs :
                Passed through to PairwiseCCM.score_matrix. Useful keys:
                - nbrs_num : int or list[int]
                - theta    : float (for "smap")

        Returns
        -------
        dict
            {
            "E_range": array-like,                         # as provided
            "tau_range": array-like,                       # as provided
            "tp_range": np.ndarray,                        # np.arange(1, tp_max)
            "result": np.ndarray,                          # shape: (tp_max, len(tau_range), len(E_range))
            "optimal_tau": int,                            # tau with highest mean over tp
            "optimal_E": int,                              # E with highest mean over tp
            "values": np.ndarray,                          # result[:, tau*, E*], shape: (tp_max,)
            }
        """
        if y is None:
            y = x  # Default Y to X if not provided
         
        # Prepare embeddings and compute CCM for each tau and E combination
        X_emb = np.concatenate([np.array([get_td_embedding_np(x[:-tp_max,None],e,tau)[:,:,0] for e in E_range],dtype=object) for tau in tau_range])
        Y_emb = [y[:(y.shape[0]+i), None] for i in np.arange(-(tp_max)+1,1)]
        
        res = np.mean([self.ccm.score_matrix(X_emb,Y_emb,
                               library_size=library_size,
                               sample_size=sample_size,
                               exclusion_window=exclusion_window,
                               tp=0,
                               method=method,
                               seed=None if seed is None else int(seed) + exp,
                               metric=metric,
                               **kwargs)[0].reshape(tp_max,tau_range.shape[0],E_range.shape[0],) for exp in range(trials)],axis=0)
        

        # Find optimal tau and E for this set
        mean_over_tp = res.mean(axis=0)
        max_idx = np.unravel_index(np.argmax(mean_over_tp), mean_over_tp.shape)

        # Return results as a dictionary
        return {
            "E_range": E_range,
            "tau_range": tau_range,
            "tp_range": np.arange(1,tp_max),
            "result": res,
            "optimal_tau": tau_range[max_idx[0]],
            "optimal_E": E_range[max_idx[1]],
            "values": res[:, max_idx[0], max_idx[1]]
        }

class Visualizer:
    def __init__(self):
        """
        Initializes the Visualizer class.
        """
        pass

    def plot_convergence_test(self, conv_test_res, X_idx=0, Y_idx=0, xscale="log", plot_means_only=False, ax=None):
        """
        Plots the results of a convergence test with error bars representing trial variability.
        Allows customization of the X-axis scale and an option to plot only mean lines.

        Parameters:
            conv_test_res (dict): Results from the Functions.convergence_test method.
            X_idx (int): Index of the embedding in X to visualize.
            Y_idx (int): Index of the embedding in Y to visualize.
            xscale (str): Scale of the X-axis, either "log" or "linear".
            plot_means_only (bool): If True, plots only the mean lines without individual dimension plots.
        """
        subset_sizes = conv_test_res["library_sizes"]
        X_to_Y_results = conv_test_res["X_to_Y"][:, :, :, Y_idx, X_idx]
        Y_to_X_results = conv_test_res["Y_to_X"][:, :, :, X_idx, Y_idx]

        # Number of dimensions for Y and X
        num_dimensions_Y = (np.isnan(X_to_Y_results).sum(axis=(0,1)) == 0).sum()
        num_dimensions_X = (np.isnan(Y_to_X_results).sum(axis=(0,1)) == 0).sum()

        # Generate color palettes dynamically based on the number of dimensions, avoiding light colors
        colors_X_to_Y = [cm.gray(0.2 + 0.6 * (i / (num_dimensions_Y - 1))) for i in range(num_dimensions_Y)]
        colors_Y_to_X = [cm.Reds(0.2 + 0.6 * (i / (num_dimensions_X - 1))) for i in range(num_dimensions_X)]
        
        if ax == None:  
            fig, ax = plt.subplots(figsize=(10, 6))

        if not plot_means_only:
            # Plot results for (x1, x2, x3) -> y1, y2, y3 with dynamically generated colors and error bars
            for dim in range(num_dimensions_Y):
                mean_values = X_to_Y_results.mean(axis=1)[:, dim]
                std_errors = X_to_Y_results.std(axis=1)[:, dim] / np.sqrt(X_to_Y_results.shape[1])  # Standard error of the mean
                ax.errorbar(subset_sizes, mean_values, yerr=std_errors, linestyle="--", label=f"X -> y{dim + 1}", color=colors_X_to_Y[dim])

            # Plot results for (y1, y2, y3) -> x1, x2, x3 with dynamically generated colors and error bars
            for dim in range(num_dimensions_X):
                mean_values = Y_to_X_results.mean(axis=1)[:, dim]
                std_errors = Y_to_X_results.std(axis=1)[:, dim] / np.sqrt(Y_to_X_results.shape[1])  # Standard error of the mean
                ax.errorbar(subset_sizes, mean_values, yerr=std_errors, linestyle="--", label=f"Y -> x{dim + 1}", color=colors_Y_to_X[dim])

        # Plot mean lines for emphasis with error bars for overall mean
        mean_X_to_Y = np.nanmean(X_to_Y_results,axis=(1, 2))
        std_X_to_Y = np.nanmean(X_to_Y_results,axis=2).std(axis=1) / np.sqrt(X_to_Y_results.shape[1])
        ax.errorbar(subset_sizes, mean_X_to_Y, yerr=std_X_to_Y, label="$Y|M_{X}$ (mean)", lw=2.5, color="black")

        mean_Y_to_X = np.nanmean(Y_to_X_results,axis=(1, 2))
        std_Y_to_X = np.nanmean(Y_to_X_results, axis=2).std(axis=1) / np.sqrt(Y_to_X_results.shape[1])
        ax.errorbar(subset_sizes, mean_Y_to_X, yerr=std_Y_to_X, label="$X|M_{Y}$ (mean)", lw=2.5, color="red")

        
        ax.set_xscale(xscale)  # Set the X-axis scale to "log" or "linear"
        ax.set_xlabel("Library Size")
        ax.set_ylabel("Metric Value")
        ax.set_title("Convergence Test Visualization" if not plot_means_only else "Convergence Test: Mean Only")
        ax.grid(True)
        ax.legend()

        return ax


    def visualize_optimal_e_tau(self, optimal_E_tau_res):
        """
        Visualizes the optimal embedding dimension (E) and time delay (tau) results.

        Args:
            optimal_E_tau_res (dict): Dictionary containing the results of the optimal embedding analysis.
                                    Expected keys are 'result', 'E_range', and 'tau_range'.
        """
        # Extract data from the dictionary
        result = optimal_E_tau_res["result"].mean(axis=0)
        E_range = optimal_E_tau_res["E_range"]
        tau_range = optimal_E_tau_res["tau_range"]

        # Create the plot using imshow for more flexibility
        plt.figure(figsize=(10, 8))
        plt.imshow(result, aspect='auto', extent=[E_range[0], E_range[-1], tau_range[0], tau_range[-1]], origin='lower')

        # Set axis labels and title
        plt.colorbar(label='Mean CCM Value')
        plt.xlabel('Embedding Dimension (E)')
        plt.ylabel('Time Delay (tau)')
        plt.title('Optimal E and Tau Analysis')

        # Set ticks for better visualization
        plt.xticks(E_range, labels=E_range)
        plt.yticks(tau_range, labels=tau_range)

        # Display the plot
        plt.show()

    def plot_interval_prediction_test(self, interval_test_res):
        """
        Plots the results of an interval prediction test.

        Parameters:
            interval_test_res (dict): Results from the Functions.prediction_interval_test method.
        """
        tp_list = interval_test_res["tp_list"]
        X_to_Y_results = interval_test_res["X_to_Y"]

        plt.figure(figsize=(10, 6))

        # Plot each dimension of the results with labels
        num_dimensions = X_to_Y_results.shape[0]

        for dim in range(num_dimensions):
            plt.plot(tp_list, X_to_Y_results[dim, :, 0].T, label=f"y{dim + 1}")

        # Plot properties
        plt.legend()
        plt.xlabel("Prediction Interval")
        plt.ylabel("Metric Value")
        plt.title("Prediction Interval Test")
        plt.grid(True)
        plt.show()