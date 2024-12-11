import torch
import numpy as np
from FastCCM.CCM import PairwiseCCM
from FastCCM.utils.utils import get_td_embedding_np
import matplotlib.pyplot as plt
from matplotlib import cm


class Functions:
    def __init__(self, device="cpu"):
        """
        Initializes the CCMFunction with a PairwiseCCM instance.

        Parameters:
            device (str): The computation device ('cpu' or 'cuda') to use for all calculations.
        """
        self.ccm = PairwiseCCM(device=device)


    def convergence_test(self, X, Y=None, subset_sizes="auto", subsample_size=None, exclusion_rad=0, tp=0, method="simplex", trials=10, **kwargs):
        """
        Conducts a convergence test by calculating pairwise CCM for increasing subset sizes.

        Parameters:
            X (list of np.array): List of input time series data embeddings.
            Y (list of np.array or None, optional): List of target time series data embeddings. If None, defaults to X.
            subset_sizes (list, np.array, or "auto"): List of subset sizes to test for convergence.
                If "auto", set as logarithmically spaced values between max(max common length//100, 10) and max common length.
            subsample_size (int, None, or "auto"): Number of random samples of embeddings to estimate prediction quality. 
                If None, set to max common length. If "auto", set to max common length//6.
            exclusion_rad (int, optional): Exclusion radius to avoid temporally close points. Default is 0.
            tp (int, optional): Prediction interval. Default is 0.
            method (str, optional): Prediction method ("simplex" or "smap"). Default is "simplex".
            trials (int, optional): Number of trials to run for each subset size. Default is 10.
            **kwargs: Additional parameters for CCM, including:
                - nbrs_num (int): Number of neighbors for "simplex". Defaults to E + 1 internally.
                - theta (float): Local weighting parameter for "smap". Defaults to 5 internally.

        Returns:
            dict: Contains results for X->Y and Y->X tests, each as a 2D array (subset size x trials).
        """
        if Y is None:
            Y = X  # Default Y to X if not provided

        num_ts_X = len(X)
        num_ts_Y = len(Y)
        min_len = torch.tensor([Y[i].shape[0] for i in range(num_ts_Y)] + [X[i].shape[0] for i in range(num_ts_X)]).min().item()

        # Handle subset_sizes
        if subset_sizes == "auto":
            subset_sizes = np.unique(np.logspace(max(np.log10(min_len // 100),np.log10(10)), np.log10(min_len), num=10, dtype=int)).tolist() #[min_len // i for i in range(10, 0, -1)]
        elif not isinstance(subset_sizes, (list, np.ndarray)):
            raise ValueError("subset_sizes must be either 'auto' or a list of integers.")

        res_X_to_Y = []
        res_Y_to_X = []

        # Running convergence test for each subset size
        for size in subset_sizes:
            res_X_to_Y_size = []
            res_Y_to_X_size = []
            for _ in range(trials):
                # Calculate CCM for X -> Y
                res_X_to_Y_size.append(self.ccm.compute(
                    X, Y,
                    subset_size=size, subsample_size=subsample_size,
                    exclusion_rad=exclusion_rad, tp=tp,
                    method=method, **kwargs
                ))

                # Calculate CCM for Y -> X
                res_Y_to_X_size.append(self.ccm.compute(
                    Y, X,
                    subset_size=size, subsample_size=subsample_size,
                    exclusion_rad=exclusion_rad, tp=tp,
                    method=method, **kwargs
                ))
                
            # Store results for current subset size
            res_X_to_Y.append(res_X_to_Y_size)
            res_Y_to_X.append(res_Y_to_X_size)

        # Convert lists to numpy arrays for easier analysis and visualization
        return {
            "subset_sizes": np.array(subset_sizes),
            "X_to_Y": np.array(res_X_to_Y),
            "Y_to_X": np.array(res_Y_to_X)
        }

    def prediction_interval_test(self, x, y, subset_size="auto", subsample_size="auto", max_tp=1, exclusion_rad=0, method="simplex", **kwargs):
        """
        Calculates CCM for different prediction intervals (tp).

        Parameters:
            x (torch.Tensor): Input time series data embedding (source).
            y (torch.Tensor): Target time series data embedding (target).
            subset_size (int, None, or "auto"): Number of random samples of embeddings taken to approximate the manifold.
            subsample_size (int, None, or "auto"): Number of random samples of embeddings used to estimate prediction quality.
            exclusion_rad (int, optional): Exclusion radius to avoid selecting temporally close points from the subset. Default is 0.
            max_tp (int, optional): Maximum prediction interval to test. Default is 1.
            method (str, optional): Method to compute the prediction ("simplex" or "smap"). Default is "simplex".
            **kwargs: Additional parameters for CCM, including:
                - nbrs_num (int): Number of neighbors for "simplex". Defaults to E + 1 internally.
                - theta (float): Local weighting parameter for "smap". Defaults to 5 internally.

        Returns:
            dict: Contains results for X->Y, with the results stored as an array for each prediction interval.
        """

        X_ = x[:-(max_tp)][None]
        Y_ = np.transpose(get_td_embedding_np(y,max_tp+1,1),axes=(1,0,2))
        
        res = self.ccm.compute(X_,Y_,subset_size=subset_size, subsample_size=subsample_size,
                    exclusion_rad=exclusion_rad, tp=0,
                    method=method, **kwargs
                )


        # Convert lists to numpy arrays for easier analysis and visualization
        return {
            "tp_list": np.arange(max_tp+1),
            "X_to_Y": np.array(res),
        }

    def find_optimal_embedding_params(self, x, y=None, subset_size="auto", subsample_size="auto", exclusion_rad=0, E_range=np.arange(1,10,1), tau_range=np.arange(1,10,1), tp_max=1,  method="simplex", trials=10, **kwargs):
        """
        Finds the optimal embedding parameters (E and tau) for CCM.

        Parameters:
            x (np.array): Input time series data (1-D).
            y (np.array or None): Target time series data (1-D). If None, defaults to x.
            subset_size (int, None, or "auto"): Number of random samples of embeddings taken to approximate the manifold.
                If "auto", defaults to max common length // 2. If None, defaults to the length of the shorter series.
            subsample_size (int, None, or "auto"): Number of random samples of embeddings used to estimate prediction quality.
                If "auto", defaults to max common length // 6. If None, defaults to the length of the shorter series.
            exclusion_rad (int, optional): Exclusion radius to avoid selecting temporally close points from the subset. Default is 0.
            E_range (np.array, optional): Range of embedding dimensions (E) to test. Default is np.arange(1, 10, 1).
            tau_range (np.array, optional): Range of time delays (tau) to test. Default is np.arange(1, 10, 1).
            tp_max (int, optional): Maximum prediction interval for delayed target embedding. Default is 1.
            method (str, optional): Method to compute the prediction ("simplex" or "smap"). Default is "simplex".
            trials (int, optional): Number of trials to run for each combination of parameters. Default is 10.
            **kwargs: Additional parameters for CCM, including:
                - nbrs_num (int): Number of neighbors for "simplex". Defaults to E + 1 internally.
                - theta (float): Local weighting parameter for "smap". Defaults to 5 internally.

        Returns:
            dict: Contains the following keys:
                - "E_range": The tested range of embedding dimensions (E).
                - "tau_range": The tested range of time delays (tau).
                - "tp_range": Array of tested prediction intervals.
                - "result": Array of CCM values for all tested combinations of E, tau, and tp.
                - "optimal_tau": The optimal tau value corresponding to the highest CCM value.
                - "optimal_E": The optimal E value corresponding to the highest CCM value.
                - "values": Array of CCM values for the optimal E and tau across all tp.
        """
        if y is None:
            y = x  # Default Y to X if not provided
         
        # Prepare embeddings and compute CCM for each tau and E combination
        X_emb = np.concatenate([np.array([get_td_embedding_np(x[:-tp_max,None],e,tau)[:,:,0] for e in E_range],dtype=object) for tau in tau_range])
        Y_emb = [y[:(y.shape[0]+i), None] for i in np.arange(-(tp_max)+1,1)]
        
        res = np.mean([self.ccm.compute(X_emb,Y_emb,
                               subset_size=subset_size,
                               subsample_size=subsample_size,
                               exclusion_rad=exclusion_rad,
                               tp=0,
                               subtract_corr=False,
                               method=method,
                               **kwargs)[0].reshape(tp_max,tau_range.shape[0],E_range.shape[0],) for exp in range(trials)],axis=0)
        

        # Find optimal tau and E for this set
        max_idx = np.unravel_index(np.argmax(res.mean(axis=0)), res.mean(axis=0).shape)

        # Return results as a dictionary
        return {
            "E_range": E_range,
            "tau_range": tau_range,
            "tp_range": np.arange(1,tp_max),
            "result": res,
            "optimal_tau": tau_range[max_idx[0]],
            "optimal_E": E_range[max_idx[1]],
            "values": res[:,*max_idx]
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
        subset_sizes = conv_test_res["subset_sizes"]
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
        ax.set_ylabel("Pearson Correlation Coefficient")
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
        plt.ylabel("Pearson Correlation Coefficient")
        plt.title("Prediction Interval Test")
        plt.grid(True)
        plt.show()