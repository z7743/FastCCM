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


    def convergence_test(self, X, Y, subset_sizes, subsample_size, exclusion_rad, tp=0, method="simplex", trials=10, **kwargs):
        """
        Conducts a convergence test by calculating pairwise CCM for increasing subset sizes.

        Parameters:
            X (list): List of input time series data embedding.
            Y (list): List of target time series data embedding.
            subset_sizes (list or np.array): List of subset sizes to test for convergence.
            subsample_size (int): Number of random samples for estimating prediction quality.
            exclusion_rad (int): Exclusion radius to avoid temporally close points.
            tp (int): Prediction interval.
            method (str): Prediction method ("simplex" or "smap").
            trials (int): Number of trials to run for each subset size.
            **kwargs: Additional parameters for CCM (e.g., nbrs_num for simplex, theta for smap).

        Returns:
            dict: Contains results for X->Y and Y->X tests, each as a 2D array (subset size x trials).
        """
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

    def prediction_interval_test(self, x, y, subset_size, subsample_size, max_tp = 1, exclusion_rad=20, method="simplex", **kwargs):
        """
        Calculates CCM for different prediction intervals (tp).

        Parameters:
            x (torch.Tensor): Input time series data embedding (source).
            y (torch.Tensor): Target time series data embedding (target).
            subset_size (int): Number of random samples of embeddings taken to approximate the manifold. Nearest neighbors for cross-mapping will be searched among this subset.
            subsample_size (int): Number of random samples of embeddings used to estimate prediction quality. Nearest neighbors for these samples will be searched.
            exclusion_rad (int): Exclusion radius to avoid selecting temporally close points from the subset.
            max_tp (int): Maximum prediction interval to test.
            method (str): Method to compute the prediction ("simplex" or "smap").
            **kwargs: Additional parameters for CCM (e.g., nbrs_num for simplex, theta for smap).

        Returns:
            dict: Contains results for X->Y, with the results stored as an array for each prediction interval.
        """

        X_ = x[:-(max_tp-1)][None]
        Y_ = np.transpose(get_td_embedding_np(y,max_tp,1),axes=(1,0,2))

        res = self.ccm.compute(X_,Y_,subset_size=subset_size, subsample_size=subsample_size,
                    exclusion_rad=exclusion_rad, tp=0,
                    method=method, **kwargs
                )


        # Convert lists to numpy arrays for easier analysis and visualization
        return {
            "tp_list": np.arange(max_tp),
            "X_to_Y": np.array(res),
        }

    def find_optimal_embedding_params(self, x, y, subset_size, subsample_size, exclusion_rad, E_range, tau_range, tp_max,  method="simplex", trials=10, **kwargs):
        """
        Finds the optimal embedding parameters (E and tau) for CCM.

        Parameters:
            x (torch.Tensor): Input time series data (1-D).
            x (torch.Tensor): Target time series data (1-D).
            E_range (np.array): Range of embedding dimensions to test.
            tau_range (np.array): Range of time delays to test.
            tp_max (int): Maximum prediction interval for embedding.
            subset_size (int): Number of random samples of embeddings taken to approximate the manifold. Nearest neighbors for cross-mapping will be searched among this subset.
            subsample_size (int): Number of random samples of embeddings used to estimate prediction quality. Nearest neighbors for these samples will be searched.
            exclusion_rad (int): Exclusion radius to avoid selecting temporally close points from the subset.
            method (str): Method to compute the prediction ("simplex" or "smap").
            **kwargs: Additional parameters for CCM (e.g., nbrs_num for simplex, theta for smap).

        Returns:
            dict: Contains the optimal tau, E, and the corresponding CCM value.
        """
                
        # Prepare embeddings and compute CCM for each tau and E combination
        X_emb = np.concatenate([np.array([get_td_embedding_np(x[:-tp_max,None],e,tau)[:,:,0] for e in E_range],dtype=object) for tau in tau_range])
        Y_emb = np.array([y[:i, None] for i in range(-(tp_max),-1)],dtype=object)
        res = np.mean([self.ccm.compute(X_emb,Y_emb,
                               subset_size=subset_size,
                               subsample_size=subsample_size,
                               exclusion_rad=exclusion_rad,
                               tp=0,
                               subtract_corr=False,
                               method=method,
                               **kwargs)[0].reshape(tp_max-1,tau_range.shape[0],E_range.shape[0],) for exp in range(trials)],axis=0)
        

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

    def plot_convergence_test(self, conv_test_res, X_idx=0, Y_idx=0, xscale="log", plot_means_only=False):
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
        X_to_Y_results = conv_test_res["X_to_Y"][:, :, :, X_idx, Y_idx]
        Y_to_X_results = conv_test_res["Y_to_X"][:, :, :, Y_idx, X_idx]

        # Number of dimensions for Y and X
        num_dimensions_Y = (np.isnan(X_to_Y_results).sum(axis=(0,1)) == 0).sum()
        num_dimensions_X = (np.isnan(Y_to_X_results).sum(axis=(0,1)) == 0).sum()

        # Generate color palettes dynamically based on the number of dimensions, avoiding light colors
        colors_X_to_Y = [cm.gray(0.2 + 0.6 * (i / (num_dimensions_Y - 1))) for i in range(num_dimensions_Y)]
        colors_Y_to_X = [cm.Reds(0.2 + 0.6 * (i / (num_dimensions_X - 1))) for i in range(num_dimensions_X)]

        plt.figure(figsize=(10, 6))

        if not plot_means_only:
            # Plot results for (x1, x2, x3) -> y1, y2, y3 with dynamically generated colors and error bars
            for dim in range(num_dimensions_Y):
                mean_values = X_to_Y_results.mean(axis=1)[:, dim]
                std_errors = X_to_Y_results.std(axis=1)[:, dim] / np.sqrt(X_to_Y_results.shape[1])  # Standard error of the mean
                plt.errorbar(subset_sizes, mean_values, yerr=std_errors, linestyle="--", label=f"X -> y{dim + 1}", color=colors_X_to_Y[dim])

            # Plot results for (y1, y2, y3) -> x1, x2, x3 with dynamically generated colors and error bars
            for dim in range(num_dimensions_X):
                mean_values = Y_to_X_results.mean(axis=1)[:, dim]
                std_errors = Y_to_X_results.std(axis=1)[:, dim] / np.sqrt(Y_to_X_results.shape[1])  # Standard error of the mean
                plt.errorbar(subset_sizes, mean_values, yerr=std_errors, linestyle="--", label=f"Y -> x{dim + 1}", color=colors_Y_to_X[dim])

        # Plot mean lines for emphasis with error bars for overall mean
        mean_X_to_Y = np.nanmean(X_to_Y_results,axis=(1, 2))
        std_X_to_Y = np.nanmean(X_to_Y_results,axis=2).std(axis=1) / np.sqrt(X_to_Y_results.shape[1])
        plt.errorbar(subset_sizes, mean_X_to_Y, yerr=std_X_to_Y, label="$Y|M_{X}$ (mean)", lw=2.5, color="black")

        mean_Y_to_X = np.nanmean(Y_to_X_results,axis=(1, 2))
        std_Y_to_X = np.nanmean(Y_to_X_results, axis=2).std(axis=1) / np.sqrt(Y_to_X_results.shape[1])
        plt.errorbar(subset_sizes, mean_Y_to_X, yerr=std_Y_to_X, label="$X|M_{Y}$ (mean)", lw=2.5, color="red")

        # Set plot properties
        plt.xscale(xscale)  # Set the X-axis scale to "log" or "linear"
        plt.xlabel("Library Size")
        plt.ylabel("Pearson Correlation Coefficient")
        plt.title("Convergence Test Visualization" if not plot_means_only else "Convergence Test: Mean Only")
        plt.grid(True)
        plt.legend()
        plt.show()

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