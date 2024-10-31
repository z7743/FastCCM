import torch
import numpy as np
from FastCCM.CCM import PairwiseCCM
from FastCCM.utils.utils import get_td_embedding_np

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

    def find_optimal_embedding_params(self, x, y, subset_size, subsample_size, exclusion_rad, E_range, tau_range, tp_max,  method="simplex", **kwargs):
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
        res = self.ccm.compute(X_emb,Y_emb,
                               subset_size=subset_size,
                               subsample_size=subsample_size,
                               exclusion_rad=exclusion_rad,
                               tp=0,
                               subtract_corr=True,
                               method="simplex",
                               nbrs_num=10)[0].reshape(tp_max-1,tau_range.shape[0],E_range.shape[0],)
        

        # Find optimal tau and E for this set
        max_idx = np.unravel_index(np.argmax(res.mean(axis=0)), res.mean(axis=0).shape)

        # Return results as a dictionary
        return {
            "result": res,
            "optimal_tau": tau_range[max_idx[0]],
            "optimal_E": E_range[max_idx[1]],
            "values": res[max_idx]
        }

