import torch

class PairwiseCCM:
    def __init__(self,device = "cpu"):
        """
        Constructs a FastCCM object to perform Convergent Cross Mapping (CCM) using PyTorch.
        This object is optimized for calculating pariwise CCM matrix and can handle large datasets by utilizing batch processing on GPUs or CPUs.

        Parameters:
            device (str): The computation device ('cpu' or 'cuda') to use for all calculations.
        """
        self.device = device


    def compute(self, X, Y=None, subset_size=None, subsample_size=None, exclusion_rad=0, tp=0, method="simplex", **kwargs):
        """
        Main computation function for Convergent Cross Mapping (CCM).

        Parameters:
            X (list of np.array): List of embeddings from which to cross-map.
            Y (list of np.array or None): List of embeddings to predict. If None, set to be the same as X.
            subset_size (int or None): Number of random samples of embeddings taken to approximate the shape of the manifold well enough. If None, set to min_len. If "auto", set to min_len//2.
            subsample_size (int or None): Number of random samples of embeddings to estimate prediction quality. If None, set to min_len. If "auto", set to min_len//6.
            exclusion_rad (int): Exclusion radius to avoid picking temporally close points from a subset.
            tp (int): Interval of the prediction.
            method (str): Method to compute the prediction ("simplex", "smap").  
            nbrs_num (int, optional): Number of neighbors to consider for nearest neighbor calculations. Required if method is "simplex".
            theta (float, optional): Parameter controlling the degree of local weighting. Required if method is "smap".

        Returns:
            np.array: A matrix of correlation coefficients between the real and predicted states.

        Raises:
            ValueError: If an invalid method is specified or if required parameters for the chosen method are not provided.
        """
        if Y is None:
            Y = X

        if method == "simplex":
            required_params = ["nbrs_num"]
        elif method == "smap":
            required_params = ["theta"]
        else:
            raise ValueError("Invalid method. Supported methods are 'simplex' and 'smap'.")

        for param in required_params:
            if param not in kwargs:
                raise ValueError(f"For method {method}, parameter '{param}' must be specified.")

        # Number of time series 
        num_ts_X = len(X)
        num_ts_Y = len(Y)
        # Max embedding dimension
        max_E_X = torch.tensor([X[i].shape[-1] for i in range(num_ts_X)]).max().item()
        max_E_Y = torch.tensor([Y[i].shape[-1] for i in range(num_ts_Y)]).max().item()
        # Max common length
        min_len = torch.tensor([Y[i].shape[0] for i in range(num_ts_Y)] + [X[i].shape[0] for i in range(num_ts_X)]).min().item()

        # Handle subset_size and subsample_size
        if subset_size is None:
            subset_size = min_len
        elif subset_size == "auto":
            subset_size = min(min_len // 2, 700)

        if subsample_size is None:
            subsample_size = min_len
        elif subsample_size == "auto":
            subsample_size = min(min_len // 6, 250)

        # Random indices for sampling
        lib_indices = self.__get_random_indices(min_len - tp, subset_size)
        smpl_indices = self.__get_random_indices(min_len - tp, subsample_size)

        # Select X_lib and X_sample at time t and Y_lib, Y_sample at time t+tp
        X_lib = self.__get_random_sample(X, min_len, lib_indices, num_ts_X, max_E_X)
        X_sample = self.__get_random_sample(X, min_len, smpl_indices, num_ts_X, max_E_X)
        Y_lib_shifted = self.__get_random_sample(Y, min_len, lib_indices + tp, num_ts_Y, max_E_Y)
        Y_sample_shifted = self.__get_random_sample(Y, min_len, smpl_indices + tp, num_ts_Y, max_E_Y)

        if method == "simplex":
            nbrs_num = kwargs["nbrs_num"]
            r_AB = self.__simplex_prediction(lib_indices, smpl_indices, X_lib, X_sample, Y_lib_shifted, Y_sample_shifted, exclusion_rad, nbrs_num)
        elif method == "smap":
            theta = kwargs["theta"]
            r_AB = self.__smap_prediction(lib_indices, smpl_indices, X_lib, X_sample, Y_lib_shifted, Y_sample_shifted, exclusion_rad, theta)

        return r_AB.to("cpu").numpy()


    def predict(self, X, Y=None, subset_size=None, exclusion_rad=0, tp=0, method="simplex", **kwargs):
        """
        Prediction function for Convergent Cross Mapping (CCM).

        Parameters:
            X (list of np.array): List of embeddings from which to cross-map.
            Y (list of np.array or None): List of embeddings to predict. If None, set to be the same as X.
            subset_size (int or None): Number of random samples of embeddings taken to approximate the shape of the manifold well enough. If None, set to min_len.
            exclusion_rad (int): Exclusion radius to avoid picking temporally close points from a subset.
            tp (int): Interval of the prediction.
            method (str): Method to compute the prediction ("simplex", "smap").  
            nbrs_num (int, optional): Number of neighbors to consider for nearest neighbor calculations. Required if method is "simplex".
            theta (float, optional): Parameter controlling the degree of local weighting. Required if method is "smap".

        Returns:
            np.array: Predictions for the target time series.

        Raises:
            ValueError: If an invalid method is specified or if required parameters for the chosen method are not provided.
        """
        if Y is None:
            Y = X

        if method == "simplex":
            required_params = ["nbrs_num"]
        elif method == "smap":
            required_params = ["theta"]
        else:
            raise ValueError("Invalid method. Supported methods are 'simplex' and 'smap'.")

        for param in required_params:
            if param not in kwargs:
                raise ValueError(f"For method {method}, parameter '{param}' must be specified.")

        # Number of time series 
        num_ts_X = len(X)
        num_ts_Y = len(Y)
        # Max embedding dimension
        max_E_X = torch.tensor([X[i].shape[-1] for i in range(num_ts_X)]).max().item()
        max_E_Y = torch.tensor([Y[i].shape[-1] for i in range(num_ts_Y)]).max().item()
        # Max common length
        min_len = torch.tensor([Y[i].shape[0] for i in range(num_ts_Y)] + [X[i].shape[0] for i in range(num_ts_X)]).min().item()

        # Handle subset_size
        if subset_size is None:
            subset_size = min_len

        # Random indices for sampling
        lib_indices = self.__get_random_indices(min_len - tp, subset_size)
        smpl_indices = torch.arange(min_len - tp, device=self.device)

        # Select X_lib and X_sample at time t and Y_lib, Y_sample at time t+tp
        X_lib = self.__get_random_sample(X, min_len, lib_indices, num_ts_X, max_E_X)
        X_sample = self.__get_random_sample(X, min_len, smpl_indices, num_ts_X, max_E_X)
        Y_lib_shifted = self.__get_random_sample(Y, min_len, lib_indices + tp, num_ts_Y, max_E_Y)
        Y_sample_shifted = self.__get_random_sample(Y, min_len, smpl_indices + tp, num_ts_Y, max_E_Y)

        if method == "simplex":
            nbrs_num = kwargs["nbrs_num"]
            _, pred = self.__simplex_prediction(lib_indices, smpl_indices, X_lib, X_sample, Y_lib_shifted, Y_sample_shifted, exclusion_rad, nbrs_num, True)
        elif method == "smap":
            theta = kwargs["theta"]
            _, pred = self.__smap_prediction(lib_indices, smpl_indices, X_lib, X_sample, Y_lib_shifted, Y_sample_shifted, exclusion_rad, theta, True)

        return pred.to("cpu").numpy()



    def __simplex_prediction(self, lib_indices, smpl_indices, X_lib, X_sample, Y_lib_shifted, Y_sample_shifted, exclusion_rad, nbrs_num, return_pred=False):
        num_ts_X = X_lib.shape[0]
        num_ts_Y = Y_lib_shifted.shape[0]
        max_E_Y = Y_lib_shifted.shape[2]

        # Find indices of a neighbors of X_sample among X_lib
        indices = self.__get_nbrs_indices(X_lib, X_sample, nbrs_num, lib_indices, smpl_indices, exclusion_rad)
        # Reshaping for comfortable usage
        I = indices.reshape(num_ts_X,-1).T 

        # Pairwise crossmapping of all indices of embedding X to all embeddings of Y_shifted. Unreadble but optimized. 
        # Match every pair of Y_shifted i-th embedding with indices of X j-th 
        #Y_lib_shifted_indexed = torch.permute(Y_lib_shifted,(1,2,0))[I[:, None,None, :],torch.arange(max_E_Y,device=self.device)[:,None,None], torch.arange(num_ts_Y,device=self.device)[None,:,None]]
        Y_lib_shifted_indexed = torch.permute(Y_lib_shifted[:,I],(1,3,0,2))
        
        # Average across nearest neighbors to get a prediction
        A = Y_lib_shifted_indexed.reshape(-1, nbrs_num, max_E_Y, num_ts_Y, num_ts_X).mean(axis=1)
        B = torch.permute(Y_sample_shifted,(1,2,0))[:,:,:,None].expand(Y_sample_shifted.shape[1], max_E_Y, num_ts_Y, num_ts_X)
        
        # Calculate correlation between all pairs of the real i-th Y and predicted i-th Y using crossmapping from j-th X 
        r_AB = self.__get_batch_corr(A, B)

        if return_pred:
            return r_AB, A
        else:
            return r_AB

    def __smap_prediction(self, lib_indices, smpl_indices, X_lib, X_sample, Y_lib_shifted, Y_sample_shifted, exclusion_rad, theta, return_pred=False):
        num_ts_X = X_lib.shape[0]
        num_ts_Y = Y_lib_shifted.shape[0]
        max_E_X = X_lib.shape[2]
        max_E_Y = Y_lib_shifted.shape[2]
        subsample_size = X_sample.shape[1]
        subset_size = X_lib.shape[1]

        #sample_X_t = sample_X.permute(2, 0, 1)
        #subset_X_t = subset_X.permute(2, 0, 1)
        #subset_y_t = subset_y.permute(2, 0, 1)
        
        weights = self.__get_local_weights(X_lib,X_sample,lib_indices, smpl_indices, exclusion_rad, theta)
        W = weights.unsqueeze(1).expand(num_ts_X, num_ts_Y, subsample_size, subset_size).reshape(num_ts_X * num_ts_Y * subsample_size, subset_size, 1)

        X = X_lib.unsqueeze(1).unsqueeze(1).expand(num_ts_X, num_ts_Y, subsample_size, subset_size, max_E_X)
        X = X.reshape(num_ts_X * num_ts_Y * subsample_size, subset_size, max_E_X)

        Y = Y_lib_shifted.unsqueeze(1).unsqueeze(0).expand(num_ts_X, num_ts_Y, subsample_size, subset_size, max_E_Y)
        Y = Y.reshape(num_ts_X * num_ts_Y * subsample_size, subset_size, max_E_Y)

        X_intercept = torch.cat([torch.ones((num_ts_X * num_ts_Y * subsample_size, subset_size, 1),device=self.device), X], dim=2)
        
        X_intercept_weighted = X_intercept * W
        Y_weighted = Y * W

        XTWX = torch.bmm(X_intercept_weighted.transpose(1, 2), X_intercept_weighted)
        XTWy = torch.bmm(X_intercept_weighted.transpose(1, 2), Y_weighted)
        beta = torch.bmm(torch.pinverse(XTWX), XTWy)
        #beta_ = beta.reshape(dim,dim,sample_size,*beta.shape[1:])

        X_ = X_sample.unsqueeze(1).expand(num_ts_X, num_ts_Y, subsample_size, max_E_X)
        X_ = X_.reshape(num_ts_X * num_ts_Y * subsample_size, max_E_X)
        X_ = torch.cat([torch.ones((num_ts_X * num_ts_Y * subsample_size, 1),device=self.device), X_], dim=1)
        X_ = X_.reshape(num_ts_X * num_ts_Y * subsample_size, 1, max_E_X+1)
        
        #A = torch.einsum('abpij,bcpi->abcpj', beta, X_)
        #A = torch.permute(A[:,0],(2,3,1,0))

        A = torch.bmm(X_, beta).reshape(num_ts_X, num_ts_Y, subsample_size, max_E_Y)
        A = torch.permute(A,(2,3,1,0))

        B = torch.permute(Y_sample_shifted,(1,2,0)).unsqueeze(-1).expand(subsample_size, max_E_Y, num_ts_Y, num_ts_X)
        #TODO: test whether B = sample_y.unsqueeze(-2).expand(sample_size, E_y, dim, dim)
        
        r_AB = self.__get_batch_corr(A,B)

        if return_pred:
            return r_AB, A
        else:
            return r_AB

    def __get_random_indices(self, num_points, sample_len):
        idxs_X = torch.argsort(torch.rand(num_points,device=self.device))[0:sample_len]

        return idxs_X


    def __get_random_sample(self, X, min_len, indices, dim, max_E):
        X_buf = torch.zeros((dim, indices.shape[0], max_E),device=self.device)

        for i in range(dim):
            X_buf[i,:,:X[i].shape[-1]] = torch.tensor(X[i][-min_len:],device=self.device)[indices]

        return X_buf


    def __get_nbrs_indices(self, lib, sample, n_nbrs, lib_idx, sample_idx, exclusion_rad):
        dist = torch.cdist(sample,lib)
        # Find N + 2*excl_rad neighbors
        indices = torch.topk(dist, n_nbrs + 2*exclusion_rad, largest=False)[1]
        if exclusion_rad > 0:
            # Among random sample (real) indices mask that are not within the exclusion radius
            mask = ~((lib_idx[indices] < (sample_idx[:,None]+exclusion_rad)) & (lib_idx[indices] > (sample_idx[:,None]-exclusion_rad)))
            # Count the number of selected indices
            cumsum_mask = mask.cumsum(dim=2)
            # Select the first n_nbrs neighbors that are outside of the exclusion radius
            selector = cumsum_mask <= n_nbrs
            selector = selector * mask
            
            indices_exc = indices[selector].view(mask.shape[0],mask.shape[1],n_nbrs)
            return indices_exc
        else:
            return indices

    def __get_local_weights(self, lib, sublib, subset_idx, sample_idx, exclusion_rad, theta):
        dist = torch.cdist(sublib,lib)
        if theta == None:
            weights = torch.exp(-(dist))
        else:
            weights = torch.exp(-(theta*dist/dist.mean(axis=2)[:,:,None]))

        if exclusion_rad > 0:
            exclusion_matrix = (torch.abs(subset_idx[None] - sample_idx[:,None]) > exclusion_rad)
            weights = weights * exclusion_matrix
        
        return weights

    def __get_batch_corr(self,A, B):
        mean_A = torch.mean(A,axis=0)
        mean_B = torch.mean(B,axis=0)
        
        sum_AB = torch.sum((A - mean_A[None,:,:]) * (B - mean_B[None,:,:]),axis=0)
        sum_AA = torch.sum((A - mean_A[None,:,:]) ** 2,axis=0)
        sum_BB = torch.sum((B - mean_B[None,:,:]) ** 2,axis=0)
        
        r_AB = sum_AB / torch.sqrt(sum_AA * sum_BB)
        return r_AB
    
    def __get_batch_rmse(self, A, B):
        """
        Computes the batch-wise Root Mean Square Error (RMSE) between two 4D tensors A and B.
        
        Args:
        A, B: Tensors of shape [num points, num dims, num components, num components].
        
        Returns:
        Tensor of RMSE values with shape [num dims, num components, num components].
        """
        # Compute the squared differences between A and B
        squared_diff = (A - B) ** 2
        
        # Compute the mean of the squared differences along the num points axis
        mean_squared_diff = torch.mean(squared_diff, dim=0)
        
        # Compute the square root of the mean squared differences
        rmse = torch.sqrt(mean_squared_diff)
        
        return rmse
    
    def __get_batch_mse(self, A, B):
        """
        Computes the batch-wise Mean Squared Error (MSE) between two 4D tensors A and B.
        
        Args:
        A, B: Tensors of shape [num points, num dims, num components, num components].
        
        Returns:
        Tensor of MSE values with shape [num dims, num components, num components].
        """
        # Compute the squared differences between A and B
        squared_diff = (A - B) ** 2
        
        # Compute the mean of the squared differences along the num points axis
        mse = torch.mean(squared_diff, dim=0)
        
        return mse
