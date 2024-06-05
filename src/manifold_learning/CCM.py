import torch

class FastCCM:
    def __init__(self,device = "cpu"):
        """
        Constructs a FastCCM object to perform Convergent Cross Mapping (CCM) using PyTorch.
        This object is optimized for calculating pariwise CCM matrix and can handle large datasets by utilizing batch processing on GPUs or CPUs.

        Parameters:
            device (str): The computation device ('cpu' or 'cuda') to use for all calculations.
        """
        self.device = device

    def compute(self, X, Y, subset_size, subsample_size, exclusion_rad, nbrs_num, tp=0):
        """
        Main computation function for Convergent Cross Mapping (CCM).

        Parameters:
            X (list of np.array): List of embeddings from which to cross-map.
            Y (list of np.array): List of embeddings to predict.
            subset_size (int): Number of random samples of embeddings taken to approximate the shape of the manifold well enough. Nearest neighbors for cross-mapping will searched among this subset.
            subsample_size (int): Number of random samples of embeddings to estimate prediction quality. Nearest neighbors for these samples will be searched.
            exclusion_rad (int): Exclusion radius to avoid picking temporally close points from a subset.
            nbrs_num (int): Number of neighbors to consider for nearest neighbor calculations.
            tp (int): Interval of the prediction.

        Returns:
            np.array: A matrix of correlation coefficients between the real and predicted states.
        """
        # Number of time series 
        num_ts_X = len(X)
        num_ts_Y = len(Y)
        # Max embedding dimension
        max_E_X = torch.tensor([X[i].shape[-1] for i in range(num_ts_X)]).max().item()
        max_E_Y = torch.tensor([Y[i].shape[-1] for i in range(num_ts_Y)]).max().item()
        # Max common length
        min_len = torch.tensor([Y[i].shape[0] for i in range(num_ts_Y)] + [X[i].shape[0] for i in range(num_ts_X)]).min().item()

        # Random indices for sampling
        lib_indices = self.__get_random_indices(min_len - tp, subset_size)
        smpl_indices = self.__get_random_indices(min_len - tp, subsample_size)

        # Select X_lib and X_sample at time t and Y_lib, Y_sample at time t+tp
        X_lib = self.__get_random_sample(X, min_len, lib_indices, num_ts_X, max_E_X)
        X_sample = self.__get_random_sample(X, min_len, smpl_indices, num_ts_X, max_E_X)
        Y_lib_shifted = self.__get_random_sample(Y, min_len, lib_indices+tp, num_ts_Y, max_E_Y)
        Y_sample_shifted = self.__get_random_sample(Y,min_len, smpl_indices+tp, num_ts_Y, max_E_Y)

        # Find indices of a neighbors of X_sample among X_lib
        indices = self.__get_nbrs_indices(X_lib, X_sample, nbrs_num, lib_indices, smpl_indices, exclusion_rad)
        # Reshaping for comfortable usage
        I = indices.reshape(num_ts_X,-1).T 

        # Pairwise crossmapping of all indices of embedding X to all embeddings of Y_shifted. Unreadble but optimized. 
        # Match every pair of Y_shifted i-th embedding with indices of X j-th 
        Y_lib_shifted_indexed = torch.permute(Y_lib_shifted,(1,2,0))[I[:, None,None, :],torch.arange(max_E_Y,device=self.device)[:,None,None], torch.arange(num_ts_Y,device=self.device)[None,:,None]]
        
        # Average across nearest neighbors to get a prediction
        A = Y_lib_shifted_indexed.reshape(-1, nbrs_num, max_E_Y, num_ts_Y, num_ts_X).mean(axis=1)
        B = torch.permute(Y_sample_shifted,(1,2,0))[:,:,:,None].expand(Y_sample_shifted.shape[1], max_E_Y, num_ts_Y, num_ts_X)
        
        # Calculate correlation between all pairs of the real i-th Y and predicted i-th Y using crossmapping from j-th X 
        r_AB = self.__get_batch_corr(A, B)
        return r_AB.to("cpu").numpy()

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

    def __get_batch_corr(self,A, B):
        mean_A = torch.mean(A,axis=0)
        mean_B = torch.mean(B,axis=0)
        
        sum_AB = torch.sum((A - mean_A[None,:,:]) * (B - mean_B[None,:,:]),axis=0)
        sum_AA = torch.sum((A - mean_A[None,:,:]) ** 2,axis=0)
        sum_BB = torch.sum((B - mean_B[None,:,:]) ** 2,axis=0)
        
        r_AB = sum_AB / torch.sqrt(sum_AA * sum_BB)
        return r_AB
    