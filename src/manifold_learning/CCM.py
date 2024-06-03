import torch

class CCM:
    def __init__(self,device = "cpu"):
        self.device = device

    def compute(self, X, Y, library_len, sample_len, exclusion_rad, nbrs_num, tp=0):
        # Number of time series 
        num_ts_X = X.shape[0]
        num_ts_Y = Y.shape[0]
        # Max embedding dimension
        max_E = torch.tensor([Y[i].shape[-1] for i in range(num_ts_Y)] + [X[i].shape[-1] for i in range(num_ts_X)]).max().item()
        # Max common length
        min_len = torch.tensor([Y[i].shape[0] for i in range(num_ts_Y)] + [X[i].shape[0] for i in range(num_ts_X)]).min().item()

        # Random indices for sampling
        lib_indices = self.get_random_indices(min_len, library_len, tp)
        smpl_indices = self.get_random_indices(min_len, sample_len, tp)

        # Select X_lib and X_sample at time t and Y_lib, Y_sample at time t+tp
        X_lib = self.get_random_sample(X, min_len, lib_indices, num_ts_X, max_E)
        X_sample = self.get_random_sample(X, min_len, smpl_indices, num_ts_X, max_E)
        Y_lib_shifted = self.get_random_sample(Y, min_len, lib_indices+tp, num_ts_Y, max_E)
        Y_sample_shifted = self.get_random_sample(Y,min_len, smpl_indices+tp, num_ts_Y, max_E)

        # Find indices of a neighbors of X_sample among X_lib
        indices = self.get_nbrs_indices(X_lib,X_sample,nbrs_num,lib_indices,smpl_indices, exclusion_rad)

        I = indices.reshape(num_ts_X,-1).T 

        # Pairwise crossmapping of all indices of X to all embeddings of Y
        subset_pred_indexed = torch.permute(Y_lib_shifted,(1,2,0))[I[:, None,None, :],torch.arange(max_E,device=self.device)[:,None,None], torch.arange(num_ts_Y,device=self.device)[None,:,None]]
        
        A = subset_pred_indexed.reshape(-1, nbrs_num, max_E, num_ts_Y, num_ts_X).mean(axis=1)
        B = torch.permute(Y_sample_shifted,(1,2,0))[:,:,:,None].expand(Y_sample_shifted.shape[1], max_E, num_ts_Y,num_ts_X)
        
        r_AB = self.get_batch_corr(A, B)
        return r_AB.to("cpu")

    def get_random_indices(self, num_points, sample_len, tp):
        idxs_X = torch.argsort(torch.rand(num_points-tp,device=self.device))[0:sample_len]

        return idxs_X

    def get_random_sample(self, X, min_len, indices, dim, max_E):
        X_buf = torch.zeros((dim, indices.shape[0], max_E),device=self.device)

        for i in range(dim):
            X_buf[i,:,:X[i].shape[-1]] = torch.tensor(X[i][-min_len:],device=self.device)[indices]

        return X_buf


    def get_nbrs_indices(self, lib, sample, n_nbrs, subset_idx, sample_idx, exclusion_rad):
        dist = torch.cdist(sample,lib)
        indices = torch.topk(dist, n_nbrs + 2*exclusion_rad, largest=False)[1]
        if exclusion_rad > 0:
            
            mask = ~((subset_idx[indices] < (sample_idx[:,None]+exclusion_rad)) & (subset_idx[indices] > (sample_idx[:,None]-exclusion_rad)))
            cumsum_mask = mask.cumsum(dim=2)
            selector = cumsum_mask <= n_nbrs
            selector = selector * mask
            
            indices_exc = indices[selector].view(mask.shape[0],mask.shape[1],n_nbrs)
            return indices_exc
        else:
            return indices

    def get_batch_corr(self,A, B):
        mean_A = torch.mean(A,axis=0)
        mean_B = torch.mean(B,axis=0)
        
        sum_AB = torch.sum((A - mean_A[None,:,:]) * (B - mean_B[None,:,:]),axis=0)
        sum_AA = torch.sum((A - mean_A[None,:,:]) ** 2,axis=0)
        sum_BB = torch.sum((B - mean_B[None,:,:]) ** 2,axis=0)
        
        r_AB = sum_AB / torch.sqrt(sum_AA * sum_BB)
        return r_AB
    