import torch

class CCM:
    def __init__(self,device = "cpu"):
        self.device = device

    def get_matrix(self, X, library_len, sample_len, exclusion_rad, nbrs_num, tp=0):
        dim = X.shape[0]
        max_E = torch.tensor([X[i].shape[-1] for i in range(dim)]).max().item()
        min_len = torch.tensor([X[i].shape[0] for i in range(dim)]).min().item()

        X_lib_indices = self.get_random_indices(min_len, library_len, tp)
        X_smpl_indices = self.get_random_indices(min_len, sample_len, tp)

        X_lib, Y_lib = self.get_random_sample(X, X_lib_indices, dim, max_E, tp)
        X_sample, Y_sample = self.get_random_sample(X, X_smpl_indices, dim, max_E, tp)

        indices = self.get_nbrs_indices(X_lib,X_sample,nbrs_num,X_lib_indices,X_smpl_indices, exclusion_rad)

        I = indices.reshape(dim,-1).T 
        
        subset_pred_indexed = torch.permute(Y_lib,(1,2,0))[I[:, None,None, :],torch.arange(max_E,device=self.device)[:,None,None], torch.arange(dim,device=self.device)[None,:,None]]
        
        A = subset_pred_indexed.to("cpu").reshape(-1, nbrs_num, max_E, dim, dim).mean(axis=1)
        B = torch.permute(Y_sample,(1,2,0))[:,:,None,:,].to("cpu").expand(Y_sample.shape[1], max_E, dim, dim)
        
        r_AB = self.get_batch_corr(A,B)
        return r_AB.to("cpu")


    def get_random_sample(self, X, indices, dim, max_E, tp):
        X_buf = torch.zeros((dim, indices.shape[0], max_E),device=self.device)
        Y_buf = torch.zeros((dim, indices.shape[0], max_E),device=self.device)

        for i in range(dim):
            X_buf[i] = torch.tensor(X[i],device=self.device)[indices]
            Y_buf[i] = torch.tensor(X[i],device=self.device)[indices + tp]

        return X_buf, Y_buf

    def get_random_indices(self, num_points, sample_len, tp):
        idxs_X = torch.argsort(torch.rand(num_points-tp-1,device=self.device))[0:sample_len]

        return idxs_X

    def get_nbrs_indices(self, lib, sample, n_nbrs, subset_idx, sample_idx, exclusion_rad):
        dist = torch.cdist(sample,lib)
        indices = torch.topk(dist, n_nbrs + 2*exclusion_rad, largest=False)[1]
        if exclusion_rad > 0:
            
            mask = ~((subset_idx[indices] <= sample_idx[:,None]+exclusion_rad) & (subset_idx[indices] >= sample_idx[:,None]-exclusion_rad))
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
    