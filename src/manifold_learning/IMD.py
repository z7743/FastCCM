import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class LinearProjectionNDim(nn.Module):
    def __init__(self, input_dim, embed_dim, output_dim, device="cuda",random_state=None):
        """
        Initializes the linear projection module.
        
        Args:
            input_dim (int): The dimension of the input data.
            embed_dim (int): The dimension of the embedding.
            output_dim (int): The dimension of the output projection.
            device (str, optional): Device to place the model on, default is "cuda".
            random_state (int): Ignored if None.
        """
        super(LinearProjectionNDim, self).__init__()
        self.device = device

        self.embed_dim = embed_dim
        self.output_dim = output_dim

        if random_state != None:
            torch.manual_seed(random_state)
        self.model = nn.Linear(input_dim, output_dim*embed_dim, bias=False,device=self.device,)

    def forward(self, x):
        """
        Forward pass of the module.
        
        Args:
            x (torch.Tensor): Input tensor of shape `(batch_size, input_dim)`.
        
        Returns:
            torch.Tensor: Output tensor of shape `(batch_size, embed_dim, output_dim)`.
        """
        x = self.model(x).reshape(-1,self.embed_dim,self.output_dim)

        return x
    
    def backward(self, y):
        """
        Backward pass using the pseudoinverse.
        
        Args:
            y (torch.Tensor): Output tensor of shape `(batch_size, embed_dim, output_dim)`.
        
        Returns:
            torch.Tensor: Input tensor of shape `(batch_size, input_dim)`.
        """
        batch_size = y.shape[0]
        y_flat = y.reshape(batch_size, -1)

        # Compute the pseudoinverse of the weight matrix
        weight = self.model.weight
        weight_pinv = torch.pinverse(weight).to(self.device)

        # Compute the input using the pseudoinverse
        x_reconstructed = y_flat @ weight_pinv.T

        return x_reconstructed

    
    def get_weights(self):
        """
        Retrieves the weights of the model.
        
        Returns:
            numpy.ndarray: Weights of the model reshaped and permuted.
        """
        return torch.permute(self.model.weight.reshape(-1,self.embed_dim,self.output_dim), dims=(0,2,1)).cpu().detach().numpy()



class IMD_nD:

    def __init__(self, input_dim, embed_dim, n_components, subtract_corr=True,device="cuda",random_state=None):
        """
        Initializes the IMD_nD model.
        
        Args:
            input_dim (int): Dimension of the input data.
            embed_dim (int): Dimension of the embedding of the component.
            n_components (int): Number of components.
            subtract_corr (bool, optional): Whether to subtract correlation, default is True.
            device (str, optional): Device to place the model on, default is "cuda".
            random_state (int): Ignored if None.
        """
        self.device = device

        self.model = LinearProjectionNDim(input_dim, embed_dim, n_components, device,random_state)

        self.subtract_corr = subtract_corr
        self.loss_history = []

    def fit(self, X, sample_len, library_len, exclusion_rad, nbrs_num, tp=1, epochs=100, num_batches=32, optimizer="Adam", learning_rate=0.001, tp_policy="range"):
        """
        Fits the model to the data.

        Args:
            X (numpy.ndarray): Input data.
            sample_len (int): Length of samples used for prediction.
            library_len (int): Length of library used for kNN search.
            exclusion_rad (int): Exclusion radius for kNN search.
            nbrs_num (int): Number of neighbors for kNN search.
            tp (int): Time lag for prediction.
            epochs (int, optional): Number of training epochs, default is 100.
            num_batches (int, optional): Number of batches, default is 32.
            optimizer (str, optional): Optimizer to use, default is "Adam".
            learning_rate (float, optional): Learning rate for the optimizer, default is 0.001.
            tp_policy (str, optional): Batch sampling policy, default is "range". If "range" then the embedding will be optimized for the range of 1...tp and the total number of samplings calculated as num_batches*tp. If "fixed" then within one optimization cycle the embedding will be optimized only for one value of tp.
        """
        X = torch.tensor(X,requires_grad=True, device=self.device, dtype=torch.float32)
        X_train, y_train = X[:X.shape[0]-tp], X[tp:]

        if tp_policy == "fixed":
            dataset = RandomSubsetDataset(X_train, y_train, sample_len, library_len, num_batches,device=self.device)
        elif tp_policy == "range":
            dataset = RandomSubsetDatasetTpRange(X, sample_len, library_len, num_batches, tp, device=self.device)
        else:
            pass #TODO: exception

        dataloader = DataLoader(dataset, batch_size=1,pin_memory=False)

        optimizer = getattr(optim, optimizer)(self.model.parameters(), lr=learning_rate,)

        for epoch in range(epochs):
            total_loss = 0
            optimizer.zero_grad()

            for subset_idx, sample_idx, subset_X, subset_y, sample_X, sample_y in dataloader:
                subset_X_z = self.model(subset_X)
                subset_y_z = self.model(subset_y)
                sample_X_z = self.model(sample_X)
                sample_y_z = self.model(sample_y)

                loss = self.loss_fn(subset_idx, sample_idx,sample_X_z, sample_y_z, subset_X_z, subset_y_z, nbrs_num, exclusion_rad)
                
                loss /= tp * num_batches
                loss.backward()
                total_loss += loss.item() 

            optimizer.step()

            print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}')
            self.loss_history += [total_loss]

    def predict(self, X):
        """
        Calculates embeddings using the trained model.
        
        Args:
            X (numpy.ndarray): Input data.
        
        Returns:
            numpy.ndarray: Predicted outputs.
        """
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32,device=self.device)
            outputs = torch.permute(self.model(inputs),dims=(0,2,1)) #Easier to interpret
        return outputs.cpu().numpy()

    def generate(self, X, nbrs_num, exclusion_rad, tp=1):
        """
        Generates the prediction using the model.
        
        Args:
            X (numpy.ndarray): Input data.
            nbrs_num (int): Number of neighbors for kNN search.
            exclusion_rad (int): Exclusion radius for kNN search.
            tp (int, optional): Time lag for prediction, default is 1.
        
        Returns:
            tuple: Generated results and the reconstructed input.
        """
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32,device=self.device)
            X = self.model(inputs)
            lib = torch.permute(X,dims=(2,0,1))

            dist = torch.cdist(lib,lib[:,:-tp])
            indices = torch.topk(dist, nbrs_num + 2 * exclusion_rad, largest=False)[1]

            mask = (indices >= (torch.arange(X.shape[0],device=self.device) + exclusion_rad)[None,:,None]) | (indices <= (torch.arange(X.shape[0],device=self.device) - exclusion_rad)[None,:,None])
            cumsum_mask = mask.cumsum(dim=2)
            selector = cumsum_mask <= nbrs_num
            selector = selector * mask
            indices = indices[selector].view(mask.shape[0],mask.shape[1],nbrs_num)

            indices += tp
            subset_pred_indexed = lib[torch.arange(X.shape[-1],device=self.device)[:,None,None],indices]
            res = torch.permute(subset_pred_indexed.mean(axis=2), dims=(1,2,0))

        return res, self.model.backward(res)

    def loss_fn(self, subset_idx, sample_idx, sample_td, sample_pred, subset_td, subset_pred, nbrs_num, exclusion_rad):
        dim = sample_td.shape[-1]
        ccm = torch.abs(self.get_ccm_matrix_approx(subset_idx, sample_idx, sample_td, sample_pred, subset_td, subset_pred, nbrs_num, exclusion_rad))
        mask = torch.eye(dim,dtype=bool,device=self.device)

        if self.subtract_corr:
            corr = torch.abs(self.get_autoreg_matrix_approx(sample_pred, sample_td))
            if dim > 1:
                score = 1 + (torch.mean(ccm[:,~mask].reshape(-1,dim,dim-1),axis=2)/2 + \
                            torch.mean(ccm[:,~mask].reshape(-1,dim-1,dim),axis=1)/2 - \
                        (ccm[:,mask]**2) \
                        +(corr[:,mask]**2)
                            ).mean()
            else:
                score = 1 + (-ccm[:,0,0] + corr[:,0,0]).mean()
            return score
        else:
            if dim > 1:
                score = 1 + (torch.mean(ccm[:,~mask].reshape(-1,dim,dim-1),axis=2)/2 + \
                            torch.mean(ccm[:,~mask].reshape(-1,dim-1,dim),axis=1)/2 - \
                        (ccm[:,mask]**2)).mean()
            else:
                score = 1 + (-ccm[:,0,0]).mean()
            return score

    def get_ccm_matrix_approx(self, subset_idx, sample_idx, sample_td, sample_pred, subset_td, subset_pred, nbrs_num, exclusion_rad):
        dim = sample_td.shape[-1]
        E = sample_td.shape[-2]
        
        indices = self.get_nbrs_indices(torch.permute(subset_td,(2,0,1)),torch.permute(sample_td,(2,0,1)), nbrs_num, subset_idx, sample_idx, exclusion_rad)
        I = indices.reshape(dim,-1).T 
        
        subset_pred_indexed = subset_pred[I[:, None,None, :],torch.arange(E,device=self.device)[:,None,None], torch.arange(dim,device=self.device)[None,:,None]]
        
        A = subset_pred_indexed.reshape(-1, nbrs_num, E, dim, dim).mean(axis=1)
        B = sample_pred[:,:,None,:,].expand(sample_pred.shape[0], E, dim, dim)
        
        r_AB = self.get_batch_corr(A,B)
        return r_AB
    
    def get_autoreg_matrix_approx(self, A, B):
        dim = A.shape[-1]
        E = A.shape[-2]
        
        A = A[:,:,:,None].expand(-1, E, dim, dim)
        B = B[:,:,None,:].expand(-1, E, dim, dim)

        r_AB = self.get_batch_corr(A,B)
        return r_AB
    
    def get_batch_corr(self,A, B):
        mean_A = torch.mean(A,axis=0)
        mean_B = torch.mean(B,axis=0)
        
        sum_AB = torch.sum((A - mean_A[None,:,:]) * (B - mean_B[None,:,:]),axis=0)
        sum_AA = torch.sum((A - mean_A[None,:,:]) ** 2,axis=0)
        sum_BB = torch.sum((B - mean_B[None,:,:]) ** 2,axis=0)
        
        r_AB = sum_AB / torch.sqrt(sum_AA * sum_BB)
        return r_AB
    
    def get_batch_cosine_similarity(self,A, B):
        """
        Computes the batch-wise cosine similarity between two 4D tensors A and B.
        
        Args:
        A, B: Tensors of shape [num points, num dims, num components, num components].
        
        Returns:
        Tensor of cosine similarities with shape [num dims, num components, num components].
        """
        # Compute the norms of A and B along the num points axis
        norm_A = torch.norm(A, p=2, dim=0, keepdim=True)
        norm_B = torch.norm(B, p=2, dim=0, keepdim=True)
        
        # Avoid division by zero
        norm_A = torch.where(norm_A == 0, torch.ones_like(norm_A), norm_A)
        norm_B = torch.where(norm_B == 0, torch.ones_like(norm_B), norm_B)
        
        # Normalize A and B
        A_normalized = A / norm_A
        B_normalized = B / norm_B
        
        # Compute the dot product between normalized A and B
        dot_product = torch.sum(A_normalized * B_normalized, dim=0)
        
        return dot_product

    def get_nbrs_indices(self, lib, sublib, n_nbrs, subset_idx, sample_idx, exclusion_rad):
        dist = torch.cdist(sublib,lib)
        indices = torch.topk(dist, n_nbrs + 2*exclusion_rad, largest=False)[1]
        if exclusion_rad > 0:
            
            mask = ~((subset_idx[0][indices] <= sample_idx[0][:,None]+exclusion_rad) & (subset_idx[0][indices] >= sample_idx[0][:,None]-exclusion_rad))
            cumsum_mask = mask.cumsum(dim=2)
            selector = cumsum_mask <= n_nbrs
            selector = selector * mask
            
            indices_exc = indices[selector].view(mask.shape[0],mask.shape[1],n_nbrs)
            return indices_exc
        else:
            return indices

    def get_loss_history(self):
        return self.loss_history
    

class RandomSubsetDataset(Dataset):
    def __init__(self, X, y, sample_len, library_len, num_batches=32, device="cuda"):
        """
        Initializes the RandomSubsetDataset.
        
        Args:
            X (torch.Tensor): Multivariate time series to be sampled.
            y (torch.Tensor): Multivariate time series coupled with X.
            sample_len (int): Number of samples used for prediction.
            library_len (int): Number of samples used for kNN search.
            num_batches (int): Number of random batches to be produced.
            device (str, optional): Device to place the dataset on, default is "cuda".
        """
        self.device = device
        self.X = X
        self.y = y
        self.sample_len = sample_len
        self.library_len = library_len
        self.num_batches = num_batches
        self.num_datapoints = X.shape[0]

    def __len__(self):
        return self.num_batches #Temporary solution to sample number of samples
    
    def __getitem__(self, idx):
        sample_idx = torch.argsort(torch.rand(self.num_datapoints,device=self.device))[0:self.sample_len]
        library_idx = torch.argsort(torch.rand(self.num_datapoints,device=self.device))[0:self.library_len + self.sample_len]
        library_idx = library_idx[(library_idx.view(1, -1) != sample_idx.view(-1, 1)).all(dim=0)][0:self.library_len]
        
        return library_idx, sample_idx, self.X[library_idx],self.y[library_idx], self.X[sample_idx], self.y[sample_idx]

class RandomSubsetDatasetTpRange(Dataset):
    def __init__(self, X, sample_len, library_len, num_batches=32, tp=1, device="cuda"):
        """
        Initializes the RandomSubsetDataset.
        
        Args:
            X (torch.Tensor): Multivariate time series to be sampled.
            y (torch.Tensor): Multivariate time series coupled with X.
            sample_len (int): Number of samples used for prediction.
            library_len (int): Number of samples used for kNN search.
            num_batches (int): Number of random batches to be produced.
            device (str, optional): Device to place the dataset on, default is "cuda".
        """
        self.device = device
        self.X = X
        self.sample_len = sample_len
        self.library_len = library_len
        self.tp = tp
        self.num_batches = num_batches
        self.num_datapoints = X.shape[0]

    def __len__(self):
        return self.tp * self.num_batches#Temporary solution to sample number of samples
    
    def __getitem__(self, idx):
        sample_idx = torch.argsort(torch.rand(self.num_datapoints-self.tp-1,device=self.device))[0:self.sample_len]
        library_idx = torch.argsort(torch.rand(self.num_datapoints-self.tp-1,device=self.device))[0:self.library_len + self.sample_len]
        library_idx = library_idx[(library_idx.view(1, -1) != sample_idx.view(-1, 1)).all(dim=0)][0:self.library_len]
        
        return library_idx, sample_idx, self.X[library_idx],self.X[library_idx+idx%self.tp+1], self.X[sample_idx], self.X[sample_idx+idx%self.tp+1]
    