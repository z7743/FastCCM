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
    
    def get_weights(self):

        return torch.permute(self.model.weight.reshape(-1,self.embed_dim,self.output_dim), dims=(0,2,1)).cpu().detach().numpy()



class IMD_nD:

    def __init__(self, input_dim, embed_dim, n_components, learning_rate=0.001,device="cuda",optimizer="Adam",random_state=None):
        self.device = device

        self.model = LinearProjectionNDim(input_dim, embed_dim, n_components, device,random_state)

        self.optimizer = getattr(optim, optimizer)(self.model.parameters(), lr=learning_rate,)
        #self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate,)

        self.loss_history = []

    def fit(self, X, sample_len, library_len, nbrs_num, tp, epochs=10, num_batches=32):

        X = torch.tensor(X,requires_grad=True, device="cpu", dtype=torch.float32)
        X_train, y_train = X[:X.shape[0]-tp], X[tp:]

        dataset = RandomSubsetDataset(X_train, y_train, sample_len, library_len, num_batches,device=self.device)
        dataloader = DataLoader(dataset, batch_size=1,pin_memory=False)
        
        for epoch in range(epochs):
            total_loss = 0
            self.optimizer.zero_grad()

            for subset_X, subset_y, sample_X, sample_y in dataloader:
                subset_X_z = self.model(subset_X)
                subset_y_z = self.model(subset_y)
                sample_X_z = self.model(sample_X)
                sample_y_z = self.model(sample_y)

                loss = self.loss_fn(sample_X_z, sample_y_z, subset_X_z, subset_y_z, nbrs_num)
                loss /= num_batches
                loss.backward()
                total_loss += loss.item() 

            self.optimizer.step()

            print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}')
            self.loss_history += [total_loss]

    def predict(self, X):
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32,device=self.device)
            outputs = torch.permute(self.model(inputs),dims=(0,2,1)) #Easier to interpret
        return outputs.cpu().numpy()


    def loss_fn(self, sample_td, sample_pred, subset_td, subset_pred, nbrs_num):
        dim = sample_td.shape[-1]
        corr = torch.abs(self.get_autoreg_matrix(sample_td, sample_pred))

        ccm = torch.abs(self.get_ccm_matrix(sample_td, sample_pred, subset_td, subset_pred, nbrs_num))
        mask = torch.eye(dim,dtype=bool,device=self.device)
        if dim > 1:
            score = 1 + (torch.mean(ccm[:,~mask].reshape(-1,dim,dim-1),axis=2)/2 + \
                        torch.mean(ccm[:,~mask].reshape(-1,dim-1,dim),axis=1)/2 - \
                    (ccm[:,mask]**2) \
                    +(corr[:,mask]**2)
                        ).mean()
        else:
            score = 1 + (-ccm[:,0,0] + corr[:,0,0]).mean()
        return score
    

    def get_ccm_matrix(self, sample_td, sample_pred, subset_td, subset_pred, nbrs_num):
        dim = sample_td.shape[-1]
        E = sample_td.shape[-2]
        
        indices = self.get_nbrs_indices(torch.permute(subset_td,(2,0,1)),torch.permute(sample_td,(2,0,1)), nbrs_num)
        I = indices.reshape(dim,-1).T 
        
        subset_pred_indexed = subset_pred[I[:, None,None, :],torch.arange(E,device=self.device)[:,None,None], torch.arange(dim,device=self.device)[None,:,None]]
        
        A = subset_pred_indexed.reshape(-1, nbrs_num, E, dim, dim).mean(axis=1)
        B = sample_pred[:,:,None,:,].expand(sample_pred.shape[0], E, dim, dim)
        
        r_AB = self.get_batch_corr(A,B)
        return r_AB
    
    def get_autoreg_matrix(self, A, B):
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

    def get_nbrs_indices(self, lib, sublib, n_nbrs):
        dist = torch.cdist(sublib,lib)
        indices = torch.topk(dist, n_nbrs, largest=False)[1]
        return indices

    def get_loss_history(self):
        return self.loss_history
    

class RandomSubsetDataset(Dataset):
    def __init__(self, X, y, sample_len, library_len, num_batches, device="cuda"):
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
    
    def p__getitem__(self, idx):
        sample_idx = torch.argsort(torch.rand(self.num_datapoints,device=self.device))[0:self.sample_len]
        library_idx = torch.argsort(torch.rand(self.num_datapoints,device=self.device))[0:self.library_len+self.sample_len]
        library_idx = library_idx[(library_idx.view(1, -1) != sample_idx.view(-1, 1)).all(dim=0)][0:self.library_len]
        
        return self.X[library_idx],self.y[library_idx], self.X[sample_idx], self.y[sample_idx]
    
    def __getitem__(self, idx):
        sample_idx = torch.argsort(torch.rand(self.num_datapoints))[0:self.sample_len]
        library_idx = torch.argsort(torch.rand(self.num_datapoints))[0:self.library_len+self.sample_len]
        library_idx = library_idx[(library_idx.view(1, -1) != sample_idx.view(-1, 1)).all(dim=0)][0:self.library_len]

        return self.X[library_idx].to(self.device), \
                self.y[library_idx].to(self.device), \
                self.X[sample_idx].to(self.device), \
                self.y[sample_idx].to(self.device)
