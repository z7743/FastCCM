import torch
import numpy as np

def get_td_embedding_torch(ts, dim, stride, return_pred=False, tp=0):
    tdemb = ts.unfold(0,(dim-1) * stride + 1,1)[...,::stride]
    tdemb = torch.swapaxes(tdemb,-1,-2)
    if return_pred:
        return tdemb[:tdemb.shape[0]-tp], ts[(dim-1) * stride + tp:]
    else:
        return tdemb
    
def get_td_embedding_np(ts, dim, stride, return_pred=False, tp=0):
    num_points, num_dims = ts.shape
    # Calculate the size of the unfolding window
    window_size = (dim - 1) * stride + 1
    
    # Ensure the time series is long enough for the unfolding
    if num_points < window_size:
        raise ValueError("Time series is too short for the given dimensions and stride.")
    
    # Create an array to hold the unfolded data
    # Calculate shape and strides for as_strided
    shape = (num_points - window_size + 1, dim, num_dims)
    strides = (ts.strides[0], stride * ts.strides[0], ts.strides[1])
    tdemb = np.lib.stride_tricks.as_strided(ts, shape=shape, strides=strides)
    
    if return_pred:
        # Return the embedded data excluding the last 'tp' points, and the prediction points starting from a specific index
        return tdemb[:tdemb.shape[0]-tp], ts[(dim - 1) * stride + tp:]
    else:
        return tdemb