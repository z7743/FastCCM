import torch
import numpy as np

def get_td_embedding_torch(ts, dim, stride, return_pred=False, tp=0):
    tdemb = ts.unfold(0,(dim-1) * stride + 1,1)[...,::stride]
    tdemb = torch.swapaxes(tdemb,-1,-2)
    if return_pred:
        return tdemb[:tdemb.shape[0]-tp], ts[(dim-1) * stride + tp:]
    else:
        return tdemb
    
def get_td_embedding_np(time_series, dim, stride, return_pred=False, tp=0):
    num_points, num_dims = time_series.shape
    # Calculate the size of the unfolding window
    window_size = (dim - 1) * stride + 1
    
    # Ensure the time series is long enough for the unfolding
    if num_points < window_size:
        raise ValueError("Time series is too short for the given dimensions and stride.")
    
    # Create an array to hold the unfolded data
    # Calculate shape and strides for as_strided
    shape = (num_points - window_size + 1, dim, num_dims)
    strides = (time_series.strides[0], stride * time_series.strides[0], time_series.strides[1])
    tdemb = np.lib.stride_tricks.as_strided(time_series, shape=shape, strides=strides)
    
    if return_pred:
        # Return the embedded data excluding the last 'tp' points, and the prediction points starting from a specific index
        return tdemb[:tdemb.shape[0]-tp], time_series[(dim - 1) * stride + tp:]
    else:
        return tdemb
    

def get_td_embedding_specified(time_series, delays):
    """
    [ChatGPT written]
    Embeds a time series using specified time delays by truncating the beginning of the series.

    Parameters:
    time_series (array-like): The input time series data.
    delays (array-like): An array of time delays.

    Returns:
    np.ndarray: A matrix where each column is the time series delayed by the respective delay.
    """
    n = len(time_series)
    max_delay = max(delays)
    
    # Ensure the time series is long enough for the maximum delay
    if max_delay >= n:
        raise ValueError("Maximum delay exceeds the length of the time series.")

    # Create an embedding matrix with shape (n - max_delay, len(delays))
    embedded = np.empty((n - max_delay, len(delays)))
    
    for i, delay in enumerate(delays):
        embedded[:, i] = time_series[max_delay - delay:n - delay]

    return embedded