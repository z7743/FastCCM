import pytest
import numpy as np
import time
import torch
from FastCCM import CCM
from FastCCM.utils import utils
from FastCCM.data.data_loader import get_truncated_lorenz_rand

### Helper functions
def prepare_embeddings(N_sys):
    X = np.concatenate([get_truncated_lorenz_rand(280, n_steps=30000)[:,[0]].T for _ in range(N_sys)])
    X_emb = [utils.get_td_embedding_np(x[:, None], 5, 10, return_pred=False)[:, :, 0] for x in X]
    Y_emb = [utils.get_td_embedding_np(x[:, None], 5, 10, return_pred=False)[:, [0], 0] for x in X]
    return X_emb, Y_emb

def time_ccm_computation(X_emb, Y_emb, trials=15, device="cpu"):
    torch.cuda.empty_cache()
    ccm = CCM.PairwiseCCM(device=device) 

    start_time = time.time()
    for _ in range(trials):
        ccm.compute(X_emb, Y_emb, 1000, 250, 30, 10, 0)
    total_time =  time.time() - start_time
    
    avg_time = total_time / trials * 1000
    time_per_pair = avg_time / len(X_emb) / len(Y_emb)
    
    return avg_time, time_per_pair

### Test function
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_time_ccm_computation(device):
    timeseries_counts = [1, 2, 5, 10, 15, 20, 30, 50, 100, 150, 200, 300]
    X_emb, Y_emb = prepare_embeddings(max(timeseries_counts))
    
    results = []
    for count in timeseries_counts:
        avg_time, time_per_pair = time_ccm_computation(X_emb[:count], Y_emb[:count], device=device)
        results.append(time_per_pair)
        print(f"N_sys={count}, Device={device}: Avg time = {avg_time:.6f}ms, Time per pair = {time_per_pair:.6f}ms")

    # Ensure that results are non-empty and contain valid time measurements
    assert len(results) == len(timeseries_counts)
    assert all(time > 0 for time in results)

### Main entry point for testing
if __name__ == "__main__":
    pytest.main()
