# FastCCM

Fast pairwise Convergent Cross Mapping in PyTorch.

FastCCM computes exact CCM scores equivalent to `pyEDM>=2.3.2`.

## Features

- Pairwise CCM and pairwise S-Map.
- Separate source and target sets.
- Vectorized `E`, `tau`, `tp` search and convergence testing.
- Blocked execution, auto batching, and memmap output for large matrices.

## Installation

**Requirements:** Python >= 3.9, pip.

**CPU-only**

```bash
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
pip install fastccm
```

**CUDA 12.6**

```bash
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126
pip install fastccm
```

**macOS (CPU / MPS)**

```bash
pip install torch==2.10.0
pip install fastccm
```

## Input format

FastCCM expects lists of 2D arrays.

- `X_emb`: source embeddings with shape `(T, E)`.
- `Y_emb`: target embeddings with shape `(T, E_y)`, or `(T, 1)` for scalar targets.
- Different lengths are end-aligned automatically.

## Examples

### 1. Generate data and find optimal `E` / `tau`

```python
import numpy as np
from fastccm import PairwiseCCM, Functions, Visualizer, utils
from fastccm.data import get_truncated_rossler_lorenz_rand

system = get_truncated_rossler_lorenz_rand(
    tmax=200,
    n_steps=4000,
    C=2,
    seed=0,
)

funcs = Functions(device="cpu", memory_budget_gb=2.0, verbose=0)
viz = Visualizer()

searches = [
    funcs.find_optimal_embedding_params(
        system[:, i],
        sample_size=400,
        exclusion_window=500,
        E_range=np.arange(1, 20),
        tau_range=np.arange(1, 20),
        tp_range=np.arange(1, 100, 10),
        seed=i,
        subtract_global=False,   # Subtract global linear model fit. Set to True to select E and tau 
                                 # based on the Simplex Projection without autoregression
    )
    for i in range(system.shape[1])
]

opt_E = [res["optimal_E"] for res in searches]
opt_tau = [res["optimal_tau"] for res in searches]

print(opt_E)
print(opt_tau)

viz.visualize_optimal_e_tau(searches[3])
```

### 2. Build a pairwise CCM matrix with the selected embeddings

```python
X_emb = utils.embed(system, E=opt_E, tau=opt_tau)
Y_emb = system.T[:, :, None]
ccm = PairwiseCCM(device="cpu", memory_budget_gb=2.0, verbose=0)

scores = ccm.score_matrix(
    X_emb=X_emb,
    Y_emb=Y_emb,
    library_size="auto",
    sample_size="auto",
    exclusion_window=20,
    method="simplex",
    seed=0,
)

ccm_matrix = scores[0]  # Y is scalar, so output shape is (1, n_Y, n_X)
print(ccm_matrix.shape)  # (6, 6)
print(ccm_matrix)
```

### 3. Run a convergence test

```python
x_idx = 1
y_idx = 3

X_pair = [utils.embed(system[:, x_idx], E=opt_E[x_idx], tau=opt_tau[x_idx])[0]]
Y_pair = [utils.embed(system[:, y_idx], E=opt_E[y_idx], tau=opt_tau[y_idx])[0]]

conv = funcs.convergence_test(
    X_emb=X_pair,
    Y_emb=Y_pair,
    library_sizes=[100, 200, 400, 800, 1600],
    sample_size="auto",
    exclusion_window=20,
    method="simplex",
    trials=10,
    seed=0,
)

print(conv["library_sizes"])
print(conv["X_to_Y"].shape)  # (n_sizes, trials, E_y, n_Y, n_X)

viz.plot_convergence_test(conv)
```

### 4. Run larger jobs in blocks

```python
import numpy as np

rng = np.random.default_rng(0)

X_emb = rng.uniform(0.0, 1.0, size=(50_000, 1000, 5)).astype(np.float32)
Y_emb = rng.uniform(0.0, 1.0, size=(50_000, 1000, 1)).astype(np.float32)

funcs = Functions(device="cpu", memory_budget_gb=2.0, verbose=1)

scores_mm = funcs.score_matrix_blocked(
    X_emb=X_emb,
    Y_emb=Y_emb,
    x_block=100,
    y_block=50_000,
    library_size="auto",
    sample_size="auto",
    exclusion_window=20,
    method="simplex",
    seed=0,
    out_path="ccm_scores.dat",
)

print(type(scores_mm))
print(scores_mm.shape)
```

## Performance

Measured on **CPU, Apple M4 Pro 64GB**

CCM matrix timings (`E=5`, `exclusion_window=5`)

| Condition | CCM matrix / simplex (s) | CCM matrix / S-MAP (s) |
|---|---:|---:|
| 100x100, T=1000 | 0.075 | 0.530 |
| 200x200, T=1000 | 0.156 | 1.188 |
| 800x800, T=500 | 0.721 | 5.501 |
| 100x100, T=8000 | 3.581 | 14.194 |

Single time series timings (`E=20`, `exclusion_window=10`)

| Condition | Simplex projection (s) | S-MAP projection (s) |
|---|---:|---:|
| T=2000 | 0.006 | 0.010 |
| T=8000 | 0.045 | 0.088 |
| T=32000 | 0.755 | 1.380 |
| T=128000 | 12.945 | 23.922 |

## Related files

- `notebooks/CCM results comparison.ipynb`
- `scripts/benchmark_performance.py`
- `scripts/benchmark_single_series_self_prediction.py`
