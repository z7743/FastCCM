# FastCCM
PyTorch-based implementation of Convergent Cross Mapping (CCM) optimized for calculating pairwise CCM matrices.

## Installation

To clone the repository and install the necessary dependencies, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/z7743/FastCCM.git
    cd FastCCM
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Install the package:
    ```bash
    pip install .
    ```

## Quickstart

This example demonstrates how to use the FastCCM package for performing Convergent Cross Mapping (CCM).

1. Import Required Libraries

```python
from fastccm import PairwiseCCM, utils
from fastccm.data import get_truncated_rossler_lorenz_rand
import numpy as np
```

2. Initialize the CCM Object

Specify the device to use (e.g., "cpu" or "cuda"):

```python
ccm = PairwiseCCM(device="cpu")
```

3. Generate Data

```python
# Generate a joint Rossler-Lorenz data
X = get_truncated_rossler_lorenz_rand(2000, 200000, alpha=6, C=2)

Rossler_emb = X[:, :3][None]  # Shape: (number of embeddings, number of points, number of dimensions)
Lorenz_emb  = X[:, 3:][None]

```
![alt text](docs/img/rossler_lorenz.png)

### Calculate cross-mapping prediction

```python
# Rossler cross-mapping Lorenz
result_rossler_xmap_lorenz = ccm.compute(
    X=Rossler_emb, Y=Lorenz_emb, 
    library_size=5000, 
    sample_size=500, 
    exclusion_window=50, 
    tp=0, 
    method="simplex"
)

# Lorenz cross-mapping Rossler
result_lorenz_xmap_rossler = ccm.compute(
    X=Lorenz_emb, Y=Rossler_emb, 
    library_size=5000, 
    sample_size=500, 
    exclusion_window=50, 
    tp=0, 
    method="simplex"
)

print("Rossler xmap Lorenz:", result_rossler_xmap_lorenz)
print("Lorenz xmap Rossler:", result_lorenz_xmap_rossler)

```

### Test convergence

```python
from fastccm import ccm_utils

conv_test_res = ccm_utils.Functions("cpu").convergence_test(
    X=Rossler_emb, Y=Lorenz_emb,
    library_sizes=[80, 160, 320, 640, 1250, 2500, 5000, 10000, 20000],
    sample_size=1000, 
    exclusion_window=20, 
    tp=0, 
    method="simplex", 
    trials=20
)
```

Plot the convergence test results:
```python
ccm_utils.Visualizer().plot_convergence_test(conv_test_res)
```

![alt text](docs/img/conv_test.png)

### Find optimal time-delay embedding parameters
```python
x = X[:,3]

optimal_E_tau_res = ccm_utils.Functions("cpu").find_optimal_embedding_params(
    x, x, 
    library_size=2000, 
    sample_size=500, 
    exclusion_window=10,
    E_range=np.arange(2,30),
    tau_range=np.arange(1,30),
    tp_max=50,
    method="simplex")
```

Plot the results
```python
ccm_utils.Visualizer().visualize_optimal_e_tau(optimal_E_tau_res)
```

![alt text](docs/img/e_tau_test.png)
