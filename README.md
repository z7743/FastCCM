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

3. Install the package in editable mode:
    ```bash
    pip install -e .
    ```

## Usage Examples

This example demonstrates how to use the FastCCM package for performing Convergent Cross Mapping (CCM).

1. Import Required Libraries

```python
from FastCCM import CCM, CCM_utils
from FastCCM.data.data_loader import get_truncated_rossler_lorenz_rand
import numpy as np
import matplotlib.pyplot as plt
```

2. Initialize the CCM Object

Specify the device to use (e.g., "cpu" or "cuda"):

```python
ccm = CCM.PairwiseCCM(device="cpu")
```

3. Generate Data

```python
# Generate a joint Rossler-Lorenz data
X = get_truncated_rossler_lorenz_rand(2000, 200000, alpha=6, C=2)

Rossler_emb = X[:, :3][None]  # Shape: (number of embeddings, number of points, number of dimensions)
Lorenz_emb = X[:, 3:][None]

```
![alt text](docs/img/rossler_lorenz.png)

### Calculate cross-mapping prediction

```python
# Rossler cross-mapping Lorenz
result_rossler_xmap_lorenz = ccm.compute(
    X=Rossler_emb, Y=Lorenz_emb, 
    subset_size=5000, 
    subsample_size=500, 
    exclusion_rad=50, 
    tp=0, method="simplex", nbrs_num=10
)

# Lorenz cross-mapping Rossler
result_lorenz_xmap_rossler = ccm.compute(
    X=Lorenz_emb, Y=Rossler_emb, 
    subset_size=5000, 
    subsample_size=500, 
    exclusion_rad=50, 
    tp=0, method="simplex", nbrs_num=10
)

print("Rossler xmap Lorenz:", result_rossler_xmap_lorenz)
print("Lorenz xmap Rossler:", result_lorenz_xmap_rossler)

```

### Test convergence

```python
conv_test_res = CCM_utils.Functions("cpu").convergence_test(
    X=Rossler_emb, Y=Lorenz_emb,
    subset_sizes=[80, 160, 320, 640, 1250, 2500, 5000, 10000, 20000],
    subsample_size=1000, exclusion_rad=20, tp=0, method="simplex", trials=20, nbrs_num=10
)
```

Plot the convergence test results:
```python
CCM_utils.Visualizer().plot_convergence_test(conv_test_res)
```


![alt text](docs/img/conv_test.png)

### Find optimal time-delay embedding parameters
```python
x = X[:,3]

optimal_E_tau_res = CCM_utils.Functions("cpu").find_optimal_embedding_params(x, x, 2000, 500, 10,
                                                         E_range=np.arange(2,30),
                                                         tau_range=np.arange(1,30),
                                                         tp_max=50,
                                                         method="simplex",nbrs_num = 5)
```

Plot the results
```python
CCM_utils.Visualizer().visualize_optimal_e_tau(optimal_E_tau_res)
```


![alt text](docs/img/e_tau_test.png)
