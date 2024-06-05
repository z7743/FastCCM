import os
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp

DATA_DIR = os.path.join(os.path.dirname(__file__), 'datasets')

def load_csv_dataset(filename: str) -> pd.DataFrame:
    
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File '{filename}' not found in datasets directory.")
    return pd.read_csv(filepath)

def load_ld2011_2014_dataset() -> pd.DataFrame:
    filename = "LD2011_2014.txt"
    filepath = os.path.join(DATA_DIR, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File '{filename}' not found in datasets directory.")
    
    # Load the CSV file with pandas, specifying the delimiter and avoiding DtypeWarning
    df = pd.read_csv(filepath, sep=';', low_memory=False)
    
    # Replace commas with dots for numeric conversion
    df = df.apply(lambda x: x.str.replace(',', '.') if x.dtype == 'object' else x)
    
    # Convert all columns to numeric, setting invalid parsing to NaN
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            pass
    
    # Replace NaNs with zero if needed
    df.fillna(0, inplace=True)
    
    return df

def load_traffic_dataset() -> pd.DataFrame:
    filename = "train_2.csv"
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File '{filename}' not found in datasets directory.")
    return pd.read_csv(filepath)

def lorenz(t, state, sigma, beta, rho):
    """
    [ChatGPT written]
    """
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

def get_truncated_lorenz_rand(tmax = 140, n_steps = 10000, sigma=10, beta=8/3, rho=28):
    initial_state = np.random.normal(size=(3))

    trunc = int(n_steps/tmax * 40) # Number of steps to get independence from initial conditions
    t_eval = np.linspace(0, tmax, trunc + n_steps)

    solution = solve_ivp(lorenz, (0, tmax), initial_state, args=(sigma, beta, rho), t_eval=t_eval).y.T[trunc:]
    return solution