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

def rossler_lorenz(t, state, alpha, C):
    x1, x2, x3, y1, y2, y3 = state
    dx1dt = -alpha * (x2 + x3)
    dx2dt = alpha * (x1 + 0.2*x2)
    dx3dt = alpha * (0.2 + x3 * (x1 - 5.7))
    dy1dt = 10 * (-y1 + y2)
    dy2dt = 28 * y1 - y2 - y1*y3 + C*x2**2
    dy3dt = y1 * y2 - 8/3 * y3
    return [dx1dt, dx2dt, dx3dt, dy1dt, dy2dt, dy3dt]


def get_truncated_lorenz_rand(tmax = 140, n_steps = 10000, sigma=10, beta=8/3, rho=28):
    initial_state = np.random.normal(size=(3))

    trunc = int(n_steps/tmax * 40) # Number of steps to get independence from initial conditions
    t_eval = np.linspace(0, tmax, trunc + n_steps)

    solution = solve_ivp(lorenz, (0, tmax), initial_state, args=(sigma, beta, rho), t_eval=t_eval).y.T[trunc:]
    return solution

def get_truncated_rossler_lorenz_rand(tmax = 140, n_steps = 10000, alpha = 6, C = 8):
    initial_state = np.random.normal(size=(6))

    trunc = int(n_steps/tmax * 40) # Number of steps to get independence from initial conditions
    t_eval = np.linspace(0, tmax, trunc + n_steps)

    solution = solve_ivp(rossler_lorenz, (0, tmax), initial_state, args=(alpha, C), t_eval=t_eval).y.T[trunc:]
    return solution