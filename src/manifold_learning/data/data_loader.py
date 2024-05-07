# src/my_conda_project/data/dataset_loader.py

import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), 'datasets')

def load_csv_dataset(filename: str) -> pd.DataFrame:
    
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File '{filename}' not found in datasets directory.")
    return pd.read_csv(filepath)
