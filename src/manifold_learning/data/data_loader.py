# src/my_conda_project/data/dataset_loader.py

import os
import pandas as pd

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
    filename = "train_1.csv"
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File '{filename}' not found in datasets directory.")
    return pd.read_csv(filepath)
