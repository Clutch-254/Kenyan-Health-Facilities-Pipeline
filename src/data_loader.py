import pandas as pd
from src import config

def load_data():
    """Loads the dataset from the raw data directory into a pandas DataFrame."""
    df = pd.read_excel(config.DATASET_PATH)
    return df
