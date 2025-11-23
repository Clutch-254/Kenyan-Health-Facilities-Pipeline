import os
import requests
from src import config

def download_data():
    """Downloads the dataset from the URL and saves it to the raw data directory."""
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    response = requests.get(config.DATASET_URL)
    with open(config.DATASET_PATH, "wb") as f:
        f.write(response.content)
    print(f"Dataset downloaded to {config.DATASET_PATH}")

if __name__ == "__main__":
    download_data()
