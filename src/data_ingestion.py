import os
import logging
import requests
from tenacity import retry, stop_after_attempt, wait_fixed
from src import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def download_data():
    """Downloads the dataset from the URL and saves it to the raw data directory."""
    try:
        logging.info(f"Downloading dataset from {config.DATASET_URL}")
        os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
        response = requests.get(config.DATASET_URL)
        response.raise_for_status()  # Raise an exception for bad status codes
        with open(config.DATASET_PATH, "wb") as f:
            f.write(response.content)
        logging.info(f"Dataset downloaded to {config.DATASET_PATH}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading dataset: {e}")
        raise

def main():
    """Downloads the dataset."""
    download_data()

if __name__ == "__main__":
    main()
