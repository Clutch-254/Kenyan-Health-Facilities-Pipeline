from pathlib import Path

# Project Root
ROOT_DIR = Path(__file__).parent.parent

# Data Ingestion
DATASET_URL = "https://data.humdata.org/dataset/7b9c2851-dc37-4a88-9dcb-62e55eb91baf/resource/df6bfc55-3b25-4309-a1b4-74afba434956/download/kenya-health-facilities-2017_08_02.xlsx"
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
DATASET_PATH = RAW_DATA_DIR / "kenya-health-facilities.xlsx"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"