from src.data_ingestion import download_data
from src.data_processing import main as preprocess_data
from src.feature_engineering import main as feature_engineering
from src.model_training import train_model

def main():
    """Main function to run the data science pipeline."""
    print("Starting the data science pipeline...")
    
    print("Step 1: Data Ingestion")
    download_data()
    print("Data ingestion complete.")
    
    print("Step 2: Data Preprocessing")
    preprocess_data()
    print("Data preprocessing complete.")
    
    print("Step 3: Feature Engineering")
    feature_engineering()
    print("Feature engineering complete.")
    
    print("Step 4: Model Training")
    train_model()
    print("Model training complete.")
    
    print("Pipeline finished.")

if __name__ == "__main__":
    main()
