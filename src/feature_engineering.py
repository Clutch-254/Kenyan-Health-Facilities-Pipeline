import pandas as pd
from src import config

def get_features(df):
    """Performs feature engineering on the dataset."""
    # One-hot encode categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    return df_encoded

def main():
    """Loads the processed data, performs feature engineering, and saves the features."""
    processed_data_path = config.PROCESSED_DATA_DIR / "kenya-health-facilities-processed.csv"
    df = pd.read_csv(processed_data_path)
    
    df_features = get_features(df)
    
    # Save features
    features_path = config.PROCESSED_DATA_DIR / "kenya-health-facilities-features.csv"
    df_features.to_csv(features_path, index=False)
    
    print(f"Features saved to {features_path}")

if __name__ == "__main__":
    main()
