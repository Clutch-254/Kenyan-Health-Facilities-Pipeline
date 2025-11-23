import pandas as pd
from src import config
from src.data_loader import load_data

def preprocess_data(df):
    """Performs preprocessing on the dataset."""
    # Drop unnecessary columns
    df = df.drop(columns=['Registration_number', 'Service_names'])

    # Fill missing values
    df['Keph level'] = df['Keph level'].fillna('Unknown')
    df['Regulatory body'] = df['Regulatory body'].fillna('Unknown')

    # Convert 'Yes'/'No' columns to boolean
    for col in ['Open_whole_day', 'Open_public_holidays', 'Open_weekends', 'Open_late_night', 'Approved', 'Public visible', 'Closed']:
        df[col] = df[col].apply(lambda x: True if x == 'Yes' else False)

    return df

def main():
    """Loads the data, preprocesses it, and saves it to the processed data directory."""
    df = load_data()
    df_processed = preprocess_data(df)
    
    # Create processed data directory if it doesn't exist
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save processed data
    processed_data_path = config.PROCESSED_DATA_DIR / "kenya-health-facilities-processed.csv"
    df_processed.to_csv(processed_data_path, index=False)
    
    print(f"Processed data saved to {processed_data_path}")

if __name__ == "__main__":
    main()
