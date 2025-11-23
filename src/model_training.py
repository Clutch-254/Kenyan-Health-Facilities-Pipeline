import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from src import config

def train_model():
    """Trains a machine learning model and saves it."""
    # Load features
    features_path = config.PROCESSED_DATA_DIR / "kenya-health-facilities-features.csv"
    df = pd.read_csv(features_path)

    # Define features (X) and target (y)
    # We need to select the correct target column. 
    # Since we one-hot encoded the 'Keph level', we will have multiple 'Keph level_*' columns.
    # Let's choose 'Keph level_Unknown' as the target for this example.
    # In a real-world scenario, you would handle the multi-class target appropriately.
    
    # For simplicity, we will predict if 'Keph level' is 'Level 2'
    # First, let's load the processed data to get the original 'Keph level'
    processed_data_path = config.PROCESSED_DATA_DIR / "kenya-health-facilities-processed.csv"
    df_processed = pd.read_csv(processed_data_path)
    
    df['target'] = (df_processed['Keph level'] == 'Level 2').astype(int)
    
    X = df.drop(columns=['target'])
    y = df['target']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a RandomForestClassifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save the model
    model_path = config.PROCESSED_DATA_DIR / "model.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model()
