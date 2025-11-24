import pandas as pd
from src import config
from src.target_encoder import TargetEncoder

def get_features(df):
    """Performs feature engineering on the dataset."""
    
    # Define target column
    target_column = 'Keph level'
    
    # Convert target column to numerical (0 or 1)
    df[target_column] = (df[target_column] == 'Level 2').astype(int)
    
    # Separate features (X) and target (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Identify high cardinality columns
    high_cardinality_cols = ['Name', 'Constituency', 'Sub county', 'Ward']
    
    # Apply Target Encoding to high cardinality columns
    encoder = TargetEncoder(columns=high_cardinality_cols, target_column='target_temp')
    
    # Create a temporary DataFrame with the target for fitting the encoder
    X_with_target = X.copy()
    X_with_target['target_temp'] = y
    
    encoder.fit(X_with_target, y)
    
    # Transform only the high cardinality columns in X
    X_target_encoded = encoder.transform(X[high_cardinality_cols])
    
    # One-hot encode remaining categorical features
    # Get all categorical columns from X
    all_categorical_cols_in_X = X.select_dtypes(include=['object']).columns
    
    # Identify columns for one-hot encoding (those that are categorical but not high cardinality)
    onehot_cols = [col for col in all_categorical_cols_in_X if col not in high_cardinality_cols]
    
    X_onehot_encoded = pd.get_dummies(X[onehot_cols], columns=onehot_cols, drop_first=True)
    
    # Get numerical columns from X
    X_numerical = X.select_dtypes(exclude=['object'])
    
    # Combine all processed features and the target
    df_features = pd.concat([X_numerical, X_target_encoded, X_onehot_encoded, y], axis=1)
    
    return df_features

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
