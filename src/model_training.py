import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
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
    processed_data_path = config.PROCESSED_DATA_DIR / "kenya-health-facilities-processed.csv"
    df_processed = pd.read_csv(processed_data_path)
    
    df['target'] = (df_processed['Keph level'] == 'Level 2').astype(int)
    
    X = df.drop(columns=['target'])
    y = df['target']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the parameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }

    # Create a RandomForestClassifier
    rf = RandomForestClassifier(random_state=42)

    # Create a GridSearchCV object
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Evaluate the model
    y_pred = best_model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save the model
    model_path = config.PROCESSED_DATA_DIR / "model.joblib"
    joblib.dump(best_model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model()
