import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from src import config
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

def train_model():
    """Trains a machine learning model and saves it."""
    # Load features
    features_path = config.PROCESSED_DATA_DIR / "kenya-health-facilities-features.csv"
    df = pd.read_csv(features_path)

    # Define features (X) and target (y)
    y = df['Keph level']
    X = df.drop(columns=['Keph level'])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Define the pipeline with SMOTE and RandomForestClassifier
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Define the parameter grid for GridSearchCV
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 20, None],
        'classifier__min_samples_split': [2, 5]
    }

    # Create StratifiedKFold for cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Create a GridSearchCV object
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=2, scoring='f1_weighted', error_score='raise')

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Evaluate the model
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Create reports directory if it doesn't exist
    os.makedirs(config.ROOT_DIR / "reports", exist_ok=True)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=best_model.classes_, yticklabels=best_model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    confusion_matrix_path = config.ROOT_DIR / "reports" / "confusion_matrix.png"
    plt.savefig(confusion_matrix_path)
    print(f"Confusion matrix saved to {confusion_matrix_path}")

    # Feature Importance
    feature_importances = best_model.named_steps['classifier'].feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df)
    plt.title('Feature Importance')
    feature_importance_path = config.ROOT_DIR / "reports" / "feature_importance.png"
    plt.savefig(feature_importance_path)
    print(f"Feature importance plot saved to {feature_importance_path}")

    # Save metrics
    metrics = {
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }
    metrics_path = config.ROOT_DIR / "reports" / "evaluation_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Evaluation metrics saved to {metrics_path}")


    # Save the model
    model_path = config.PROCESSED_DATA_DIR / "model.joblib"
    joblib.dump(best_model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model()
