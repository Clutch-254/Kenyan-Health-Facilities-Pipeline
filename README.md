# Kenyan Health Facilities: A Data Science Pipeline

This project is a hands-on demonstration of a simple, end-to-end data science pipeline. I am using a real-world dataset of health facilities in Kenya to show how you can automatically fetch, clean, and prepare data for machine learning.

##  What's Inside?

This project is organized into a few key directories:

*   `data/`: This is where all the data lives. It's split into `raw/` for the original, untouched data, and `processed/` for the cleaned-up data and features.
*   `notebooks/`: Contains Jupyter notebooks for exploratory data analysis (EDA). This is where we get to know the data before we start building the pipeline.
*   `src/`: The heart of the project. This directory holds all the Python scripts that make up our pipeline.
*   `tests/`: Contains unit tests for the data processing and model training modules.

##  Getting Started

Ready to run the pipeline yourself? Here's how:

**1. Set up your environment:**

First, you'll need to create a virtual environment and install the required packages.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Run the pipeline:**

Once your environment is set up, you can run the entire pipeline with a single command:

```bash
python3 -m src.main
```

This will kick off the whole process, from downloading the data to training the model. A data profiling report will be generated at `reports/health_facilities_profile.html`.

##  Pipeline Steps

The pipeline is made up of a few key steps, each with its own script in the `src/` directory:

1.  **Data Ingestion (`data_ingestion.py`):** The pipeline starts by downloading the dataset of Kenyan health facilities from `data.humdata.org`.
2.  **Data Preprocessing (`data_processing.py`):** Next, we clean up the data. This involves removing unnecessary columns, filling in missing values, and converting some columns to a more useful format.
3.  **Feature Engineering (`feature_engineering.py`):** We then create features that can be used by a machine learning model. In this case, we're using one-hot encoding to convert categorical variables into a numerical format.
4.  **Model Training (`model_training.py`):** Finally, we train a machine learning model on our processed data.

## Data Quality and Validation

To ensure the quality and integrity of the data, the pipeline now includes:

*   **Schema Validation:** `pandera` is used to validate the data against a predefined schema. This checks for correct data types, non-null values, and ensures that the `Code` column (MFL Code) is unique for each facility.
*   **Automated Data Profiling:** `ydata-profiling` is used to generate a detailed data profile report. This report provides a comprehensive overview of the dataset, including descriptive statistics, distributions, and correlations. The report is saved as `reports/health_facilities_profile.html`.

##  The Model

We're training a `RandomForestClassifier` to predict whether a health facility is a "Level 2" facility. This is a simple classification task to demonstrate the final step of the pipeline. The trained model is saved to the `data/processed/` directory.

## Advanced Model Evaluation

To provide a more comprehensive understanding of the model's performance, the following evaluation features have been added:

*   **Confusion Matrix Analysis:** A confusion matrix is generated to visualize the model's performance in classifying different classes. The heatmap is saved as `reports/confusion_matrix.png`.
*   **Precision-Recall and F1-Score:** A detailed classification report is generated, including precision, recall, and F1-score for each class. This provides a more nuanced view of the model's performance, especially with imbalanced data.
*   **Feature Importance Visualization:** A bar plot of feature importances is created to understand which features are most influential in the model's predictions. The plot is saved as `reports/feature_importance.png`.
*   **Saved Evaluation Metrics:** All evaluation metrics, including the classification report and confusion matrix, are saved to `reports/evaluation_metrics.json` for tracking and comparison over time.

## Recent Changes

### Enhanced Data Ingestion
The data ingestion process has been made more robust:
- **Retries:** The `tenacity` library has been integrated to automatically retry downloading the dataset in case of network issues.
- **Error Handling:** `try/except` blocks have been added to catch and log any exceptions during the download process, ensuring the pipeline fails gracefully.

### Advanced Model Training
The model training process has been improved with hyperparameter tuning:
- **GridSearchCV:** `GridSearchCV` is now used to find the best hyperparameters for the `RandomForestClassifier` using 5-fold cross-validation.
- **Evaluation:** The best model is evaluated on a separate test set to provide a more accurate measure of its performance.

### Comprehensive Unit Testing
Unit tests have been added to ensure the reliability of the pipeline:
- **Test Coverage:** Tests have been created for the data processing and model training modules.
- **Mocking:** `pytest` and `pytest-mock` are used to mock external dependencies, allowing for isolated and reliable tests.

After running the pipeline, you'll see a classification report in your terminal that shows how well the model performed.


