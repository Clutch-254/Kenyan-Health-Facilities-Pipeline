import pandas as pd

def explore_data():
    # Load the processed data
    try:
        df = pd.read_csv('data/processed/kenya-health-facilities-processed.csv')
    except FileNotFoundError:
        print('Processed data not found. Please run the main pipeline first.')
        # As a fallback for exploration, let's try to load the raw data and do minimal processing
        try:
            df = pd.read_excel('data/raw/kenya-health-facilities.xlsx')
        except FileNotFoundError:
            print('Raw data not found. Please run the data ingestion part of the pipeline.')
            df = None

    if df is not None:
        # Identify categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns

        # Calculate cardinality
        cardinality = df[categorical_cols].nunique()

        # Define a threshold for high cardinality
        high_cardinality_threshold = 50

        # Identify high cardinality columns
        high_cardinality_cols = cardinality[cardinality > high_cardinality_threshold].index.tolist()

        print('Categorical columns:', categorical_cols.tolist())
        print('\nHigh cardinality columns:', high_cardinality_cols)
        print('\nCardinality of high cardinality columns:\n', cardinality[high_cardinality_cols])
    
    return df, high_cardinality_cols

if __name__ == '__main__':
    df, high_cardinality_cols = explore_data()

