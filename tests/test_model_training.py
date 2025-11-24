
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import pytest
from src.model_training import train_model
from src import config

def test_train_model(mocker):
    """Test the train_model function."""
    # Create a sample DataFrame to be returned by the mocked pd.read_csv
    features_data = {
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [0, 1, 0, 1, 0]
    }
    features_df = pd.DataFrame(features_data)

    processed_data = {
        'Keph level': ['Level 2', 'Level 3', 'Level 2', 'Level 4', 'Level 2']
    }
    processed_df = pd.DataFrame(processed_data)

    # Mock pd.read_csv
    def read_csv_mock(path):
        if path == config.PROCESSED_DATA_DIR / "kenya-health-facilities-features.csv":
            return features_df
        elif path == config.PROCESSED_DATA_DIR / "kenya-health-facilities-processed.csv":
            return processed_df
        return pd.DataFrame()

    mocker.patch('pandas.read_csv', side_effect=read_csv_mock)

    # Mock train_test_split
    mock_train_test_split = mocker.patch('src.model_training.train_test_split', return_value=(pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series([0, 1, 0, 1, 0])))

    # Mock GridSearchCV
    mock_model = mocker.MagicMock()
    mock_model.predict.return_value = pd.Series([0, 1, 0, 1, 0])
    mock_grid_search = mocker.MagicMock()
    mock_grid_search.best_estimator_ = mock_model
    mocker.patch('src.model_training.GridSearchCV', return_value=mock_grid_search)

    # Mock joblib.dump
    mock_joblib_dump = mocker.patch('src.model_training.joblib.dump')

    # Assert that GridSearchCV was called
    mock_grid_search_class = mocker.patch('src.model_training.GridSearchCV')
    mock_grid_search_instance = mock_grid_search_class.return_value
    mock_grid_search_instance.best_estimator_ = mock_model

    # Run the function
    train_model()

    # Assert that pd.read_csv was called with the correct paths
    assert pd.read_csv.call_count == 2
    pd.read_csv.assert_any_call(config.PROCESSED_DATA_DIR / "kenya-health-facilities-features.csv")
    pd.read_csv.assert_any_call(config.PROCESSED_DATA_DIR / "kenya-health-facilities-processed.csv")

    # Assert that train_test_split was called
    mock_train_test_split.assert_called_once()

    # Assert that GridSearchCV was called with the correct parameters
    mock_grid_search_class.assert_called_once()
    assert isinstance(mock_grid_search_class.call_args[1]['estimator'], RandomForestClassifier)
    assert mock_grid_search_class.call_args[1]['cv'] == 5
    assert mock_grid_search_class.call_args[1]['n_jobs'] == -1
    assert mock_grid_search_class.call_args[1]['verbose'] == 2

    # Assert that the model was trained
    mock_grid_search_instance.fit.assert_called_once()

    # Assert that the model was saved
    mock_joblib_dump.assert_called_once()
    # Check that the first argument to joblib.dump is the model
    assert mock_joblib_dump.call_args[0][0] == mock_model
    # Check that the second argument is the correct path
    assert mock_joblib_dump.call_args[0][1] == config.PROCESSED_DATA_DIR / "model.joblib"
