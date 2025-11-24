
import pandas as pd
import pytest
from src.data_processing import preprocess_data

@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    data = {
        'Registration_number': ['123', '456'],
        'Service_names': ['Service A', 'Service B'],
        'Keph level': [None, 'Level 2'],
        'Regulatory body': ['MOH', None],
        'Open_whole_day': ['Yes', 'No'],
        'Open_public_holidays': ['No', 'Yes'],
        'Open_weekends': ['Yes', 'No'],
        'Open_late_night': ['No', 'Yes'],
        'Approved': ['Yes', 'No'],
        'Public visible': ['No', 'Yes'],
        'Closed': ['Yes', 'No']
    }
    return pd.DataFrame(data)

def test_preprocess_data(sample_dataframe):
    """Test the preprocess_data function."""
    processed_df = preprocess_data(sample_dataframe)

    # Test that unnecessary columns are dropped
    assert 'Registration_number' not in processed_df.columns
    assert 'Service_names' not in processed_df.columns

    # Test that missing values are filled
    assert processed_df['Keph level'].isnull().sum() == 0
    assert processed_df['Regulatory body'].isnull().sum() == 0
    assert 'Unknown' in processed_df['Keph level'].values
    assert 'Unknown' in processed_df['Regulatory body'].values

    # Test that 'Yes'/'No' columns are converted to boolean
    for col in ['Open_whole_day', 'Open_public_holidays', 'Open_weekends', 'Open_late_night', 'Approved', 'Public visible', 'Closed']:
        assert processed_df[col].dtype == 'bool'

    # Test the shape of the dataframe
    assert processed_df.shape == (2, 9)
