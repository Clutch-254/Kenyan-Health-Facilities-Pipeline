import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check
from ydata_profiling import ProfileReport
from src import config

# Define the Pandera schema for health facilities data
health_facilities_schema = pa.DataFrameSchema(
    columns={
        "Code": Column(int, unique=True, nullable=False),
        "Name": Column(str, nullable=False),
        "Registration_number": Column(str, nullable=True),
        "Keph level": Column(str, nullable=True),
        "Facility type": Column(str, nullable=False),
        "Owner": Column(str, nullable=False),
        "Regulatory body": Column(str, nullable=True),
        "Beds": Column(int, nullable=False),
        "Cots": Column(int, nullable=False),
        "County": Column(str, nullable=False),
        "Constituency": Column(str, nullable=False),
        "Sub county": Column(str, nullable=False),
        "Ward": Column(str, nullable=False),
        "Operation status": Column(str, nullable=False),
        "Open_whole_day": Column(str, nullable=False),
        "Open_public_holidays": Column(str, nullable=False),
        "Open_weekends": Column(str, nullable=False),
        "Open_late_night": Column(str, nullable=False),
        "Service_names": Column(float, nullable=True), # This column seems to be all nulls, but we'll keep its type as float
        "Approved": Column(str, nullable=False),
        "Public visible": Column(str, nullable=False),
        "Closed": Column(str, nullable=False),
    },
    strict=True # Ensure no extra columns are present
)

def load_data():
    """
    Loads the dataset from the raw data directory into a pandas DataFrame,
    validates it against a schema, and generates a data profiling report.
    """
    df = pd.read_excel(config.DATASET_PATH)

    # Validate the DataFrame against the schema
    try:
        df = health_facilities_schema.validate(df)
        print("Data validation successful!")
    except pa.errors.SchemaErrors as err:
        print("Data validation failed!")
        print(err.failure_cases)
        raise err

    # Generate data profiling report
    print("Generating data profiling report...")
    profile = ProfileReport(df, title="Health Facilities Data Profile")
    profile.to_file(config.ROOT_DIR / "reports" / "health_facilities_profile.html")
    print(f"Data profiling report generated at {config.ROOT_DIR / 'reports' / 'health_facilities_profile.html'}")

    return df
