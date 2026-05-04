import os
import json
import pandas as pd
import mlflow
import tempfile

from backend_v2.pipeline.validate_data import validate_dataframe_schema, validate_csv_paths
from backend_v2.pipeline.train_model import train_model_from_directory
from backend_v2.utils import write_json_atomic


def test_validate_dataframe_schema_basic():
    df_events = pd.DataFrame({
        'event_id': ['e1'],
        'event_date': ["2024-01-01"],
        'year': [2024],
        'event_type': ['type1'],
        'sub_event_type': ['sub'],
        'actor1': ['A'],
        'actor2': ['B'],
        'region': ['R'],
        'country': ['C'],
        'admin1': ['Admin1'],
        'admin2': ['Admin2'],
        'location': ['Loc'],
        'latitude': [0.0],
        'longitude': [0.0],
        'fatalities': [1],
        'source': ['src'],
        'notes': ['note'],
    })
    results = validate_dataframe_schema({'events': df_events})
    assert 'events' in results
    assert results['events']['event_id'] is True
    assert results['events']['fatalities'] is True


def test_validate_csv_paths_basic(tmp_path, monkeypatch):
    # Create a simple CSV file with required columns
    df = pd.DataFrame({
        'event_id': ['e1'],
        'event_date': ['2024-01-01'],
        'year': [2024],
        'event_type': ['type'],
        'sub_event_type': ['sub'],
        'actor1': ['A'],
        'actor2': ['B'],
        'region': ['R'],
        'country': ['C'],
        'admin1': ['A1'],
        'admin2': ['A2'],
        'location': ['Loc'],
        'latitude': [0.0],
        'longitude': [0.0],
        'fatalities': [0],
        'source': ['src'],
        'notes': ['note'],
    })
    path = tmp_path / "events.csv"
    df.to_csv(path, index=False)

    res = validate_csv_paths([str(path)])
    assert res.get("valid", False) is True
    assert str(path) in res.get("validation", {})


def test_training_pipeline_with_synthetic_data(tmp_path):
    # Create a synthetic dataset with enough samples (>100)
    n = 150
    df = pd.DataFrame({
        'feature1': range(n),
        'feature2': [i * 2 for i in range(n)],
        'fatalities': [i % 5 for i in range(n)],  # target
    })
    data_path = tmp_path / 'train.csv'
    df.to_csv(data_path, index=False)

    # Train using our pipeline into a temporary models directory
    model_out_dir = tmp_path / 'models'
    result = train_model_from_directory([str(data_path)], model_out_dir=str(model_out_dir))

    assert 'model_path' in result
    model_path = result['model_path']
    assert os.path.exists(model_path)
    assert result['rmse'] >= 0


def test_write_json_atomic(tmp_path):
    data = {'hello': 'world'}
    path = tmp_path / 'config.json'
    write_json_atomic(str(path), data)
    with open(path, 'r') as f:
        loaded = json.load(f)
    assert loaded == data
