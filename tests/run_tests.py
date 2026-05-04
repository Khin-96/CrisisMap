import os
import json
import pandas as pd
import tempfile
import numpy as np

from backend_v2.pipeline.validate_data import validate_dataframe_schema, validate_csv_paths
from backend_v2.pipeline.train_model import train_model_from_directory
from backend_v2.utils import write_json_atomic


def run():
    # Create temp dir for training data and models
    with tempfile.TemporaryDirectory() as td:
        td_path = os.path.abspath(td)
        # 1) Validation test - create small events.csv
        df = pd.DataFrame({
            'event_id': ['e1','e2'],
            'event_date': ['2024-01-01','2024-01-02'],
            'year': [2024,2024],
            'event_type': ['type','type'],
            'sub_event_type': ['sub','sub'],
            'actor1': ['A','B'],
            'actor2': ['C','D'],
            'region': ['R','R'],
            'country': ['C','C'],
            'admin1': ['A1','A1'],
            'admin2': ['A2','A2'],
            'location': ['Loc','Loc'],
            'latitude': [0.0, 0.1],
            'longitude': [0.0, -0.1],
            'fatalities': [0,1],
            'source': ['src','src'],
            'notes': ['n','n'],
        })
        events_path = os.path.join(td_path, 'events.csv')
        df.to_csv(events_path, index=False)

        # 2) Training data: synthetic dataset with enough samples (>100)
        n = 120
        train_df = pd.DataFrame({
            'feature1': np.arange(n),
            'feature2': np.arange(n, 2*n),
            'fatalities': np.random.randint(0, 5, size=n)
        })
        train_path = os.path.join(td_path, 'train.csv')
        train_df.to_csv(train_path, index=False)

        # 3) Run validation on produced paths
        v = validate_csv_paths([events_path])
        if not v.get('valid', False):
            raise SystemExit('Validation failed for events.csv')

        # 4) Train model using the training path
        model_out_dir = os.path.join(td_path, 'models')
        result = train_model_from_directory([train_path], model_out_dir=model_out_dir)
        if 'model_path' not in result:
            raise SystemExit('Training did not produce a model_path')
        if not os.path.exists(result['model_path']):
            raise SystemExit('Trained model file not found')

        # 5) Json write test
        config_path = os.path.join(td_path, 'config.json')
        write_json_atomic(config_path, {'ok': True})
        with open(config_path, 'r') as f:
            loaded = json.load(f)
        if loaded != {'ok': True}:
            raise SystemExit('JSON write validation failed')

        print("ALL TESTS PASSED")

if __name__ == '__main__':
    run()
