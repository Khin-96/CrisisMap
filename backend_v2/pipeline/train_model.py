import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import mlflow
from datetime import datetime


def _discover_target(df: pd.DataFrame) -> str:
    possible = ['fatalities', 'target', 'target_fatalities', 'death_toll']
    for c in df.columns:
        if c in possible:
            return c
    # Fallback: if a column named 'fatalities' exists, use it
    if 'fatalities' in df.columns:
        return 'fatalities'
    return ''


def _select_features_and_target(df: pd.DataFrame, target_col: str):
    if target_col and target_col in df.columns:
        y = df[target_col]
        X = df.drop(columns=[target_col])
        # Simple numeric-only features; drop non-numeric columns for baseline
        X = X.select_dtypes(include=['number'])
        return X, y
    raise ValueError("Target column not found in training data")


def train_model_from_directory(data_paths: list, model_out_dir: str = 'backend_v2/models'):
    # Load all CSV/Excel files from data_paths and concatenate
    frames = []
    for p in data_paths:
        if p.lower().endswith('.csv'):
            frames.append(pd.read_csv(p))
        else:
            try:
                frames.append(pd.read_excel(p))
            except Exception:
                continue
    if not frames:
        raise ValueError('No valid data files found for training')
    df = pd.concat(frames, ignore_index=True)

    target_col = _discover_target(df)
    if not target_col:
        raise ValueError('Could not determine target column in training data')

    X, y = _select_features_and_target(df, target_col)
    if X.shape[0] < 100:
        raise ValueError('Insufficient training samples')

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Simple baseline model selection; can be extended via config/MLFlow
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_val)
    rmse = mean_squared_error(y_val, preds, squared=False)

    # MLflow logging for reproducibility
    mlflow.set_experiment("crisismap_training")
    with mlflow.start_run(run_name=f"train_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"):
        mlflow.log_param("model", type(rf).__name__)
        mlflow.log_param("target", target_col)
        mlflow.log_metric("rmse", rmse)
        # Save model artifact
        os.makedirs(model_out_dir, exist_ok=True)
        model_path = os.path.join(model_out_dir, f"{target_col}_rf.pkl")
        joblib.dump(rf, model_path)
        mlflow.log_artifact(model_path)

    return {
        'model_path': model_path,
        'rmse': rmse,
        'target': target_col,
        'n_features': X.shape[1],
        'n_samples': X.shape[0],
    }


def train_from_training_data_paths(paths: list):
    # Entry point for training using supplied paths
    return train_model_from_directory(paths)


__all__ = ["train_model_from_directory", "train_from_training_data_paths"]
