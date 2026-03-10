import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta
import uuid
import os
from pathlib import Path
import asyncio
from database import DatabaseManager

logger = logging.getLogger(__name__)

class MLPipeline:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.model_metadata = {}
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
    async def load_models(self):
        """Load existing trained models from disk"""
        try:
            for model_file in self.models_dir.glob("*.joblib"):
                model_id = model_file.stem
                self.models[model_id] = joblib.load(model_file)
                logger.info(f"Loaded model: {model_id}")
                
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
    
    async def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML training"""
        try:
            # Create a copy to avoid modifying original data
            features_df = df.copy()
            
            # Convert event_date to datetime
            features_df['event_date'] = pd.to_datetime(features_df['event_date'])
            
            # Extract temporal features
            features_df['year'] = features_df['event_date'].dt.year
            features_df['month'] = features_df['event_date'].dt.month
            features_df['day_of_year'] = features_df['event_date'].dt.dayofyear
            features_df['day_of_week'] = features_df['event_date'].dt.dayofweek
            
            # Geographic features
            features_df['lat_rounded'] = features_df['latitude'].round(1)
            features_df['lng_rounded'] = features_df['longitude'].round(1)
            
            # Encode categorical variables
            categorical_columns = ['event_type', 'country', 'actor1', 'actor2']
            for col in categorical_columns:
                if col in features_df.columns:
                    # Always fit encoder on current data to avoid unseen label issues
                    if col not in self.encoders:
                        self.encoders[col] = LabelEncoder()
                    
                    # Fill NaN values and fit encoder on current data
                    features_df[col] = features_df[col].fillna('unknown')
                    self.encoders[col].fit(features_df[col])
                    features_df[f'{col}_encoded'] = self.encoders[col].transform(features_df[col])
            
            # Historical features (rolling averages)
            features_df = features_df.sort_values('event_date')
            features_df['fatalities_7d_avg'] = features_df['fatalities'].rolling(
                window=7, min_periods=1
            ).mean()
            features_df['fatalities_30d_avg'] = features_df['fatalities'].rolling(
                window=30, min_periods=1
            ).mean()
            
            # Event frequency features (simplified)
            features_df['events_7d_count'] = features_df.groupby(['lat_rounded', 'lng_rounded']).cumcount() + 1
            
            return features_df
            
        except Exception as e:
            logger.error(f"Failed to prepare features: {e}")
            raise
    
    async def train_model(
        self,
        training_id: str,
        dataset_id: str,
        model_type: str = "random_forest",
        hyperparameters: Dict[str, Any] = None
    ):
        """Train ML model on dataset"""
        try:
            db_manager = DatabaseManager()
            await db_manager.initialize()
            
            # Fetch training data - if dataset_id is "current_database", get all events
            if dataset_id == "current_database":
                events = await db_manager.get_events({}, limit=50000)
            else:
                events = await db_manager.get_events(
                    filters={"upload_id": dataset_id},
                    limit=10000
                )
            
            if len(events) < 100:
                raise ValueError(f"Insufficient training data (minimum 100 events required, got {len(events)})")
            
            print(f"Training model with {len(events)} events")
            df = pd.DataFrame(events)
            
            # Prepare features
            features_df = await self.prepare_features(df)
            
            # Select feature columns (enhanced feature set)
            feature_columns = [
                'latitude', 'longitude', 'year', 'month', 'day_of_year', 'day_of_week',
                'lat_rounded', 'lng_rounded', 'fatalities_7d_avg', 'fatalities_30d_avg',
                'events_7d_count'
            ]
            
            # Add encoded categorical features
            for col in ['event_type', 'country', 'actor1', 'actor2']:
                if f'{col}_encoded' in features_df.columns:
                    feature_columns.append(f'{col}_encoded')
            
            # Prepare training data
            X = features_df[feature_columns].fillna(0)
            y = features_df['fatalities']
            
            # Remove extreme outliers (fatalities > 99th percentile)
            fatality_threshold = y.quantile(0.99)
            mask = y <= fatality_threshold
            X = X[mask]
            y = y[mask]
            
            print(f"After outlier removal: {len(X)} samples, max fatalities: {y.max()}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=pd.cut(y, bins=5, duplicates='drop')
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Set default hyperparameters if not provided
            if hyperparameters is None:
                hyperparameters = {}
            
            # Train model based on type with enhanced parameters
            if model_type == "random_forest":
                model = RandomForestRegressor(
                    n_estimators=hyperparameters.get('n_estimators', 100),
                    max_depth=hyperparameters.get('max_depth', 15),
                    min_samples_split=hyperparameters.get('min_samples_split', 5),
                    min_samples_leaf=hyperparameters.get('min_samples_leaf', 2),
                    max_features=hyperparameters.get('max_features', 'sqrt'),
                    random_state=42,
                    n_jobs=-1
                )
            elif model_type == "gradient_boosting":
                model = GradientBoostingRegressor(
                    n_estimators=hyperparameters.get('n_estimators', 100),
                    max_depth=hyperparameters.get('max_depth', 8),
                    learning_rate=hyperparameters.get('learning_rate', 0.1),
                    subsample=hyperparameters.get('subsample', 0.8),
                    random_state=42
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Train model
            print(f"Training {model_type} model...")
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            
            # Calculate comprehensive metrics
            metrics = {
                "train_r2": float(r2_score(y_train, y_pred_train)),
                "test_r2": float(r2_score(y_test, y_pred_test)),
                "train_mse": float(mean_squared_error(y_train, y_pred_train)),
                "test_mse": float(mean_squared_error(y_test, y_pred_test)),
                "train_mae": float(mean_absolute_error(y_train, y_pred_train)),
                "test_mae": float(mean_absolute_error(y_test, y_pred_test)),
                "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
                "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
                "r2": float(r2_score(y_test, y_pred_test)),  # Main metric
                "mse": float(mean_squared_error(y_test, y_pred_test)),
                "mae": float(mean_absolute_error(y_test, y_pred_test)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
            }
            
            # Feature importance (for tree-based models)
            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                importance_scores = model.feature_importances_
                feature_importance = dict(zip(feature_columns, importance_scores))
                # Sort by importance
                feature_importance = dict(sorted(feature_importance.items(), 
                                               key=lambda x: x[1], reverse=True))
            
            # Save model
            model_id = f"{model_type}_{training_id}"
            model_path = self.models_dir / f"{model_id}.joblib"
            scaler_path = self.models_dir / f"{model_id}_scaler.joblib"
            
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            
            # Store in memory
            self.models[model_id] = model
            self.scalers[model_id] = scaler
            
            # Store metadata in database
            model_metadata = {
                "model_id": model_id,
                "training_id": training_id,
                "dataset_id": dataset_id,
                "model_type": model_type,
                "hyperparameters": hyperparameters or {},
                "metrics": metrics,
                "feature_columns": feature_columns,
                "feature_importance": feature_importance,
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "total_samples": len(X),
                "outliers_removed": len(df) - len(X),
                "status": "completed",
                "model_path": str(model_path),
                "scaler_path": str(scaler_path),
                "created_at": datetime.utcnow().isoformat(),
                "data_quality": {
                    "completeness": float(features_df.notna().mean().mean()),
                    "feature_count": len(feature_columns),
                    "target_variance": float(y.var()),
                    "target_range": [float(y.min()), float(y.max())]
                }
            }
            
            await db_manager.store_model(model_metadata)
            self.model_metadata[model_id] = model_metadata
            
            print(f"Model {model_id} trained successfully:")
            print(f"  - R² score: {metrics['r2']:.3f}")
            print(f"  - RMSE: {metrics['rmse']:.2f}")
            print(f"  - Training samples: {len(X_train):,}")
            print(f"  - Test samples: {len(X_test):,}")
            
            await db_manager.close()
            return model_id
            
        except Exception as e:
            print(f"Failed to train model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    async def generate_predictions(
        self,
        country: Optional[str] = None,
        horizon_days: int = 14,
        model_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate predictions for future conflict events"""
        try:
            # Use latest model if not specified
            if not model_id:
                model_id = max(self.models.keys()) if self.models else None
                
            if not model_id or model_id not in self.models:
                return {"error": "No trained model available"}
            
            model = self.models[model_id]
            scaler = self.scalers.get(model_id)
            
            if not scaler:
                return {"error": "Model scaler not found"}
            
            # Generate prediction scenarios
            predictions = []
            base_date = datetime.now()
            
            # Create prediction grid (simplified for demo)
            if country == "Democratic Republic of Congo":
                # DRC hotspot coordinates
                locations = [
                    {"lat": -4.038333, "lng": 21.758664, "name": "Kinshasa"},
                    {"lat": -11.717, "lng": 27.479, "name": "Lubumbashi"},
                    {"lat": 1.674, "lng": 29.234, "name": "Goma"},
                ]
            else:
                # Default locations
                locations = [
                    {"lat": 0.0, "lng": 0.0, "name": "Default Location"}
                ]
            
            for days_ahead in range(1, horizon_days + 1):
                pred_date = base_date + timedelta(days=days_ahead)
                
                for location in locations:
                    # Create feature vector
                    features = np.array([[
                        location["lat"],  # latitude
                        location["lng"],  # longitude
                        pred_date.year,   # year
                        pred_date.month,  # month
                        pred_date.timetuple().tm_yday,  # day_of_year
                        pred_date.weekday(),  # day_of_week
                        round(location["lat"], 1),  # lat_rounded
                        round(location["lng"], 1),  # lng_rounded
                        5.0,  # fatalities_7d_avg (placeholder)
                        10.0,  # fatalities_30d_avg (placeholder)
                        2.0,  # events_7d_count (placeholder)
                        1,    # event_type_encoded (placeholder)
                        0,    # country_encoded (placeholder)
                        0,    # actor1_encoded (placeholder)
                        0,    # actor2_encoded (placeholder)
                    ]])
                    
                    # Scale features
                    features_scaled = scaler.transform(features)
                    
                    # Make prediction
                    pred_fatalities = model.predict(features_scaled)[0]
                    
                    predictions.append({
                        "date": pred_date.isoformat(),
                        "location": location["name"],
                        "latitude": location["lat"],
                        "longitude": location["lng"],
                        "predicted_fatalities": max(0, float(pred_fatalities)),
                        "confidence": 0.75,  # Placeholder confidence score
                        "risk_level": "high" if pred_fatalities > 10 else "medium" if pred_fatalities > 5 else "low"
                    })
            
            return {
                "model_id": model_id,
                "prediction_date": base_date.isoformat(),
                "horizon_days": horizon_days,
                "country": country,
                "predictions": predictions,
                "model_metrics": self.model_metadata.get(model_id, {}).get("metrics", {})
            }
            
        except Exception as e:
            logger.error(f"Failed to generate predictions: {e}")
            return {"error": str(e)}
    
    async def get_model_metrics(self, model_id: str) -> Dict[str, Any]:
        """Get performance metrics for a model"""
        try:
            if model_id not in self.model_metadata:
                raise ValueError(f"Model {model_id} not found")
            
            return self.model_metadata[model_id]
            
        except Exception as e:
            logger.error(f"Failed to get model metrics: {e}")
            raise
    
    async def detect_model_drift(self, model_id: str, recent_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect if model performance has degraded (drift detection)"""
        try:
            # Simplified drift detection
            # In production, this would compare feature distributions and prediction accuracy
            
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            # Calculate basic statistics on recent data
            recent_stats = {
                "mean_fatalities": float(recent_data['fatalities'].mean()),
                "std_fatalities": float(recent_data['fatalities'].std()),
                "event_count": len(recent_data),
                "unique_locations": recent_data['location'].nunique(),
            }
            
            # Compare with training data statistics (placeholder)
            drift_score = 0.1  # Placeholder drift score
            
            return {
                "model_id": model_id,
                "drift_score": drift_score,
                "drift_detected": drift_score > 0.3,
                "recent_stats": recent_stats,
                "recommendation": "retrain" if drift_score > 0.3 else "monitor"
            }
            
        except Exception as e:
            logger.error(f"Failed to detect model drift: {e}")
            raise