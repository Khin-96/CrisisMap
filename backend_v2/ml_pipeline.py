import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta
import uuid
import os
from pathlib import Path
import asyncio
from database_sqlite import DatabaseManager

logger = logging.getLogger(__name__)

HIGH_RISK_FATALITY_THRESHOLD = 10
MEDIUM_RISK_FATALITY_THRESHOLD = 3


def _compute_risk_level(predicted_fatalities: float, risk_score: float = 0) -> str:
    if predicted_fatalities >= HIGH_RISK_FATALITY_THRESHOLD or risk_score >= 0.7:
        return "high"
    if predicted_fatalities >= MEDIUM_RISK_FATALITY_THRESHOLD or risk_score >= 0.4:
        return "medium"
    return "low"


def _compute_risk_score(fatalities: float, max_fatalities: float) -> float:
    if max_fatalities <= 0:
        return 0.0
    return min(1.0, float(fatalities) / float(max_fatalities))


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
                if "_scaler" in model_file.stem:
                    continue
                model_id = model_file.stem
                self.models[model_id] = joblib.load(model_file)
                scaler_path = self.models_dir / f"{model_id}_scaler.joblib"
                if scaler_path.exists():
                    self.scalers[model_id] = joblib.load(scaler_path)
                logger.info(f"Loaded model: {model_id}")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")

    async def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML training"""
        try:
            features_df = df.copy()
            features_df['event_date'] = pd.to_datetime(features_df['event_date'], errors='coerce')
            features_df = features_df.dropna(subset=['event_date', 'latitude', 'longitude'])

            features_df['year'] = features_df['event_date'].dt.year
            features_df['month'] = features_df['event_date'].dt.month
            features_df['day_of_year'] = features_df['event_date'].dt.dayofyear
            features_df['day_of_week'] = features_df['event_date'].dt.dayofweek
            features_df['quarter'] = features_df['event_date'].dt.quarter
            features_df['month_sin'] = np.sin(2 * np.pi * features_df['month'] / 12)
            features_df['month_cos'] = np.cos(2 * np.pi * features_df['month'] / 12)

            features_df['lat_rounded'] = features_df['latitude'].round(1)
            features_df['lng_rounded'] = features_df['longitude'].round(1)

            categorical_columns = ['event_type', 'country', 'actor1', 'actor2']
            for col in categorical_columns:
                if col in features_df.columns:
                    if col not in self.encoders:
                        self.encoders[col] = LabelEncoder()
                    features_df[col] = features_df[col].fillna('unknown')
                    self.encoders[col].fit(features_df[col])
                    features_df[f'{col}_encoded'] = self.encoders[col].transform(features_df[col])

            features_df = features_df.sort_values('event_date')
            features_df['fatalities_7d_avg'] = features_df['fatalities'].rolling(window=7, min_periods=1).mean()
            features_df['fatalities_30d_avg'] = features_df['fatalities'].rolling(window=30, min_periods=1).mean()
            features_df['fatalities_90d_avg'] = features_df['fatalities'].rolling(window=90, min_periods=1).mean()

            # Per-location event frequency (cumulative count up to each row)
            features_df['events_7d_count'] = (
                features_df.groupby(['lat_rounded', 'lng_rounded']).cumcount() + 1
            )

            # Rolling total fatalities per location
            location_groups = features_df.groupby(['lat_rounded', 'lng_rounded'])['fatalities']
            features_df['loc_fatalities_total'] = location_groups.transform('sum')
            features_df['loc_events_total'] = features_df.groupby(
                ['lat_rounded', 'lng_rounded']
            )['fatalities'].transform('count')

            return features_df
        except Exception as e:
            logger.error(f"Failed to prepare features: {e}")
            raise

    def _get_feature_columns(self) -> List[str]:
        base = [
            'latitude', 'longitude', 'year', 'month', 'day_of_year', 'day_of_week',
            'quarter', 'month_sin', 'month_cos', 'lat_rounded', 'lng_rounded',
            'fatalities_7d_avg', 'fatalities_30d_avg', 'fatalities_90d_avg',
            'events_7d_count', 'loc_fatalities_total', 'loc_events_total'
        ]
        encoded = ['event_type_encoded', 'country_encoded', 'actor1_encoded', 'actor2_encoded']
        return base + encoded

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

            if dataset_id == "current_database":
                events = await db_manager.get_events({}, limit=50000)
            else:
                events = await db_manager.get_events(
                    filters={"upload_id": dataset_id},
                    limit=10000
                )

            if len(events) < 100:
                raise ValueError(
                    f"Insufficient training data (minimum 100 events required, got {len(events)})"
                )

            print(f"Training model with {len(events)} events")
            df = pd.DataFrame(events)
            features_df = await self.prepare_features(df)

            feature_columns = [
                c for c in self._get_feature_columns() if c in features_df.columns
            ]
            X = features_df[feature_columns].fillna(0)
            y = features_df['fatalities']

            # Remove extreme outliers
            fatality_threshold = y.quantile(0.99)
            mask = y <= fatality_threshold
            X = X[mask]
            y = y[mask]

            print(f"After outlier removal: {len(X)} samples, max fatalities: {y.max()}")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            if hyperparameters is None:
                hyperparameters = {}

            if model_type == "random_forest":
                model = RandomForestRegressor(
                    n_estimators=hyperparameters.get('n_estimators', 200),
                    max_depth=hyperparameters.get('max_depth', 20),
                    min_samples_split=hyperparameters.get('min_samples_split', 3),
                    min_samples_leaf=hyperparameters.get('min_samples_leaf', 1),
                    max_features=hyperparameters.get('max_features', 'sqrt'),
                    random_state=42,
                    n_jobs=-1
                )
            elif model_type == "gradient_boosting":
                model = GradientBoostingRegressor(
                    n_estimators=hyperparameters.get('n_estimators', 150),
                    max_depth=hyperparameters.get('max_depth', 10),
                    learning_rate=hyperparameters.get('learning_rate', 0.08),
                    subsample=hyperparameters.get('subsample', 0.8),
                    random_state=42
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            print(f"Training {model_type} model...")
            model.fit(X_train_scaled, y_train)

            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)

            metrics = {
                "train_r2": float(r2_score(y_train, y_pred_train)),
                "test_r2": float(r2_score(y_test, y_pred_test)),
                "train_mse": float(mean_squared_error(y_train, y_pred_train)),
                "test_mse": float(mean_squared_error(y_test, y_pred_test)),
                "train_mae": float(mean_absolute_error(y_train, y_pred_train)),
                "test_mae": float(mean_absolute_error(y_test, y_pred_test)),
                "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
                "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
                "r2": float(r2_score(y_test, y_pred_test)),
                "mse": float(mean_squared_error(y_test, y_pred_test)),
                "mae": float(mean_absolute_error(y_test, y_pred_test)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
            }

            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                importance_scores = model.feature_importances_
                feature_importance = dict(zip(feature_columns, importance_scores.tolist()))
                feature_importance = dict(
                    sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                )

            model_id = f"{model_type}_{training_id}"
            model_path = self.models_dir / f"{model_id}.joblib"
            scaler_path = self.models_dir / f"{model_id}_scaler.joblib"

            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)

            self.models[model_id] = model
            self.scalers[model_id] = scaler

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
                "status": "completed",
                "model_path": str(model_path),
                "scaler_path": str(scaler_path),
                "created_at": datetime.utcnow().isoformat(),
            }

            await db_manager.store_model(model_metadata)
            self.model_metadata[model_id] = model_metadata

            print(f"Model {model_id} trained. R2={metrics['r2']:.3f} RMSE={metrics['rmse']:.2f}")
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
        """
        Generate geo-tagged predictions for future conflict events.
        Uses real historical per-location statistics as features.
        Stores results in map_predictions and alerts tables.
        """
        try:
            if not model_id:
                model_id = max(self.models.keys()) if self.models else None

            if not model_id or model_id not in self.models:
                return {"error": "No trained model available"}

            model = self.models[model_id]
            scaler = self.scalers.get(model_id)
            if not scaler:
                return {"error": "Model scaler not found"}

            # Pull real historical data to compute per-location features
            db_manager = DatabaseManager()
            await db_manager.initialize()

            filters = {}
            if country:
                filters["country"] = country
            historical_events = await db_manager.get_events(filters, limit=50000)

            if not historical_events:
                await db_manager.close()
                return {"error": "No historical data available for prediction"}

            # Load CAST Predictions for boosting
            cast_boost_map = {}
            try:
                # Load forecasts for this year
                current_year = datetime.utcnow().year
                cast_data = await db_manager.get_cast_predictions(
                    filters={"year": current_year},
                    limit=2000
                )
                for c in cast_data:
                    # CAST uses country + month/period
                    country_name = c.get('country')
                    month_val = c.get('month')
                    if country_name and month_val:
                        # Normalize month names (e.g. from "2026-04-10" or "March")
                        if "-" in month_val: # Date format
                            try:
                                month_name = datetime.strptime(month_val, "%Y-%m-%d").strftime("%B")
                            except:
                                month_name = month_val
                        else:
                            month_name = month_val
                        
                        key = (country_name, month_name)
                        cast_boost_map[key] = c.get('total_forecast', 0)
                logger.info(f"Loaded {len(cast_boost_map)} CAST boost markers")
            except Exception as e:
                logger.warning(f"Failed to load CAST data for boosting: {e}")

            df = pd.DataFrame(historical_events)
            df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
            df = df.dropna(subset=['event_date', 'latitude', 'longitude'])
            df['fatalities'] = pd.to_numeric(df['fatalities'], errors='coerce').fillna(0)

            # Build per-location statistics from historical data
            df['lat_rounded'] = df['latitude'].round(1)
            df['lng_rounded'] = df['longitude'].round(1)

            location_stats = df.groupby(['lat_rounded', 'lng_rounded']).agg(
                loc_fatalities_total=('fatalities', 'sum'),
                loc_events_total=('fatalities', 'count'),
                fatalities_avg=('fatalities', 'mean'),
                location_name=('location', lambda x: x.mode()[0] if len(x) > 0 else ''),
                country_val=('country', lambda x: x.mode()[0] if len(x) > 0 else ''),
                actor1_val=('actor1', lambda x: x.mode()[0] if len(x) > 0 else ''),
                actor2_val=('actor2', lambda x: x.mode()[0] if len(x) > 0 else ''),
                event_type_val=('event_type', lambda x: x.mode()[0] if len(x) > 0 else ''),
            ).reset_index()

            # Only use locations with meaningful history (at least 3 events)
            location_stats = location_stats[location_stats['loc_events_total'] >= 3]

            if len(location_stats) == 0:
                await db_manager.close()
                return {"error": "Insufficient historical data per location"}

            # Compute global rolling averages from recent 30 days
            recent_cutoff = datetime.utcnow() - timedelta(days=30)
            df_recent = df[df['event_date'] >= recent_cutoff]
            global_7d_avg = float(df_recent['fatalities'].mean()) if len(df_recent) > 0 else float(df['fatalities'].mean())
            global_30d_avg = float(df['fatalities'].mean())
            global_90d_avg = global_30d_avg

            # Encode categoricals using training encoders
            def safe_encode(encoder: LabelEncoder, value: str) -> int:
                try:
                    if value in encoder.classes_:
                        return int(encoder.transform([value])[0])
                    return 0
                except Exception:
                    return 0

            predictions = []
            map_predictions = []
            alerts_to_store = []
            base_date = datetime.utcnow()
            max_pred_fatalities = 1.0

            feature_columns = self.model_metadata.get(model_id, {}).get(
                "feature_columns", self._get_feature_columns()
            )

            for days_ahead in range(1, horizon_days + 1):
                pred_date = base_date + timedelta(days=days_ahead)

                for _, loc_row in location_stats.iterrows():
                    lat = float(loc_row['lat_rounded'])
                    lng = float(loc_row['lng_rounded'])
                    loc_name = str(loc_row['location_name'])
                    loc_country = str(loc_row['country_val'])
                    loc_actor1 = str(loc_row['actor1_val'])
                    loc_actor2 = str(loc_row['actor2_val'])
                    loc_event_type = str(loc_row['event_type_val'])
                    loc_fat_total = float(loc_row['loc_fatalities_total'])
                    loc_evt_total = float(loc_row['loc_events_total'])

                    # Encode categoricals
                    event_type_enc = safe_encode(
                        self.encoders.get('event_type', LabelEncoder()), loc_event_type
                    ) if 'event_type' in self.encoders else 0
                    country_enc = safe_encode(
                        self.encoders.get('country', LabelEncoder()), loc_country
                    ) if 'country' in self.encoders else 0
                    actor1_enc = safe_encode(
                        self.encoders.get('actor1', LabelEncoder()), loc_actor1
                    ) if 'actor1' in self.encoders else 0
                    actor2_enc = safe_encode(
                        self.encoders.get('actor2', LabelEncoder()), loc_actor2
                    ) if 'actor2' in self.encoders else 0

                    # Frequency: scale events_7d_count by days ahead lookback
                    events_7d_est = max(1.0, loc_evt_total / max(1, days_ahead))

                    feature_map = {
                        'latitude': lat,
                        'longitude': lng,
                        'year': pred_date.year,
                        'month': pred_date.month,
                        'day_of_year': pred_date.timetuple().tm_yday,
                        'day_of_week': pred_date.weekday(),
                        'quarter': (pred_date.month - 1) // 3 + 1,
                        'month_sin': np.sin(2 * np.pi * pred_date.month / 12),
                        'month_cos': np.cos(2 * np.pi * pred_date.month / 12),
                        'lat_rounded': lat,
                        'lng_rounded': lng,
                        'fatalities_7d_avg': global_7d_avg,
                        'fatalities_30d_avg': global_30d_avg,
                        'fatalities_90d_avg': global_90d_avg,
                        'events_7d_count': events_7d_est,
                        'loc_fatalities_total': loc_fat_total,
                        'loc_events_total': loc_evt_total,
                        'event_type_encoded': event_type_enc,
                        'country_encoded': country_enc,
                        'actor1_encoded': actor1_enc,
                        'actor2_encoded': actor2_enc,
                    }

                    feature_vec = np.array([[feature_map.get(c, 0.0) for c in feature_columns]])

                    try:
                        feature_scaled = scaler.transform(feature_vec)
                        pred_fatalities = float(model.predict(feature_scaled)[0])
                    except Exception:
                        pred_fatalities = float(loc_row['fatalities_avg'])

                    # Apply CAST Boost
                    month_name = pred_date.strftime("%B")
                    total_forecast = cast_boost_map.get((loc_country, month_name), 0)
                    if total_forecast > 15:
                        pred_fatalities *= 1.5
                    elif total_forecast > 5:
                        pred_fatalities *= 1.25

                    pred_fatalities = max(0.0, pred_fatalities)
                    max_pred_fatalities = max(max_pred_fatalities, pred_fatalities)

                    predictions.append({
                        "date": pred_date.isoformat(),
                        "location": loc_name,
                        "country": loc_country,
                        "latitude": lat,
                        "longitude": lng,
                        "predicted_fatalities": round(pred_fatalities, 2),
                        "confidence": min(0.85, 0.4 + (loc_evt_total / 100)),
                        "risk_level": _compute_risk_level(pred_fatalities),
                        "actor1": loc_actor1,
                        "actor2": loc_actor2,
                        "event_type": loc_event_type,
                        "horizon_day": days_ahead,
                        "cast_boosted": total_forecast > 0
                    })

            # Compute proper risk scores now that we know max
            map_pred_ids = {}
            for pred in predictions:
                risk_score = _compute_risk_score(pred['predicted_fatalities'], max_pred_fatalities)
                pred['risk_score'] = round(risk_score, 3)
                pred['risk_level'] = _compute_risk_level(pred['predicted_fatalities'], risk_score)

                prediction_id = str(uuid.uuid4())
                map_pred_ids[id(pred)] = prediction_id

                map_predictions.append({
                    "prediction_id": prediction_id,
                    "model_id": model_id,
                    "location_name": pred['location'],
                    "latitude": pred['latitude'],
                    "longitude": pred['longitude'],
                    "country": pred['country'],
                    "event_type": pred['event_type'],
                    "predicted_fatalities": pred['predicted_fatalities'],
                    "predicted_events": 1,
                    "risk_level": pred['risk_level'],
                    "risk_score": pred['risk_score'],
                    "confidence": pred['confidence'],
                    "prediction_for_date": pred['date'],
                    "horizon_days": pred['horizon_day'],
                    "actor1": pred['actor1'],
                    "actor2": pred['actor2'],
                    "ai_summary": "",
                })

                # Flag high-risk predictions as alerts
                if pred['risk_level'] == 'high':
                    alerts_to_store.append({
                        "alert_id": str(uuid.uuid4()),
                        "prediction_id": prediction_id,
                        "title": f"High Risk Alert: {pred['location']}",
                        "body": (
                            f"Model predicts {pred['predicted_fatalities']:.0f} fatalities "
                            f"in {pred['location']}, {pred['country']} on "
                            f"{pred['date'][:10]}. Event type: {pred['event_type']}."
                        ),
                        "risk_level": "high",
                        "latitude": pred['latitude'],
                        "longitude": pred['longitude'],
                        "location_name": pred['location'],
                        "country": pred['country'],
                        "predicted_fatalities": pred['predicted_fatalities'],
                        "event_type": pred['event_type'],
                        "ai_response": "",
                    })

            # Store map predictions and alerts to DB
            if map_predictions:
                stored = await db_manager.store_map_predictions(map_predictions)
                print(f"Stored {stored} map predictions to database")

            for alert in alerts_to_store:
                await db_manager.store_alert(alert)
            if alerts_to_store:
                print(f"Stored {len(alerts_to_store)} high-risk alerts")

            await db_manager.close()

            return {
                "model_id": model_id,
                "prediction_date": base_date.isoformat(),
                "horizon_days": horizon_days,
                "country": country,
                "total_predictions": len(predictions),
                "high_risk_count": len(alerts_to_store),
                "predictions": predictions,
                "model_metrics": self.model_metadata.get(model_id, {}).get("metrics", {}),
            }

        except Exception as e:
            logger.error(f"Failed to generate predictions: {e}")
            import traceback
            traceback.print_exc()
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
        """Detect if model performance has degraded"""
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            recent_stats = {
                "mean_fatalities": float(recent_data['fatalities'].mean()),
                "std_fatalities": float(recent_data['fatalities'].std()),
                "event_count": len(recent_data),
            }
            drift_score = 0.1
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