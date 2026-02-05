import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class ConflictPredictor:
    """Machine learning models for conflict prediction and forecasting"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        self.is_trained = False
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML models"""
        if df.empty:
            return df
        
        feature_df = df.copy()
        
        # Temporal features
        feature_df["event_date"] = pd.to_datetime(feature_df["event_date"])
        feature_df["year"] = feature_df["event_date"].dt.year
        feature_df["month"] = feature_df["event_date"].dt.month
        feature_df["day_of_year"] = feature_df["event_date"].dt.dayofyear
        feature_df["quarter"] = feature_df["event_date"].dt.quarter
        feature_df["week"] = feature_df["event_date"].dt.isocalendar().week
        
        # Cyclical encoding for temporal features
        feature_df["month_sin"] = np.sin(2 * np.pi * feature_df["month"] / 12)
        feature_df["month_cos"] = np.cos(2 * np.pi * feature_df["month"] / 12)
        feature_df["day_sin"] = np.sin(2 * np.pi * feature_df["day_of_year"] / 365)
        feature_df["day_cos"] = np.cos(2 * np.pi * feature_df["day_of_year"] / 365)
        
        # Location-based features
        if "latitude" in feature_df.columns and "longitude" in feature_df.columns:
            # Distance from reference points (major cities)
            goma_lat, goma_lon = -1.6833, 29.2333
            bukavu_lat, bukavu_lon = -2.5167, 28.8667
            
            feature_df["dist_from_goma"] = self._haversine_distance(
                feature_df["latitude"], feature_df["longitude"], goma_lat, goma_lon
            )
            feature_df["dist_from_bukavu"] = self._haversine_distance(
                feature_df["latitude"], feature_df["longitude"], bukavu_lat, bukavu_lon
            )
        
        # Actor-based features
        if "actor1" in feature_df.columns:
            # Actor frequency encoding
            actor_counts = feature_df["actor1"].value_counts()
            feature_df["actor_frequency"] = feature_df["actor1"].map(actor_counts)
            
            # Actor type classification
            actor_types = self._classify_actors(feature_df["actor1"])
            feature_df["actor_type"] = actor_types
        
        # Event type encoding
        if "event_type" in feature_df.columns:
            event_type_counts = feature_df["event_type"].value_counts()
            feature_df["event_type_frequency"] = feature_df["event_type"].map(event_type_counts)
        
        # Lag features (for time series)
        feature_df = feature_df.sort_values("event_date")
        feature_df["fatalities_lag_7"] = feature_df["fatalities"].rolling(7).mean()
        feature_df["events_lag_7"] = feature_df.groupby(feature_df["event_date"].dt.date)["event_id"].transform("count").rolling(7).mean()
        
        # Violence intensity features
        feature_df["violence_intensity"] = pd.cut(
            feature_df["fatalities"], 
            bins=[-1, 0, 5, 20, 50, float("inf")], 
            labels=["none", "low", "medium", "high", "extreme"]
        )
        
        return feature_df
    
    def train_fatalities_model(self, df: pd.DataFrame) -> Dict:
        """Train model to predict fatalities"""
        feature_df = self.prepare_features(df)
        
        # Select features
        numeric_features = [
            "year", "month", "day_of_year", "quarter", "week",
            "month_sin", "month_cos", "day_sin", "day_cos",
            "dist_from_goma", "dist_from_bukavu",
            "actor_frequency", "event_type_frequency",
            "fatalities_lag_7", "events_lag_7"
        ]
        
        # Filter available features
        available_features = [f for f in numeric_features if f in feature_df.columns]
        
        if len(available_features) < 5:
            return {"error": "Insufficient features for training"}
        
        X = feature_df[available_features].fillna(0)
        y = feature_df["fatalities"]
        
        # Remove outliers (extreme fatalities)
        Q1 = y.quantile(0.25)
        Q3 = y.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        mask = (y >= lower_bound) & (y <= upper_bound)
        X_clean = X[mask]
        y_clean = y[mask]
        
        if len(X_clean) < 20:
            return {"error": "Insufficient data after outlier removal"}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train multiple models
        models = {
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "Ridge": Ridge(alpha=1.0),
            "Linear": LinearRegression()
        }
        
        results = {}
        best_model = None
        best_score = float("inf")
        
        for name, model in models.items():
            try:
                if name in ["Ridge", "Linear"]:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results[name] = {
                    "mae": mae,
                    "mse": mse,
                    "rmse": np.sqrt(mse),
                    "r2": r2
                }
                
                # Store model
                self.models[f"fatalities_{name}"] = model
                
                # Track best model
                if mae < best_score:
                    best_score = mae
                    best_model = f"fatalities_{name}"
                
                # Feature importance for tree-based models
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[f"fatalities_{name}"] = dict(
                        zip(available_features, model.feature_importances_)
                    )
                
            except Exception as e:
                results[name] = {"error": str(e)}
        
        # Store scaler
        self.scalers["fatalities"] = scaler
        
        return {
            "model_results": results,
            "best_model": best_model,
            "features_used": available_features,
            "training_samples": len(X_train),
            "test_samples": len(X_test)
        }
    
    def train_event_frequency_model(self, df: pd.DataFrame) -> Dict:
        """Train model to predict event frequency"""
        # Aggregate by date
        daily_data = df.groupby(df["event_date"].dt.date).agg({
            "event_id": "count",
            "fatalities": "sum",
            "latitude": "mean",
            "longitude": "mean"
        }).reset_index()
        
        daily_data.columns = ["date", "event_count", "total_fatalities", "avg_lat", "avg_lon"]
        daily_data["date"] = pd.to_datetime(daily_data["date"])
        
        # Prepare features
        feature_df = daily_data.copy()
        feature_df["year"] = feature_df["date"].dt.year
        feature_df["month"] = feature_df["date"].dt.month
        feature_df["day_of_year"] = feature_df["date"].dt.dayofyear
        
        # Lag features
        feature_df["event_count_lag_7"] = feature_df["event_count"].rolling(7).mean()
        feature_df["fatalities_lag_7"] = feature_df["total_fatalities"].rolling(7).mean()
        
        # Moving averages
        feature_df["event_count_ma_7"] = feature_df["event_count"].rolling(7).mean()
        feature_df["event_count_ma_30"] = feature_df["event_count"].rolling(30, min_periods=1).mean()
        
        # Prepare training data
        feature_df = feature_df.dropna()
        
        if len(feature_df) < 30:
            return {"error": "Insufficient daily data for training"}
        
        features = ["year", "month", "day_of_year", "event_count_lag_7", 
                   "fatalities_lag_7", "event_count_ma_7", "event_count_ma_30"]
        
        X = feature_df[features]
        y = feature_df["event_count"]
        
        # Split data (time series split)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        models = {
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "Ridge": Ridge(alpha=1.0)
        }
        
        results = {}
        
        for name, model in models.items():
            try:
                if name == "Ridge":
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results[name] = {
                    "mae": mae,
                    "mse": mse,
                    "rmse": np.sqrt(mse),
                    "r2": r2
                }
                
                # Store model
                self.models[f"frequency_{name}"] = model
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[f"frequency_{name}"] = dict(
                        zip(features, model.feature_importances_)
                    )
                
            except Exception as e:
                results[name] = {"error": str(e)}
        
        # Store scaler
        self.scalers["frequency"] = scaler
        
        return {
            "model_results": results,
            "features_used": features,
            "training_samples": len(X_train),
            "test_samples": len(X_test)
        }
    
    def predict_fatalities(self, df: pd.DataFrame, days_ahead: int = 7) -> Dict:
        """Predict fatalities for future periods"""
        if not self.models:
            return {"error": "Models not trained yet"}
        
        # Prepare features
        feature_df = self.prepare_features(df)
        
        # Get the best fatalities model
        fatalities_models = [k for k in self.models.keys() if k.startswith("fatalities_")]
        if not fatalities_models:
            return {"error": "No fatalities model available"}
        
        best_model_key = fatalities_models[0]  # Use first available
        model = self.models[best_model_key]
        
        # Get recent data for prediction
        recent_data = feature_df.tail(30)  # Last 30 days
        
        if recent_data.empty:
            return {"error": "No recent data available for prediction"}
        
        # Prepare prediction features
        numeric_features = [
            "year", "month", "day_of_year", "quarter", "week",
            "month_sin", "month_cos", "day_sin", "day_cos",
            "dist_from_goma", "dist_from_bukavu",
            "actor_frequency", "event_type_frequency",
            "fatalities_lag_7", "events_lag_7"
        ]
        
        available_features = [f for f in numeric_features if f in recent_data.columns]
        X_pred = recent_data[available_features].fillna(0)
        
        # Scale if needed
        if "RandomForest" not in best_model_key and "GradientBoosting" not in best_model_key:
            scaler = self.scalers.get("fatalities")
            if scaler:
                X_pred = scaler.transform(X_pred)
        
        # Make predictions
        predictions = model.predict(X_pred)
        
        # Generate future predictions
        future_predictions = []
        last_date = recent_data["event_date"].max()
        
        for i in range(days_ahead):
            future_date = last_date + pd.Timedelta(days=i+1)
            # Use average of recent predictions as estimate
            pred_value = np.mean(predictions[-7:]) if len(predictions) >= 7 else np.mean(predictions)
            future_predictions.append({
                "date": future_date.strftime("%Y-%m-%d"),
                "predicted_fatalities": max(0, pred_value),
                "confidence": "medium"  # Simplified confidence
            })
        
        return {
            "predictions": future_predictions,
            "model_used": best_model_key,
            "prediction_period": f"{days_ahead} days",
            "historical_avg": np.mean(df["fatalities"]),
            "predicted_trend": "increasing" if future_predictions[-1]["predicted_fatalities"] > np.mean(df["fatalities"]) else "decreasing"
        }
    
    def predict_hotspots(self, df: pd.DataFrame, days_ahead: int = 7) -> Dict:
        """Predict future conflict hotspots"""
        if not self.models:
            return {"error": "Models not trained yet"}
        
        # Cluster recent events
        recent_data = df.tail(df.shape[0] // 2)  # Use recent half of data
        
        if recent_data.empty:
            return {"error": "No recent data available"}
        
        # Prepare features for clustering
        coords = recent_data[["latitude", "longitude"]].values
        
        # K-means clustering
        kmeans = KMeans(n_clusters=min(10, len(recent_data) // 5), random_state=42)
        clusters = kmeans.fit_predict(coords)
        
        # Analyze cluster intensity
        recent_data["cluster"] = clusters
        cluster_analysis = recent_data.groupby("cluster").agg({
            "event_id": "count",
            "fatalities": "sum",
            "latitude": "mean",
            "longitude": "mean"
        }).reset_index()
        
        # Calculate intensity scores
        cluster_analysis["intensity_score"] = (
            cluster_analysis["event_id"] * 0.3 + 
            cluster_analysis["fatalities"] * 0.7
        )
        
        # Rank clusters
        cluster_analysis = cluster_analysis.sort_values("intensity_score", ascending=False)
        
        # Generate hotspot predictions
        hotspot_predictions = []
        for _, cluster in cluster_analysis.head(5).iterrows():
            hotspot_predictions.append({
                "latitude": cluster["latitude"],
                "longitude": cluster["longitude"],
                "predicted_events": max(1, int(cluster["event_id"] * 1.2)),  # 20% increase estimate
                "predicted_fatalities": max(1, int(cluster["fatalities"] * 1.2)),
                "intensity_score": cluster["intensity_score"],
                "risk_level": "high" if cluster["intensity_score"] > 50 else "medium" if cluster["intensity_score"] > 20 else "low"
            })
        
        return {
            "hotspot_predictions": hotspot_predictions,
            "prediction_period": f"{days_ahead} days",
            "total_predicted_hotspots": len(hotspot_predictions),
            "high_risk_hotspots": len([h for h in hotspot_predictions if h["risk_level"] == "high"])
        }
    
    def get_feature_importance(self, model_type: str = "fatalities") -> Dict:
        """Get feature importance for trained models"""
        importance_keys = [k for k in self.feature_importance.keys() if k.startswith(model_type)]
        
        if not importance_keys:
            return {"error": "No feature importance available"}
        
        # Return importance from best model
        best_key = importance_keys[0]
        importance_dict = self.feature_importance[best_key]
        
        # Sort by importance
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "model": best_key,
            "feature_importance": dict(sorted_importance),
            "top_features": sorted_importance[:10]
        }
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two points (supports scalars and Series)"""
        R = 6371  # Earth's radius in kilometers
        
        # Convert to radians (handles both float and pd.Series)
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def _classify_actors(self, actors: pd.Series) -> pd.Series:
        """Classify actors into types"""
        # Simple actor classification based on keywords
        military_keywords = ["army", "military", "forces", "fardc", "police"]
        rebel_keywords = ["m23", "adf", "mai-mai", "rebel", "militia"]
        civilian_keywords = ["civilians", "protesters", "demonstrators"]
        unknown_keywords = ["unknown", "unidentified"]
        
        actor_types = []
        for actor in actors:
            actor_lower = str(actor).lower()
            if any(keyword in actor_lower for keyword in military_keywords):
                actor_types.append("military")
            elif any(keyword in actor_lower for keyword in rebel_keywords):
                actor_types.append("rebel")
            elif any(keyword in actor_lower for keyword in civilian_keywords):
                actor_types.append("civilian")
            elif any(keyword in actor_lower for keyword in unknown_keywords):
                actor_types.append("unknown")
            else:
                actor_types.append("other")
        
        return pd.Series(actor_types)