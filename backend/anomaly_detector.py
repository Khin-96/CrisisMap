import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

class ConflictAnomalyDetector:
    """Automated anomaly detection for conflict patterns"""
    
    def __init__(self):
        self.anomaly_models = {}
        self.anomaly_thresholds = {}
        self.baseline_statistics = {}
        self.detected_anomalies = []
    
    def load_data(self, df: pd.DataFrame):
        """Load conflict data for anomaly detection"""
        self.data = df.copy()
        self.data["event_date"] = pd.to_datetime(self.data["event_date"])
        self.data = self.data.sort_values("event_date")
    
    def detect_statistical_anomalies(self, 
                                    variable: str = "fatalities",
                                    method: str = "zscore",
                                    threshold: float = 3.0) -> Dict:
        """Detect statistical anomalies in conflict data"""
        
        if self.data is None:
            raise ValueError("No data loaded")
        
        anomalies = []
        
        # Prepare time series data
        daily_data = self.data.groupby(self.data["event_date"].dt.date).agg({
            variable: "sum",
            "event_id": "count"
        }).reset_index()
        daily_data.columns = ["date", variable, "event_count"]
        daily_data["date"] = pd.to_datetime(daily_data["date"])
        
        if method == "zscore":
            anomalies = self._zscore_anomaly_detection(daily_data, variable, threshold)
        elif method == "iqr":
            anomalies = self._iqr_anomaly_detection(daily_data, variable, threshold)
        elif method == "modified_zscore":
            anomalies = self._modified_zscore_detection(daily_data, variable, threshold)
        elif method == "percentile":
            anomalies = self._percentile_anomaly_detection(daily_data, variable, threshold)
        
        # Calculate baseline statistics
        baseline_stats = self._calculate_baseline_statistics(daily_data[variable])
        
        return {
            "anomalies": anomalies,
            "baseline_statistics": baseline_stats,
            "method": method,
            "threshold": threshold,
            "total_anomalies": len(anomalies),
            "anomaly_rate": len(anomalies) / len(daily_data) if len(daily_data) > 0 else 0
        }
    
    def detect_spatial_anomalies(self, 
                                 method: str = "dbscan",
                                 eps: float = 0.5) -> Dict:
        """Detect spatial anomalies in conflict distribution"""
        
        if self.data is None:
            raise ValueError("No data loaded")
        
        # Prepare coordinate data
        coords = self.data[["latitude", "longitude"]].values
        
        if method == "dbscan":
            spatial_anomalies = self._dbscan_spatial_anomaly_detection(coords, eps)
        elif method == "isolation_forest":
            spatial_anomalies = self._isolation_forest_spatial_detection(coords)
        elif method == "density_based":
            spatial_anomalies = self._density_based_spatial_detection(coords)
        
        # Analyze anomalous locations
        if spatial_anomalies:
            anomaly_indices = [i for i, is_anomaly in enumerate(spatial_anomalies) if is_anomaly]
            anomaly_locations = self.data.iloc[anomaly_indices]
            
            location_analysis = {
                "anomalous_locations": anomaly_locations[["location", "latitude", "longitude", "event_type", "fatalities"]].to_dict("records"),
                "total_anomalous_events": len(anomaly_locations),
                "anomalous_locations_count": anomaly_locations["location"].nunique(),
                "common_event_types": anomaly_locations["event_type"].value_counts().to_dict()
            }
        else:
            location_analysis = {"anomalous_locations": [], "total_anomalous_events": 0}
        
        return {
            "spatial_anomalies": spatial_anomalies,
            "location_analysis": location_analysis,
            "method": method,
            "total_events": len(self.data)
        }
    
    def detect_temporal_anomalies(self, 
                                 frequency: str = "weekly",
                                 methods: List[str] = None) -> Dict:
        """Detect temporal anomalies in conflict patterns"""
        
        if methods is None:
            methods = ["statistical", "pattern", "seasonal"]
        
        # Aggregate data by frequency
        if frequency == "daily":
            freq_data = self.data.groupby(self.data["event_date"].dt.date).agg({
                "event_id": "count",
                "fatalities": "sum"
            }).reset_index()
        elif frequency == "weekly":
            freq_data = self.data.groupby(self.data["event_date"].dt.to_period("W")).agg({
                "event_id": "count",
                "fatalities": "sum"
            }).reset_index()
        elif frequency == "monthly":
            freq_data = self.data.groupby(self.data["event_date"].dt.to_period("M")).agg({
                "event_id": "count",
                "fatalities": "sum"
            }).reset_index()
        
        temporal_anomalies = {}
        
        # Statistical anomalies
        if "statistical" in methods:
            stat_anomalies = self._detect_statistical_temporal_anomalies(freq_data)
            temporal_anomalies["statistical"] = stat_anomalies
        
        # Pattern anomalies
        if "pattern" in methods:
            pattern_anomalies = self._detect_pattern_temporal_anomalies(freq_data)
            temporal_anomalies["pattern"] = pattern_anomalies
        
        # Seasonal anomalies
        if "seasonal" in methods:
            seasonal_anomalies = self._detect_seasonal_temporal_anomalies(freq_data)
            temporal_anomalies["seasonal"] = seasonal_anomalies
        
        return {
            "temporal_anomalies": temporal_anomalies,
            "frequency": frequency,
            "methods_used": methods,
            "data_points": len(freq_data)
        }
    
    def detect_multivariate_anomalies(self, 
                                    variables: List[str] = None,
                                    method: str = "isolation_forest") -> Dict:
        """Detect multivariate anomalies across multiple variables"""
        
        if variables is None:
            variables = ["fatalities", "event_id"]
        
        # Prepare multivariate dataset
        daily_data = self.data.groupby(self.data["event_date"].dt.date).agg({
            "fatalities": "sum",
            "event_id": "count",
            "latitude": "mean",
            "longitude": "mean"
        }).reset_index()
        
        # Add derived variables
        daily_data["fatalities_per_event"] = daily_data["fatalities"] / daily_data["event_id"]
        daily_data["fatalities_per_event"] = daily_data["fatalities_per_event"].fillna(0)
        
        # Select available variables
        available_vars = [var for var in variables if var in daily_data.columns]
        
        if len(available_vars) < 2:
            return {"error": "Insufficient variables for multivariate analysis"}
        
        X = daily_data[available_vars].fillna(0)
        
        # Detect anomalies
        if method == "isolation_forest":
            anomaly_scores = self._isolation_forest_multivariate(X)
        elif method == "one_class_svm":
            anomaly_scores = self._one_class_svm_multivariate(X)
        elif method == "elliptic_envelope":
            anomaly_scores = self._elliptic_envelope_multivariate(X)
        
        # Identify anomalies
        threshold = np.percentile(anomaly_scores, 95)  # Top 5% as anomalies
        anomaly_flags = anomaly_scores > threshold
        
        # Analyze anomalous periods
        anomalous_periods = daily_data[anomaly_flags]
        
        return {
            "anomaly_scores": anomaly_scores.tolist(),
            "anomaly_flags": anomaly_flags.tolist(),
            "threshold": threshold,
            "anomalous_periods": anomalous_periods.to_dict("records"),
            "total_anomalies": int(anomaly_flags.sum()),
            "anomaly_rate": float(anomaly_flags.mean()),
            "variables_used": available_vars,
            "method": method
        }
    
    def detect_event_sequence_anomalies(self, 
                                       sequence_length: int = 7) -> Dict:
        """Detect anomalies in event sequences"""
        
        # Create event sequences
        event_sequences = self._create_event_sequences(sequence_length)
        
        if len(event_sequences) < 10:
            return {"error": "Insufficient sequences for analysis"}
        
        # Analyze sequence patterns
        sequence_anomalies = self._analyze_sequence_patterns(event_sequences)
        
        # Detect unusual event combinations
        combination_anomalies = self._detect_unusual_combinations(event_sequences)
        
        return {
            "sequence_anomalies": sequence_anomalies,
            "combination_anomalies": combination_anomalies,
            "sequence_length": sequence_length,
            "total_sequences": len(event_sequences)
        }
    
    def generate_anomaly_report(self) -> Dict:
        """Generate comprehensive anomaly detection report"""
        
        if self.data is None:
            return {"error": "No data loaded"}
        
        report = {
            "report_metadata": {
                "generated_at": pd.Timestamp.now().isoformat(),
                "data_period": f"{self.data['event_date'].min().strftime('%Y-%m-%d')} to {self.data['event_date'].max().strftime('%Y-%m-%d')}",
                "total_events": len(self.data)
            }
        }
        
        # Statistical anomalies
        stat_anomalies = self.detect_statistical_anomalies()
        report["statistical_anomalies"] = stat_anomalies
        
        # Spatial anomalies
        spatial_anomalies = self.detect_spatial_anomalies()
        report["spatial_anomalies"] = spatial_anomalies
        
        # Temporal anomalies
        temporal_anomalies = self.detect_temporal_anomalies()
        report["temporal_anomalies"] = temporal_anomalies
        
        # Multivariate anomalies
        multivariate_anomalies = self.detect_multivariate_anomalies()
        report["multivariate_anomalies"] = multivariate_anomalies
        
        # Summary statistics
        report["anomaly_summary"] = {
            "total_statistical_anomalies": stat_anomalies.get("total_anomalies", 0),
            "total_spatial_anomalies": spatial_anomalies.get("location_analysis", {}).get("total_anomalous_events", 0),
            "total_temporal_anomalies": sum(
                len(temp_anomalies.get(method, {}).get("anomalies", []))
                for method in temporal_anomalies.get("temporal_anomalies", {})
            ),
            "total_multivariate_anomalies": multivariate_anomalies.get("total_anomalies", 0)
        }
        
        # Key insights
        report["key_insights"] = self._generate_anomaly_insights(report)
        
        return report
    
    def _zscore_anomaly_detection(self, data: pd.DataFrame, variable: str, threshold: float) -> List[Dict]:
        """Z-score based anomaly detection"""
        
        values = data[variable].values
        z_scores = np.abs(stats.zscore(values))
        
        anomalies = []
        for i, z_score in enumerate(z_scores):
            if z_score > threshold:
                anomalies.append({
                    "date": data.iloc[i]["date"].strftime("%Y-%m-%d"),
                    "value": float(values[i]),
                    "z_score": float(z_score),
                    "anomaly_type": "statistical_outlier",
                    "severity": "high" if z_score > 4 else "medium"
                })
        
        return anomalies
    
    def _iqr_anomaly_detection(self, data: pd.DataFrame, variable: str, multiplier: float) -> List[Dict]:
        """IQR-based anomaly detection"""
        
        values = data[variable].values
        Q1 = np.percentile(values, 25)
        Q3 = np.percentile(values, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        anomalies = []
        for i, value in enumerate(values):
            if value < lower_bound or value > upper_bound:
                anomalies.append({
                    "date": data.iloc[i]["date"].strftime("%Y-%m-%d"),
                    "value": float(value),
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                    "anomaly_type": "iqr_outlier",
                    "severity": "high" if value < Q1 - 2 * IQR or value > Q3 + 2 * IQR else "medium"
                })
        
        return anomalies
    
    def _modified_zscore_detection(self, data: pd.DataFrame, variable: str, threshold: float) -> List[Dict]:
        """Modified Z-score using median"""
        
        values = data[variable].values
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        
        modified_z_scores = 0.6745 * (values - median) / mad if mad != 0 else np.zeros_like(values)
        
        anomalies = []
        for i, z_score in enumerate(modified_z_scores):
            if abs(z_score) > threshold:
                anomalies.append({
                    "date": data.iloc[i]["date"].strftime("%Y-%m-%d"),
                    "value": float(values[i]),
                    "modified_z_score": float(z_score),
                    "anomaly_type": "modified_zscore_outlier",
                    "severity": "high" if abs(z_score) > 4 else "medium"
                })
        
        return anomalies
    
    def _percentile_anomaly_detection(self, data: pd.DataFrame, variable: str, percentile: float) -> List[Dict]:
        """Percentile-based anomaly detection"""
        
        values = data[variable].values
        upper_threshold = np.percentile(values, 100 - percentile)
        lower_threshold = np.percentile(values, percentile)
        
        anomalies = []
        for i, value in enumerate(values):
            if value > upper_threshold or value < lower_threshold:
                anomalies.append({
                    "date": data.iloc[i]["date"].strftime("%Y-%m-%d"),
                    "value": float(value),
                    "upper_threshold": float(upper_threshold),
                    "lower_threshold": float(lower_threshold),
                    "anomaly_type": "percentile_outlier",
                    "severity": "high" if value > np.percentile(values, 99) or value < np.percentile(values, 1) else "medium"
                })
        
        return anomalies
    
    def _dbscan_spatial_anomaly_detection(self, coords: np.ndarray, eps: float) -> List[bool]:
        """DBSCAN-based spatial anomaly detection"""
        
        clustering = DBSCAN(eps=eps, min_samples=5).fit(coords)
        
        # Noise points (-1) are considered anomalies
        anomalies = clustering.labels_ == -1
        
        return anomalies.tolist()
    
    def _isolation_forest_spatial_detection(self, coords: np.ndarray) -> List[bool]:
        """Isolation Forest for spatial anomalies"""
        
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(coords)
        
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_labels = iso_forest.fit_predict(coords_scaled)
        
        # -1 indicates anomalies
        anomalies = anomaly_labels == -1
        
        return anomalies.tolist()
    
    def _density_based_spatial_detection(self, coords: np.ndarray) -> List[bool]:
        """Density-based spatial anomaly detection"""
        
        from sklearn.neighbors import NearestNeighbors
        
        # Calculate k-distance
        k = 5
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        
        # Use distance to k-th neighbor as density measure
        k_distances = distances[:, k]
        
        # Points with high k-distance are anomalies (low density)
        threshold = np.percentile(k_distances, 90)
        anomalies = k_distances > threshold
        
        return anomalies.tolist()
    
    def _detect_statistical_temporal_anomalies(self, freq_data: pd.DataFrame) -> Dict:
        """Detect statistical anomalies in temporal data"""
        
        anomalies = []
        
        for variable in ["event_id", "fatalities"]:
            if variable in freq_data.columns:
                values = freq_data[variable].values
                z_scores = np.abs(stats.zscore(values))
                
                variable_anomalies = []
                for i, z_score in enumerate(z_scores):
                    if z_score > 3:
                        variable_anomalies.append({
                            "period": str(freq_data.iloc[i].name),
                            "value": float(values[i]),
                            "z_score": float(z_score),
                            "variable": variable
                        })
                
                anomalies.extend(variable_anomalies)
        
        return {"anomalies": anomalies, "method": "statistical"}
    
    def _detect_pattern_temporal_anomalies(self, freq_data: pd.DataFrame) -> Dict:
        """Detect pattern-based temporal anomalies"""
        
        anomalies = []
        
        # Look for unusual patterns (e.g., sudden spikes)
        for variable in ["event_id", "fatalities"]:
            if variable in freq_data.columns:
                values = freq_data[variable].values
                
                # Calculate week-over-week change
                changes = np.diff(values)
                change_z_scores = np.abs(stats.zscore(changes))
                
                for i, z_score in enumerate(change_z_scores):
                    if z_score > 3:
                        anomalies.append({
                            "period": str(freq_data.iloc[i+1].name),
                            "change": float(changes[i]),
                            "change_z_score": float(z_score),
                            "variable": variable,
                            "pattern_type": "sudden_change"
                        })
        
        return {"anomalies": anomalies, "method": "pattern"}
    
    def _detect_seasonal_temporal_anomalies(self, freq_data: pd.DataFrame) -> Dict:
        """Detect seasonal anomalies"""
        
        anomalies = []
        
        # Simple seasonal anomaly detection
        for variable in ["event_id", "fatalities"]:
            if variable in freq_data.columns:
                values = freq_data[variable].values
                
                # Calculate seasonal patterns (if enough data)
                if len(values) >= 52:  # At least one year of weekly data
                    # Calculate seasonal averages
                    seasonal_averages = []
                    for week in range(52):
                        week_values = values[week::52]
                        if len(week_values) > 0:
                            seasonal_averages.append(np.mean(week_values))
                        else:
                            seasonal_averages.append(0)
                    
                    # Detect deviations from seasonal patterns
                    for i, value in enumerate(values):
                        week_of_year = i % 52
                        seasonal_avg = seasonal_averages[week_of_year]
                        
                        if seasonal_avg > 0:
                            deviation = abs(value - seasonal_avg) / seasonal_avg
                            if deviation > 2.0:  # More than 200% deviation
                                anomalies.append({
                                    "period": str(freq_data.iloc[i].name),
                                    "value": float(value),
                                    "seasonal_average": float(seasonal_avg),
                                    "deviation": float(deviation),
                                    "variable": variable,
                                    "pattern_type": "seasonal_deviation"
                                })
        
        return {"anomalies": anomalies, "method": "seasonal"}
    
    def _isolation_forest_multivariate(self, X: pd.DataFrame) -> np.ndarray:
        """Isolation Forest for multivariate anomaly detection"""
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_scores = iso_forest.fit_predict(X_scaled)
        
        # Convert to anomaly scores (higher = more anomalous)
        return -anomaly_scores
    
    def _one_class_svm_multivariate(self, X: pd.DataFrame) -> np.ndarray:
        """One-Class SVM for multivariate anomaly detection"""
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        svm = OneClassSVM(nu=0.1)
        anomaly_scores = svm.fit_predict(X_scaled)
        
        # Convert to anomaly scores
        return -anomaly_scores
    
    def _elliptic_envelope_multivariate(self, X: pd.DataFrame) -> np.ndarray:
        """Elliptic Envelope for multivariate anomaly detection"""
        
        from sklearn.covariance import EllipticEnvelope
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        envelope = EllipticEnvelope(contamination=0.1)
        anomaly_scores = envelope.fit_predict(X_scaled)
        
        # Convert to anomaly scores
        return -anomaly_scores
    
    def _create_event_sequences(self, sequence_length: int) -> List[List[Dict]]:
        """Create sequences of events for pattern analysis"""
        
        sequences = []
        
        # Sort data by date
        sorted_data = self.data.sort_values("event_date")
        
        # Create sliding window sequences
        for i in range(len(sorted_data) - sequence_length + 1):
            sequence = sorted_data.iloc[i:i+sequence_length]
            sequences.append(sequence.to_dict("records"))
        
        return sequences
    
    def _analyze_sequence_patterns(self, sequences: List[List[Dict]]) -> Dict:
        """Analyze patterns in event sequences"""
        
        # Calculate sequence statistics
        sequence_lengths = [len(seq) for seq in sequences]
        avg_fatalities_per_sequence = [sum(event["fatalities"] for event in seq) for seq in sequences]
        
        # Detect unusual sequences
        fatalities_threshold = np.percentile(avg_fatalities_per_sequence, 95)
        
        unusual_sequences = []
        for i, seq in enumerate(sequences):
            total_fatalities = sum(event["fatalities"] for event in seq)
            if total_fatalities > fatalities_threshold:
                unusual_sequences.append({
                    "sequence_index": i,
                    "total_fatalities": total_fatalities,
                    "sequence_length": len(seq),
                    "event_types": list(set(event["event_type"] for event in seq))
                })
        
        return {
            "unusual_sequences": unusual_sequences,
            "total_sequences": len(sequences),
            "fatalities_threshold": fatalities_threshold
        }
    
    def _detect_unusual_combinations(self, sequences: List[List[Dict]]) -> Dict:
        """Detect unusual event combinations"""
        
        # Count event type combinations
        combinations = {}
        
        for seq in sequences:
            if len(seq) >= 2:
                # Create combination signature
                event_types = [event["event_type"] for event in seq]
                combination = tuple(sorted(event_types))
                
                combinations[combination] = combinations.get(combination, 0) + 1
        
        # Find rare combinations (occur less than 5% of the time)
        total_sequences = len(sequences)
        rare_combinations = {
            combo: count for combo, count in combinations.items()
            if count / total_sequences < 0.05
        }
        
        return {
            "rare_combinations": rare_combinations,
            "total_combinations": len(combinations),
            "rare_combination_count": len(rare_combinations)
        }
    
    def _calculate_baseline_statistics(self, values: pd.Series) -> Dict:
        """Calculate baseline statistics for anomaly detection"""
        
        if values.empty:
            return {}
            
        return {
            "mean": float(values.mean()),
            "median": float(values.median()),
            "std": float(values.std()) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
            "q25": float(values.quantile(0.25)),
            "q75": float(values.quantile(0.75)),
            "skewness": float(values.skew()) if len(values) > 2 else 0.0,
            "kurtosis": float(values.kurtosis()) if len(values) > 2 else 0.0
        }
    
    def _generate_anomaly_insights(self, report: Dict) -> List[str]:
        """Generate insights from anomaly detection results"""
        
        insights = []
        
        # Statistical anomalies
        stat_anomalies = report.get("statistical_anomalies", {})
        if stat_anomalies.get("total_anomalies", 0) > 0:
            insights.append(f"Detected {stat_anomalies['total_anomalies']} statistical outliers in conflict data")
        
        # Spatial anomalies
        spatial_anomalies = report.get("spatial_anomalies", {})
        spatial_count = spatial_anomalies.get("location_analysis", {}).get("total_anomalous_events", 0)
        if spatial_count > 0:
            insights.append(f"Found {spatial_count} events in anomalous spatial locations")
        
        # Temporal anomalies
        temporal_anomalies = report.get("temporal_anomalies", {})
        temp_anomaly_count = sum(
            len(temp_anomalies.get("temporal_anomalies", {}).get(method, {}).get("anomalies", []))
            for method in ["statistical", "pattern", "seasonal"]
            if method in temporal_anomalies.get("temporal_anomalies", {})
        )
        if temp_anomaly_count > 0:
            insights.append(f"Identified {temp_anomaly_count} temporal anomalies in conflict patterns")
        
        # Multivariate anomalies
        multivariate_anomalies = report.get("multivariate_anomalies", {})
        if multivariate_anomalies.get("total_anomalies", 0) > 0:
            insights.append(f"Detected {multivariate_anomalies['total_anomalies']} multivariate anomalies")
        
        # Overall assessment
        total_anomalies = report.get("anomaly_summary", {})
        total_detected = sum(total_anomalies.values())
        
        if total_detected > 0:
            insights.append(f"Total of {total_detected} anomalies detected across all dimensions")
            if total_detected > len(self.data) * 0.1:  # More than 10% anomalies
                insights.append("High anomaly rate suggests significant instability in the region")
            else:
                insights.append("Anomaly rate within expected range for conflict monitoring")
        
        return insights