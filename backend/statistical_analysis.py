import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

class StatisticalAnalyzer:
    """Advanced statistical analysis for conflict data"""
    
    def __init__(self):
        self.data = None
        self.scaler = StandardScaler()
    
    def load_data(self, df: pd.DataFrame):
        """Load conflict data for analysis"""
        self.data = df.copy()
        # Ensure date column is datetime
        self.data["event_date"] = pd.to_datetime(self.data["event_date"])
    
    def temporal_trend_analysis(self, period: str = "monthly") -> Dict:
        """Analyze temporal trends with statistical significance"""
        if self.data is None:
            raise ValueError("No data loaded")
        
        # Group by period
        if period == "monthly":
            self.data["period"] = self.data["event_date"].dt.to_period("M")
        elif period == "weekly":
            self.data["period"] = self.data["event_date"].dt.to_period("W")
        else:
            self.data["period"] = self.data["event_date"].dt.to_period("D")
        
        # Calculate metrics
        temporal_data = self.data.groupby("period").agg({
            "event_id": "count",
            "fatalities": "sum"
        }).reset_index()
        
        temporal_data.columns = ["period", "events", "fatalities"]
        
        # Trend analysis
        if len(temporal_data) >= 3:
            # Linear regression for trend
            x = np.arange(len(temporal_data))
            events_slope, _, events_r, _, _ = stats.linregress(x, temporal_data["events"])
            fatalities_slope, _, fatalities_r, _, _ = stats.linregress(x, temporal_data["fatalities"])
            
            # Mann-Kendall trend test
            events_trend = self._mann_kendall_test(temporal_data["events"])
            fatalities_trend = self._mann_kendall_test(temporal_data["fatalities"])
            
            # Seasonality detection
            seasonality = self._detect_seasonality(temporal_data)
        else:
            events_slope = fatalities_slope = 0
            events_r = fatalities_r = 0
            events_trend = fatalities_trend = "no_data"
            seasonality = {}
        
        return {
            "temporal_data": temporal_data.to_dict("records"),
            "trend_analysis": {
                "events_slope": events_slope,
                "fatalities_slope": fatalities_slope,
                "events_correlation": events_r,
                "fatalities_correlation": fatalities_r,
                "events_trend_significance": events_trend,
                "fatalities_trend_significance": fatalities_trend
            },
            "seasonality": seasonality
        }
    
    def spatial_hotspot_analysis(self, method: str = "dbscan") -> Dict:
        """Identify spatial hotspots using clustering algorithms"""
        if self.data is None:
            raise ValueError("No data loaded")
        
        # Prepare coordinates
        coords = self.data[["latitude", "longitude"]].values
        
        if method == "dbscan":
            # DBSCAN clustering
            clustering = DBSCAN(eps=0.5, min_samples=5).fit(coords)
            self.data["cluster"] = clustering.labels_
            
            # Analyze clusters
            clusters = []
            for cluster_id in set(clustering.labels_):
                if cluster_id != -1:  # Ignore noise
                    cluster_data = self.data[self.data["cluster"] == cluster_id]
                    clusters.append({
                        "cluster_id": int(cluster_id),
                        "center_lat": float(cluster_data["latitude"].mean()),
                        "center_lon": float(cluster_data["longitude"].mean()),
                        "event_count": len(cluster_data),
                        "total_fatalities": int(cluster_data["fatalities"].sum()),
                        "radius_km": self._calculate_cluster_radius(cluster_data),
                        "intensity_score": self._calculate_intensity_score(cluster_data)
                    })
        
        # Get top hotspots
        hotspots = sorted(clusters, key=lambda x: x["intensity_score"], reverse=True)
        
        return {
            "method": method,
            "hotspots": hotspots,
            "total_clusters": len(clusters),
            "noise_points": int(sum(clustering.labels_ == -1))
        }
    
    def actor_network_analysis(self) -> Dict:
        """Analyze actor patterns and co-occurrence"""
        if self.data is None:
            raise ValueError("No data loaded")
        
        # Actor statistics
        actor_stats = self.data.groupby("actor1").agg({
            "event_id": "count",
            "fatalities": "sum",
            "event_type": lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "Unknown"
        }).reset_index()
        
        actor_stats.columns = ["actor", "events", "fatalities", "primary_event_type"]
        
        # Calculate actor metrics
        actor_stats["fatality_rate"] = actor_stats["fatalities"] / actor_stats["events"]
        actor_stats["activity_score"] = self._calculate_activity_score(actor_stats)
        
        # Actor co-occurrence network (if actor2 exists)
        network_data = {}
        if "actor2" in self.data.columns:
            co_occurrences = self.data[self.data["actor2"].notna()]
            network_edges = co_occurrences.groupby(["actor1", "actor2"]).size().reset_index()
            network_edges.columns = ["source", "target", "weight"]
            network_data["edges"] = network_edges.to_dict("records")
        
        return {
            "actor_statistics": actor_stats.sort_values("activity_score", ascending=False).to_dict("records"),
            "network_analysis": network_data,
            "top_actors": actor_stats.nlargest(10, "activity_score").to_dict("records")
        }
    
    def violence_pattern_analysis(self) -> Dict:
        """Analyze patterns of violence and escalation"""
        if self.data is None:
            raise ValueError("No data loaded")
        
        # Event type analysis
        event_patterns = self.data.groupby("event_type").agg({
            "event_id": "count",
            "fatalities": ["sum", "mean", "std"]
        }).round(2)
        
        event_patterns.columns = ["count", "total_fatalities", "mean_fatalities", "std_fatalities"]
        
        # Escalation detection
        escalation_events = self._detect_escalation_patterns()
        
        # Violence intensity classification
        self.data["violence_intensity"] = self._classify_violence_intensity()
        
        intensity_distribution = self.data["violence_intensity"].value_counts().to_dict()
        
        return {
            "event_patterns": event_patterns.reset_index().to_dict("records"),
            "escalation_events": escalation_events,
            "intensity_distribution": intensity_distribution,
            "high_intensity_periods": self._identify_high_intensity_periods()
        }
    
    def early_warning_indicators(self) -> Dict:
        """Calculate early warning indicators for conflict escalation"""
        if self.data is None:
            raise ValueError("No data loaded")
        
        # Recent activity metrics
        recent_cutoff = datetime.now() - timedelta(days=30)
        recent_data = self.data[self.data["event_date"] >= recent_cutoff]
        
        # Baseline metrics (previous 90 days)
        baseline_cutoff = datetime.now() - timedelta(days=120)
        baseline_data = self.data[(self.data["event_date"] >= baseline_cutoff) & 
                                  (self.data["event_date"] < recent_cutoff)]
        
        # Calculate indicators
        indicators = {
            "activity_spike": self._calculate_activity_spike(recent_data, baseline_data),
            "geographic_spread": self._calculate_geographic_spread(recent_data),
            "actor_diversification": self._calculate_actor_diversification(recent_data),
            "fatality_increase": self._calculate_fatality_increase(recent_data, baseline_data),
            "event_type_diversification": self._calculate_event_diversification(recent_data),
            "risk_score": 0  # Will be calculated below
        }
        
        # Overall risk score
        indicators["risk_score"] = self._calculate_overall_risk_score(indicators)
        
        return indicators
    
    def _mann_kendall_test(self, data: pd.Series) -> str:
        """Mann-Kendall trend test"""
        n = len(data)
        if n < 3:
            return "no_data"
        
        # Calculate S statistic
        s = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                if data[j] > data[i]:
                    s += 1
                elif data[j] < data[i]:
                    s -= 1
        
        # Calculate variance
        var_s = n * (n - 1) * (2 * n + 5) / 18
        
        # Calculate Z statistic
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
        
        # Determine significance
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        if p_value < 0.05:
            return "increasing" if s > 0 else "decreasing"
        else:
            return "no_trend"
    
    def _detect_seasonality(self, temporal_data: pd.DataFrame) -> Dict:
        """Detect seasonal patterns in temporal data"""
        if len(temporal_data) < 12:
            return {"seasonal": False}
        
        # Simple seasonality detection
        events = temporal_data["events"].values
        
        # Autocorrelation for seasonality
        autocorr = [np.corrcoef(events[:-i], events[i:])[0, 1] for i in range(1, min(12, len(events)//2))]
        
        return {
            "seasonal": max(autocorr) > 0.3 if autocorr else False,
            "peak_lag": np.argmax(autocorr) + 1 if autocorr else 0,
            "autocorrelations": autocorr
        }
    
    def _calculate_cluster_radius(self, cluster_data: pd.DataFrame) -> float:
        """Calculate radius of cluster in kilometers"""
        center_lat = cluster_data["latitude"].mean()
        center_lon = cluster_data["longitude"].mean()
        
        # Calculate maximum distance from center
        distances = []
        for _, row in cluster_data.iterrows():
            dist = self._haversine_distance(center_lat, center_lon, row["latitude"], row["longitude"])
            distances.append(dist)
        
        return max(distances) if distances else 0
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points using Haversine formula"""
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def _calculate_intensity_score(self, cluster_data: pd.DataFrame) -> float:
        """Calculate intensity score for a cluster"""
        event_count = len(cluster_data)
        total_fatalities = cluster_data["fatalities"].sum()
        
        # Weighted intensity score
        return event_count * 0.3 + total_fatalities * 0.7
    
    def _calculate_activity_score(self, actor_stats: pd.DataFrame) -> float:
        """Calculate activity score for actors"""
        # Normalize metrics
        max_events = actor_stats["events"].max()
        max_fatalities = actor_stats["fatalities"].max()
        
        if max_events > 0 and max_fatalities > 0:
            normalized_events = actor_stats["events"] / max_events
            normalized_fatalities = actor_stats["fatalities"] / max_fatalities
            return (normalized_events * 0.6 + normalized_fatalities * 0.4).mean()
        return 0
    
    def _detect_escalation_patterns(self) -> List[Dict]:
        """Detect patterns of conflict escalation"""
        escalation_events = []
        
        # Look for rapid increases in fatalities
        self.data = self.data.sort_values("event_date")
        
        for i in range(1, len(self.data)):
            current_fatalities = self.data.iloc[i]["fatalities"]
            previous_avg = self.data.iloc[max(0, i-5):i]["fatalities"].mean()
            
            if current_fatalities > previous_avg * 3 and current_fatalities > 10:
                escalation_events.append({
                    "date": self.data.iloc[i]["event_date"].strftime("%Y-%m-%d"),
                    "location": self.data.iloc[i]["location"],
                    "fatalities": current_fatalities,
                    "severity": "high" if current_fatalities > 50 else "medium"
                })
        
        return escalation_events
    
    def _classify_violence_intensity(self) -> pd.Series:
        """Classify violence intensity based on fatalities"""
        def classify(fatalities):
            if fatalities == 0:
                return "none"
            elif fatalities <= 5:
                return "low"
            elif fatalities <= 20:
                return "medium"
            elif fatalities <= 50:
                return "high"
            else:
                return "extreme"
        
        return self.data["fatalities"].apply(classify)
    
    def _identify_high_intensity_periods(self) -> List[Dict]:
        """Identify periods with high violence intensity"""
        high_intensity = self.data[self.data["violence_intensity"].isin(["high", "extreme"])]
        
        if high_intensity.empty:
            return []
        
        # Group by week to identify intense periods
        high_intensity["week"] = high_intensity["event_date"].dt.to_period("W")
        weekly_intensity = high_intensity.groupby("week").agg({
            "event_id": "count",
            "fatalities": "sum"
        }).reset_index()
        
        # Identify top intense periods
        intense_periods = weekly_intensity.nlargest(5, "fatalities")
        
        return intense_periods.to_dict("records")
    
    def _calculate_activity_spike(self, recent_data: pd.DataFrame, baseline_data: pd.DataFrame) -> float:
        """Calculate activity spike compared to baseline"""
        if baseline_data.empty:
            return 0
        
        recent_events = len(recent_data)
        baseline_events = len(baseline_data) / 3  # Normalize to same period length
        
        if baseline_events == 0:
            return 1.0 if recent_events > 0 else 0
        
        return recent_events / baseline_events
    
    def _calculate_geographic_spread(self, recent_data: pd.DataFrame) -> float:
        """Calculate geographic spread of recent events"""
        if recent_data.empty:
            return 0
        
        locations = recent_data[["latitude", "longitude"]].drop_duplicates()
        return len(locations)
    
    def _calculate_actor_diversification(self, recent_data: pd.DataFrame) -> float:
        """Calculate actor diversification"""
        if recent_data.empty:
            return 0
        
        unique_actors = recent_data["actor1"].nunique()
        total_events = len(recent_data)
        
        return unique_actors / total_events if total_events > 0 else 0
    
    def _calculate_fatality_increase(self, recent_data: pd.DataFrame, baseline_data: pd.DataFrame) -> float:
        """Calculate fatality increase compared to baseline"""
        if baseline_data.empty:
            return 0
        
        recent_fatalities = recent_data["fatalities"].sum()
        baseline_fatalities = baseline_data["fatalities"].sum() / 3  # Normalize
        
        if baseline_fatalities == 0:
            return 1.0 if recent_fatalities > 0 else 0
        
        return recent_fatalities / baseline_fatalities
    
    def _calculate_event_diversification(self, recent_data: pd.DataFrame) -> float:
        """Calculate event type diversification"""
        if recent_data.empty:
            return 0
        
        unique_event_types = recent_data["event_type"].nunique()
        total_events = len(recent_data)
        
        return unique_event_types / total_events if total_events > 0 else 0
    
    def _calculate_overall_risk_score(self, indicators: Dict) -> float:
        """Calculate overall risk score from indicators"""
        weights = {
            "activity_spike": 0.25,
            "geographic_spread": 0.15,
            "actor_diversification": 0.15,
            "fatality_increase": 0.3,
            "event_type_diversification": 0.15
        }
        
        risk_score = 0
        for indicator, weight in weights.items():
            # Normalize indicator values
            normalized_value = min(indicators[indicator], 2.0) / 2.0  # Cap at 2x baseline
            risk_score += normalized_value * weight
        
        return min(risk_score, 1.0)