import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime, timedelta

class DataQualityValidator:
    """Comprehensive data quality assessment and validation framework"""
    
    def __init__(self):
        self.quality_metrics = {}
        self.validation_rules = {}
        self.quality_scores = {}
        self.issues_detected = []
    
    def assess_data_quality(self, df: pd.DataFrame, data_type: str = "conflict_events") -> Dict:
        """Comprehensive data quality assessment"""
        
        quality_report = {
            "assessment_metadata": {
                "timestamp": datetime.now().isoformat(),
                "data_type": data_type,
                "total_records": len(df),
                "total_columns": len(df.columns)
            }
        }
        
        # 1. Completeness Assessment
        completeness = self._assess_completeness(df)
        quality_report["completeness"] = completeness
        
        # 2. Accuracy Assessment
        accuracy = self._assess_accuracy(df, data_type)
        quality_report["accuracy"] = accuracy
        
        # 3. Consistency Assessment
        consistency = self._assess_consistency(df)
        quality_report["consistency"] = consistency
        
        # 4. Validity Assessment
        validity = self._assess_validity(df, data_type)
        quality_report["validity"] = validity
        
        # 5. Uniqueness Assessment
        uniqueness = self._assess_uniqueness(df)
        quality_report["uniqueness"] = uniqueness
        
        # 6. Timeliness Assessment
        timeliness = self._assess_timeliness(df)
        quality_report["timeliness"] = timeliness
        
        # 7. Overall Quality Score
        overall_score = self._calculate_overall_quality_score(quality_report)
        quality_report["overall_quality"] = overall_score
        
        # 8. Quality Issues Summary
        issues_summary = self._summarize_quality_issues(quality_report)
        quality_report["issues_summary"] = issues_summary
        
        # 9. Recommendations
        recommendations = self._generate_quality_recommendations(quality_report)
        quality_report["recommendations"] = recommendations
        
        return quality_report
    
    def validate_data_rules(self, df: pd.DataFrame, rules: Dict = None) -> Dict:
        """Validate data against predefined rules"""
        
        if rules is None:
            rules = self._get_default_validation_rules()
        
        validation_results = {}
        
        for rule_name, rule_config in rules.items():
            rule_result = self._apply_validation_rule(df, rule_name, rule_config)
            validation_results[rule_name] = rule_result
        
        # Calculate validation summary
        total_rules = len(validation_results)
        passed_rules = sum(1 for result in validation_results.values() if result["status"] == "passed")
        
        validation_summary = {
            "total_rules": total_rules,
            "passed_rules": passed_rules,
            "failed_rules": total_rules - passed_rules,
            "pass_rate": passed_rules / total_rules if total_rules > 0 else 0,
            "validation_results": validation_results
        }
        
        return validation_summary
    
    def detect_data_anomalies(self, df: pd.DataFrame) -> Dict:
        """Detect anomalies in data that might indicate quality issues"""
        
        anomalies = {
            "missing_data_anomalies": self._detect_missing_data_anomalies(df),
            "duplicate_anomalies": self._detect_duplicate_anomalies(df),
            "outlier_anomalies": self._detect_outlier_anomalies(df),
            "format_anomalies": self._detect_format_anomalies(df),
            "logic_anomalies": self._detect_logic_anomalies(df)
        }
        
        # Calculate anomaly summary
        total_anomalies = sum(len(anomaly_list) for anomaly_list in anomalies.values())
        
        anomaly_summary = {
            "total_anomalies": total_anomalies,
            "anomaly_types": {k: len(v) for k, v in anomalies.items()},
            "anomaly_details": anomalies
        }
        
        return anomaly_summary
    
    def generate_data_profile(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive data profile"""
        
        profile = {
            "dataset_info": {
                "shape": df.shape,
                "memory_usage": df.memory_usage(deep=True).sum(),
                "dtypes": df.dtypes.to_dict()
            },
            "column_profiles": {}
        }
        
        # Profile each column
        for column in df.columns:
            column_profile = self._profile_column(df[column])
            profile["column_profiles"][column] = column_profile
        
        # Cross-column relationships
        profile["relationships"] = self._analyze_column_relationships(df)
        
        # Data patterns
        profile["patterns"] = self._identify_data_patterns(df)
        
        return profile
    
    def monitor_data_quality_trends(self, 
                                   historical_data: List[pd.DataFrame],
                                   timestamps: List[datetime]) -> Dict:
        """Monitor data quality trends over time"""
        
        if len(historical_data) != len(timestamps):
            raise ValueError("Historical data and timestamps must have same length")
        
        quality_trends = {
            "timestamps": [ts.isoformat() for ts in timestamps],
            "metrics": {}
        }
        
        # Calculate quality metrics for each time period
        for i, (data_df, timestamp) in enumerate(zip(historical_data, timestamps)):
            quality_assessment = self.assess_data_quality(data_df)
            
            for metric_category, metrics in quality_assessment.items():
                if metric_category not in ["assessment_metadata", "overall_quality", "issues_summary", "recommendations"]:
                    if metric_category not in quality_trends["metrics"]:
                        quality_trends["metrics"][metric_category] = {}
                    
                    for metric_name, metric_value in metrics.items():
                        if isinstance(metric_value, (int, float)):
                            if metric_name not in quality_trends["metrics"][metric_category]:
                                quality_trends["metrics"][metric_category][metric_name] = []
                            quality_trends["metrics"][metric_category][metric_name].append(metric_value)
        
        # Calculate trend statistics
        trend_analysis = self._analyze_quality_trends(quality_trends)
        quality_trends["trend_analysis"] = trend_analysis
        
        return quality_trends
    
    def _assess_completeness(self, df: pd.DataFrame) -> Dict:
        """Assess data completeness"""
        
        completeness_scores = {}
        missing_data_summary = {}
        
        for column in df.columns:
            total_count = len(df)
            missing_count = df[column].isnull().sum()
            non_missing_count = total_count - missing_count
            
            completeness_score = non_missing_count / total_count if total_count > 0 else 0
            
            completeness_scores[column] = {
                "completeness_score": completeness_score,
                "missing_count": int(missing_count),
                "non_missing_count": int(non_missing_count),
                "missing_percentage": float(missing_count / total_count * 100) if total_count > 0 else 0
            }
            
            if missing_count > 0:
                missing_data_summary[column] = {
                    "missing_pattern": self._analyze_missing_pattern(df[column]),
                    "missing_impact": "high" if completeness_score < 0.7 else "medium" if completeness_score < 0.9 else "low"
                }
        
        # Overall completeness
        overall_completeness = np.mean([score["completeness_score"] for score in completeness_scores.values()])
        
        return {
            "column_completeness": completeness_scores,
            "missing_data_summary": missing_data_summary,
            "overall_completeness": overall_completeness,
            "completeness_grade": self._grade_completeness(overall_completeness)
        }
    
    def _assess_accuracy(self, df: pd.DataFrame, data_type: str) -> Dict:
        """Assess data accuracy"""
        
        accuracy_results = {}
        
        # Geographic accuracy
        if "latitude" in df.columns and "longitude" in df.columns:
            geo_accuracy = self._assess_geographic_accuracy(df)
            accuracy_results["geographic"] = geo_accuracy
        
        # Temporal accuracy
        if "event_date" in df.columns:
            temporal_accuracy = self._assess_temporal_accuracy(df)
            accuracy_results["temporal"] = temporal_accuracy
        
        # Numeric accuracy
        if "fatalities" in df.columns:
            numeric_accuracy = self._assess_numeric_accuracy(df, "fatalities")
            accuracy_results["fatalities"] = numeric_accuracy
        
        # Categorical accuracy
        if "event_type" in df.columns:
            categorical_accuracy = self._assess_categorical_accuracy(df, "event_type")
            accuracy_results["event_type"] = categorical_accuracy
        
        # Overall accuracy
        accuracy_scores = [result.get("accuracy_score", 0) for result in accuracy_results.values()]
        overall_accuracy = np.mean(accuracy_scores) if accuracy_scores else 0
        
        return {
            "accuracy_results": accuracy_results,
            "overall_accuracy": overall_accuracy,
            "accuracy_grade": self._grade_accuracy(overall_accuracy)
        }
    
    def _assess_consistency(self, df: pd.DataFrame) -> Dict:
        """Assess data consistency"""
        
        consistency_results = {}
        
        # Internal consistency
        internal_consistency = self._assess_internal_consistency(df)
        consistency_results["internal"] = internal_consistency
        
        # Cross-record consistency
        cross_record_consistency = self._assess_cross_record_consistency(df)
        consistency_results["cross_record"] = cross_record_consistency
        
        # Temporal consistency
        if "event_date" in df.columns:
            temporal_consistency = self._assess_temporal_consistency(df)
            consistency_results["temporal"] = temporal_consistency
        
        # Overall consistency
        consistency_scores = [result.get("consistency_score", 0) for result in consistency_results.values()]
        overall_consistency = np.mean(consistency_scores) if consistency_scores else 0
        
        return {
            "consistency_results": consistency_results,
            "overall_consistency": overall_consistency,
            "consistency_grade": self._grade_consistency(overall_consistency)
        }
    
    def _assess_validity(self, df: pd.DataFrame, data_type: str) -> Dict:
        """Assess data validity"""
        
        validity_results = {}
        
        # Range validity
        range_validity = self._assess_range_validity(df, data_type)
        validity_results["range"] = range_validity
        
        # Format validity
        format_validity = self._assess_format_validity(df)
        validity_results["format"] = format_validity
        
        # Domain validity
        domain_validity = self._assess_domain_validity(df, data_type)
        validity_results["domain"] = domain_validity
        
        # Overall validity
        validity_scores = [result.get("validity_score", 0) for result in validity_results.values()]
        overall_validity = np.mean(validity_scores) if validity_scores else 0
        
        return {
            "validity_results": validity_results,
            "overall_validity": overall_validity,
            "validity_grade": self._grade_validity(overall_validity)
        }
    
    def _assess_uniqueness(self, df: pd.DataFrame) -> Dict:
        """Assess data uniqueness"""
        
        uniqueness_results = {}
        
        # Record uniqueness
        total_records = len(df)
        unique_records = len(df.drop_duplicates())
        record_uniqueness = unique_records / total_records if total_records > 0 else 0
        
        uniqueness_results["record_uniqueness"] = {
            "total_records": total_records,
            "unique_records": unique_records,
            "duplicate_records": total_records - unique_records,
            "uniqueness_score": record_uniqueness
        }
        
        # Key field uniqueness
        key_fields = ["event_id", "event_date", "location"]
        
        for field in key_fields:
            if field in df.columns:
                field_uniqueness = self._assess_field_uniqueness(df, field)
                uniqueness_results[f"{field}_uniqueness"] = field_uniqueness
        
        # Overall uniqueness
        uniqueness_scores = [result.get("uniqueness_score", 0) for result in uniqueness_results.values()]
        overall_uniqueness = np.mean(uniqueness_scores) if uniqueness_scores else 0
        
        return {
            "uniqueness_results": uniqueness_results,
            "overall_uniqueness": overall_uniqueness,
            "uniqueness_grade": self._grade_uniqueness(overall_uniqueness)
        }
    
    def _assess_timeliness(self, df: pd.DataFrame) -> Dict:
        """Assess data timeliness"""
        
        timeliness_results = {}
        
        if "event_date" in df.columns:
            # Convert to datetime if needed
            event_dates = pd.to_datetime(df["event_date"])
            
            # Calculate data age
            current_time = datetime.now()
            data_age = current_time - event_dates.max()
            
            # Data recency
            recent_data = event_dates[event_dates > (current_time - timedelta(days=30))]
            recency_score = len(recent_data) / len(event_dates) if len(event_dates) > 0 else 0
            
            # Data frequency
            if len(event_dates) > 1:
                date_range = event_dates.max() - event_dates.min()
                frequency_score = len(event_dates) / max(date_range.days, 1)
            else:
                frequency_score = 0
            
            timeliness_results = {
                "data_age_days": data_age.days,
                "most_recent_date": event_dates.max().isoformat(),
                "oldest_date": event_dates.min().isoformat(),
                "recency_score": recency_score,
                "frequency_score": min(frequency_score / 10, 1),  # Normalize to 0-1
                "date_range_days": (event_dates.max() - event_dates.min()).days
            }
        
        # Overall timeliness
        timeliness_score = 0
        if timeliness_results:
            timeliness_score = np.mean([
                timeliness_results.get("recency_score", 0),
                timeliness_results.get("frequency_score", 0)
            ])
        
        return {
            "timeliness_results": timeliness_results,
            "overall_timeliness": timeliness_score,
            "timeliness_grade": self._grade_timeliness(timeliness_score)
        }
    
    def _calculate_overall_quality_score(self, quality_report: Dict) -> Dict:
        """Calculate overall data quality score"""
        
        # Extract individual quality scores
        completeness_score = quality_report.get("completeness", {}).get("overall_completeness", 0)
        accuracy_score = quality_report.get("accuracy", {}).get("overall_accuracy", 0)
        consistency_score = quality_report.get("consistency", {}).get("overall_consistency", 0)
        validity_score = quality_report.get("validity", {}).get("overall_validity", 0)
        uniqueness_score = quality_report.get("uniqueness", {}).get("overall_uniqueness", 0)
        timeliness_score = quality_report.get("timeliness", {}).get("overall_timeliness", 0)
        
        # Weighted average (weights can be adjusted based on importance)
        weights = {
            "completeness": 0.25,
            "accuracy": 0.20,
            "consistency": 0.15,
            "validity": 0.15,
            "uniqueness": 0.10,
            "timeliness": 0.15
        }
        
        overall_score = (
            completeness_score * weights["completeness"] +
            accuracy_score * weights["accuracy"] +
            consistency_score * weights["consistency"] +
            validity_score * weights["validity"] +
            uniqueness_score * weights["uniqueness"] +
            timeliness_score * weights["timeliness"]
        )
        
        return {
            "overall_score": overall_score,
            "component_scores": {
                "completeness": completeness_score,
                "accuracy": accuracy_score,
                "consistency": consistency_score,
                "validity": validity_score,
                "uniqueness": uniqueness_score,
                "timeliness": timeliness_score
            },
            "weights": weights,
            "quality_grade": self._grade_overall_quality(overall_score)
        }
    
    def _summarize_quality_issues(self, quality_report: Dict) -> Dict:
        """Summarize quality issues detected"""
        
        issues = {
            "critical_issues": [],
            "major_issues": [],
            "minor_issues": [],
            "total_issues": 0
        }
        
        # Check completeness issues
        completeness = quality_report.get("completeness", {})
        if completeness.get("overall_completeness", 1) < 0.8:
            issues["major_issues"].append("Low data completeness")
        
        # Check accuracy issues
        accuracy = quality_report.get("accuracy", {})
        if accuracy.get("overall_accuracy", 1) < 0.8:
            issues["major_issues"].append("Low data accuracy")
        
        # Check consistency issues
        consistency = quality_report.get("consistency", {})
        if consistency.get("overall_consistency", 1) < 0.8:
            issues["minor_issues"].append("Data consistency issues detected")
        
        # Check validity issues
        validity = quality_report.get("validity", {})
        if validity.get("overall_validity", 1) < 0.8:
            issues["major_issues"].append("Data validity issues detected")
        
        # Check uniqueness issues
        uniqueness = quality_report.get("uniqueness", {})
        duplicate_rate = 1 - uniqueness.get("overall_uniqueness", 1)
        if duplicate_rate > 0.1:
            issues["minor_issues"].append(f"High duplicate rate: {duplicate_rate:.1%}")
        
        # Check timeliness issues
        timeliness = quality_report.get("timeliness", {})
        if timeliness.get("overall_timeliness", 1) < 0.5:
            issues["major_issues"].append("Data timeliness issues detected")
        
        # Total issues count
        issues["total_issues"] = len(issues["critical_issues"]) + len(issues["major_issues"]) + len(issues["minor_issues"])
        
        return issues
    
    def _generate_quality_recommendations(self, quality_report: Dict) -> List[str]:
        """Generate recommendations for improving data quality"""
        
        recommendations = []
        
        # Completeness recommendations
        completeness = quality_report.get("completeness", {})
        if completeness.get("overall_completeness", 1) < 0.9:
            recommendations.append("Implement data validation to reduce missing values")
            recommendations.append("Establish data entry standards and training")
        
        # Accuracy recommendations
        accuracy = quality_report.get("accuracy", {})
        if accuracy.get("overall_accuracy", 1) < 0.9:
            recommendations.append("Implement automated data verification checks")
            recommendations.append("Establish data source validation procedures")
        
        # Consistency recommendations
        consistency = quality_report.get("consistency", {})
        if consistency.get("overall_consistency", 1) < 0.9:
            recommendations.append("Standardize data formats and conventions")
            recommendations.append("Implement cross-field validation rules")
        
        # Validity recommendations
        validity = quality_report.get("validity", {})
        if validity.get("overall_validity", 1) < 0.9:
            recommendations.append("Define and enforce data validation rules")
            recommendations.append("Implement range and format checks")
        
        # Uniqueness recommendations
        uniqueness = quality_report.get("uniqueness", {})
        if uniqueness.get("overall_uniqueness", 1) < 0.95:
            recommendations.append("Implement duplicate detection and prevention")
            recommendations.append("Establish unique identifier requirements")
        
        # Timeliness recommendations
        timeliness = quality_report.get("timeliness", {})
        if timeliness.get("overall_timeliness", 1) < 0.8:
            recommendations.append("Optimize data collection and processing workflows")
            recommendations.append("Implement real-time data updates where possible")
        
        return recommendations
    
    def _profile_column(self, series: pd.Series) -> Dict:
        """Profile individual column"""
        
        profile = {
            "column_name": series.name,
            "data_type": str(series.dtype),
            "non_null_count": series.count(),
            "null_count": series.isnull().sum(),
            "unique_count": series.nunique(),
            "memory_usage": series.memory_usage(deep=True)
        }
        
        # Numeric columns
        if pd.api.types.is_numeric_dtype(series):
            profile.update({
                "min": series.min(),
                "max": series.max(),
                "mean": series.mean(),
                "median": series.median(),
                "std": series.std(),
                "skewness": series.skew(),
                "kurtosis": series.kurtosis(),
                "quartiles": [series.quantile(q) for q in [0.25, 0.5, 0.75]]
            })
        
        # Categorical columns
        elif pd.api.types.is_categorical_dtype(series) or series.dtype == 'object':
            value_counts = series.value_counts()
            profile.update({
                "most_frequent": value_counts.index[0] if len(value_counts) > 0 else None,
                "most_frequent_count": value_counts.iloc[0] if len(value_counts) > 0 else 0,
                "least_frequent": value_counts.index[-1] if len(value_counts) > 0 else None,
                "least_frequent_count": value_counts.iloc[-1] if len(value_counts) > 0 else 0,
                "cardinality": series.nunique(),
                "top_values": value_counts.head(10).to_dict()
            })
        
        # DateTime columns
        elif pd.api.types.is_datetime64_any_dtype(series):
            profile.update({
                "min_date": series.min().isoformat() if not series.empty else None,
                "max_date": series.max().isoformat() if not series.empty else None,
                "date_range_days": (series.max() - series.min()).days if not series.empty else 0
            })
        
        return profile
    
    def _analyze_missing_pattern(self, series: pd.Series) -> str:
        """Analyze missing data pattern"""
        
        if series.isnull().sum() == 0:
            return "no_missing_data"
        
        # Check for patterns
        missing_indices = series.isnull()
        
        # Completely missing
        if missing_indices.all():
            return "completely_missing"
        
        # Random missing
        if missing_indices.sum() / len(series) < 0.1:
            return "sparse_random"
        
        # Systematic missing (check for blocks)
        consecutive_missing = 0
        max_consecutive = 0
        
        for is_missing in missing_indices:
            if is_missing:
                consecutive_missing += 1
                max_consecutive = max(max_consecutive, consecutive_missing)
            else:
                consecutive_missing = 0
        
        if max_consecutive > len(series) * 0.2:
            return "systematic_blocks"
        
        return "mixed_pattern"
    
    def _assess_geographic_accuracy(self, df: pd.DataFrame) -> Dict:
        """Assess geographic coordinate accuracy"""
        
        lat_col = "latitude"
        lon_col = "longitude"
        
        if lat_col not in df.columns or lon_col not in df.columns:
            return {"accuracy_score": 0, "error": "Missing coordinate columns"}
        
        # Valid coordinate ranges
        valid_lat = df[lat_col].between(-90, 90)
        valid_lon = df[lon_col].between(-180, 180)
        
        valid_coords = valid_lat & valid_lon
        accuracy_score = valid_coords.sum() / len(df)
        
        # Check for specific region (DRC)
        drc_lat_range = df[lat_col].between(-13.5, 5.5)
        drc_lon_range = df[lon_col].between(12, 32)
        in_drc = drc_lat_range & drc_lon_range
        
        return {
            "accuracy_score": accuracy_score,
            "valid_coordinates": int(valid_coords.sum()),
            "invalid_coordinates": int((~valid_coords).sum()),
            "in_drc_region": int(in_drc.sum()),
            "outside_drc_region": int((~in_drc).sum())
        }
    
    def _assess_temporal_accuracy(self, df: pd.DataFrame) -> Dict:
        """Assess temporal data accuracy"""
        
        date_col = "event_date"
        
        if date_col not in df.columns:
            return {"accuracy_score": 0, "error": "Missing date column"}
        
        # Convert to datetime
        try:
            dates = pd.to_datetime(df[date_col])
        except:
            return {"accuracy_score": 0, "error": "Invalid date format"}
        
        # Check for future dates
        current_time = datetime.now()
        future_dates = dates > current_time
        
        # Check for very old dates (before 1990)
        very_old_dates = dates < pd.Timestamp("1990-01-01")
        
        # Valid date range
        valid_dates = ~(future_dates | very_old_dates)
        accuracy_score = valid_dates.sum() / len(dates)
        
        return {
            "accuracy_score": accuracy_score,
            "valid_dates": int(valid_dates.sum()),
            "future_dates": int(future_dates.sum()),
            "very_old_dates": int(very_old_dates.sum()),
            "date_range": f"{dates.min().isoformat()} to {dates.max().isoformat()}"
        }
    
    def _assess_numeric_accuracy(self, df: pd.DataFrame, column: str) -> Dict:
        """Assess numeric column accuracy"""
        
        if column not in df.columns:
            return {"accuracy_score": 0, "error": f"Missing {column} column"}
        
        series = df[column]
        
        # Check for negative values where inappropriate
        if column == "fatalities":
            non_negative = series >= 0
            accuracy_score = non_negative.sum() / len(series)
            
            return {
                "accuracy_score": accuracy_score,
                "non_negative_count": int(non_negative.sum()),
                "negative_count": int((~non_negative).sum()),
                "max_value": float(series.max()),
                "mean_value": float(series.mean())
            }
        
        # General numeric accuracy
        valid_numeric = pd.to_numeric(series, errors='coerce').notna()
        accuracy_score = valid_numeric.sum() / len(series)
        
        return {
            "accuracy_score": accuracy_score,
            "valid_numeric": int(valid_numeric.sum()),
            "invalid_numeric": int((~valid_numeric).sum())
        }
    
    def _assess_categorical_accuracy(self, df: pd.DataFrame, column: str) -> Dict:
        """Assess categorical column accuracy"""
        
        if column not in df.columns:
            return {"accuracy_score": 0, "error": f"Missing {column} column"}
        
        series = df[column]
        
        # Check for empty values
        non_empty = series.notna() & (series != "") & (series != "Unknown")
        accuracy_score = non_empty.sum() / len(series)
        
        # Check for consistent categorization
        value_counts = series.value_counts()
        unique_values = series.nunique()
        
        return {
            "accuracy_score": accuracy_score,
            "non_empty_count": int(non_empty.sum()),
            "empty_count": int((~non_empty).sum()),
            "unique_values": unique_values,
            "most_common": value_counts.index[0] if len(value_counts) > 0 else None,
            "value_distribution": value_counts.head(10).to_dict()
        }
    
    def _grade_completeness(self, score: float) -> str:
        """Grade completeness score"""
        if score >= 0.95:
            return "excellent"
        elif score >= 0.85:
            return "good"
        elif score >= 0.70:
            return "fair"
        else:
            return "poor"
    
    def _grade_accuracy(self, score: float) -> str:
        """Grade accuracy score"""
        if score >= 0.95:
            return "excellent"
        elif score >= 0.85:
            return "good"
        elif score >= 0.70:
            return "fair"
        else:
            return "poor"
    
    def _grade_consistency(self, score: float) -> str:
        """Grade consistency score"""
        if score >= 0.90:
            return "excellent"
        elif score >= 0.80:
            return "good"
        elif score >= 0.65:
            return "fair"
        else:
            return "poor"
    
    def _grade_validity(self, score: float) -> str:
        """Grade validity score"""
        if score >= 0.95:
            return "excellent"
        elif score >= 0.85:
            return "good"
        elif score >= 0.70:
            return "fair"
        else:
            return "poor"
    
    def _grade_uniqueness(self, score: float) -> str:
        """Grade uniqueness score"""
        if score >= 0.95:
            return "excellent"
        elif score >= 0.85:
            return "good"
        elif score >= 0.70:
            return "fair"
        else:
            return "poor"
    
    def _grade_timeliness(self, score: float) -> str:
        """Grade timeliness score"""
        if score >= 0.80:
            return "excellent"
        elif score >= 0.60:
            return "good"
        elif score >= 0.40:
            return "fair"
        else:
            return "poor"
    
    def _grade_overall_quality(self, score: float) -> str:
        """Grade overall quality score"""
        if score >= 0.90:
            return "excellent"
        elif score >= 0.80:
            return "good"
        elif score >= 0.65:
            return "fair"
        else:
            return "poor"
    
    def _get_default_validation_rules(self) -> Dict:
        """Get default validation rules"""
        return {
            "required_fields": {
                "type": "presence",
                "fields": ["event_date", "location", "event_type"],
                "description": "Required fields must be present"
            },
            "valid_coordinates": {
                "type": "range",
                "field": "latitude",
                "min": -90,
                "max": 90,
                "description": "Latitude must be valid"
            },
            "non_negative_fatalities": {
                "type": "range",
                "field": "fatalities",
                "min": 0,
                "description": "Fatalities cannot be negative"
            },
            "valid_date_range": {
                "type": "date_range",
                "field": "event_date",
                "min": "1990-01-01",
                "max": "current_date",
                "description": "Event date must be reasonable"
            }
        }
    
    def _apply_validation_rule(self, df: pd.DataFrame, rule_name: str, rule_config: Dict) -> Dict:
        """Apply a single validation rule"""
        
        rule_type = rule_config.get("type")
        
        if rule_type == "presence":
            return self._validate_presence(df, rule_config)
        elif rule_type == "range":
            return self._validate_range(df, rule_config)
        elif rule_type == "date_range":
            return self._validate_date_range(df, rule_config)
        else:
            return {
                "status": "error",
                "message": f"Unknown rule type: {rule_type}"
            }
    
    def _validate_presence(self, df: pd.DataFrame, rule_config: Dict) -> Dict:
        """Validate field presence"""
        
        required_fields = rule_config.get("fields", [])
        missing_fields = []
        
        for field in required_fields:
            if field not in df.columns:
                missing_fields.append(field)
            else:
                # Check for null values
                null_count = df[field].isnull().sum()
                if null_count > 0:
                    missing_fields.append(f"{field} (has {null_count} null values)")
        
        status = "passed" if len(missing_fields) == 0 else "failed"
        
        return {
            "status": status,
            "missing_fields": missing_fields,
            "total_required": len(required_fields),
            "total_missing": len(missing_fields)
        }
    
    def _validate_range(self, df: pd.DataFrame, rule_config: Dict) -> Dict:
        """Validate field range"""
        
        field = rule_config.get("field")
        min_val = rule_config.get("min")
        max_val = rule_config.get("max")
        
        if field not in df.columns:
            return {
                "status": "error",
                "message": f"Field {field} not found"
            }
        
        series = df[field]
        
        # Convert to numeric if needed
        try:
            numeric_series = pd.to_numeric(series, errors='coerce')
        except:
            return {
                "status": "error",
                "message": f"Field {field} cannot be converted to numeric"
            }
        
        # Check range
        if min_val is not None and max_val is not None:
            in_range = numeric_series.between(min_val, max_val)
        elif min_val is not None:
            in_range = numeric_series >= min_val
        elif max_val is not None:
            in_range = numeric_series <= max_val
        else:
            return {
                "status": "error",
                "message": "No range bounds specified"
            }
        
        valid_count = in_range.sum()
        total_count = len(numeric_series)
        
        status = "passed" if valid_count == total_count else "failed"
        
        return {
            "status": status,
            "valid_count": int(valid_count),
            "invalid_count": int(total_count - valid_count),
            "total_count": int(total_count),
            "min_value": float(numeric_series.min()) if not numeric_series.empty else None,
            "max_value": float(numeric_series.max()) if not numeric_series.empty else None
        }
    
    def _validate_date_range(self, df: pd.DataFrame, rule_config: Dict) -> Dict:
        """Validate date range"""
        
        field = rule_config.get("field")
        min_date = rule_config.get("min")
        max_date = rule_config.get("max", "current_date")
        
        if field not in df.columns:
            return {
                "status": "error",
                "message": f"Field {field} not found"
            }
        
        try:
            dates = pd.to_datetime(df[field])
        except:
            return {
                "status": "error",
                "message": f"Field {field} cannot be converted to datetime"
            }
        
        # Parse date bounds
        if min_date:
            min_dt = pd.to_datetime(min_date)
        else:
            min_dt = None
        
        if max_date == "current_date":
            max_dt = datetime.now()
        elif max_date:
            max_dt = pd.to_datetime(max_date)
        else:
            max_dt = None
        
        # Check date range
        valid_dates = pd.Series([True] * len(dates))
        
        if min_dt is not None:
            valid_dates = valid_dates & (dates >= min_dt)
        
        if max_dt is not None:
            valid_dates = valid_dates & (dates <= max_dt)
        
        valid_count = valid_dates.sum()
        total_count = len(dates)
        
        status = "passed" if valid_count == total_count else "failed"
        
        return {
            "status": status,
            "valid_count": int(valid_count),
            "invalid_count": int(total_count - valid_count),
            "total_count": int(total_count),
            "min_date": dates.min().isoformat() if not dates.empty else None,
            "max_date": dates.max().isoformat() if not dates.empty else None
        }
    
    def _detect_missing_data_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """Detect missing data anomalies"""
        
        anomalies = []
        
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            missing_rate = missing_count / len(df)
            
            # High missing rate
            if missing_rate > 0.5:
                anomalies.append({
                    "type": "high_missing_rate",
                    "column": column,
                    "missing_rate": missing_rate,
                    "severity": "high"
                })
            elif missing_rate > 0.2:
                anomalies.append({
                    "type": "moderate_missing_rate",
                    "column": column,
                    "missing_rate": missing_rate,
                    "severity": "medium"
                })
        
        return anomalies
    
    def _detect_duplicate_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """Detect duplicate anomalies"""
        
        anomalies = []
        
        # Check for completely duplicate rows
        duplicate_rows = df.duplicated().sum()
        if duplicate_rows > 0:
            anomalies.append({
                "type": "duplicate_rows",
                "count": duplicate_rows,
                "severity": "medium"
            })
        
        # Check for duplicate key fields
        if "event_id" in df.columns:
            duplicate_ids = df["event_id"].duplicated().sum()
            if duplicate_ids > 0:
                anomalies.append({
                    "type": "duplicate_event_ids",
                    "count": duplicate_ids,
                    "severity": "high"
                })
        
        return anomalies
    
    def _detect_outlier_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """Detect outlier anomalies"""
        
        anomalies = []
        
        # Check numeric columns for outliers
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            series = df[column].dropna()
            
            if len(series) > 10:  # Only check if enough data
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = series[(series < lower_bound) | (series > upper_bound)]
                
                if len(outliers) > 0:
                    outlier_rate = len(outliers) / len(series)
                    anomalies.append({
                        "type": "statistical_outliers",
                        "column": column,
                        "count": len(outliers),
                        "rate": outlier_rate,
                        "severity": "high" if outlier_rate > 0.1 else "medium"
                    })
        
        return anomalies
    
    def _detect_format_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """Detect format anomalies"""
        
        anomalies = []
        
        # Check date formats
        if "event_date" in df.columns:
            try:
                pd.to_datetime(df["event_date"])
            except:
                anomalies.append({
                    "type": "invalid_date_format",
                    "column": "event_date",
                    "severity": "high"
                })
        
        # Check coordinate formats
        if "latitude" in df.columns and "longitude" in df.columns:
            invalid_lat = ~df["latitude"].between(-90, 90)
            invalid_lon = ~df["longitude"].between(-180, 180)
            
            if invalid_lat.any():
                anomalies.append({
                    "type": "invalid_latitude",
                    "count": invalid_lat.sum(),
                    "severity": "high"
                })
            
            if invalid_lon.any():
                anomalies.append({
                    "type": "invalid_longitude",
                    "count": invalid_lon.sum(),
                    "severity": "high"
                })
        
        return anomalies
    
    def _detect_logic_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """Detect logic anomalies"""
        
        anomalies = []
        
        # Check for negative fatalities
        if "fatalities" in df.columns:
            negative_fatalities = df["fatalities"] < 0
            if negative_fatalities.any():
                anomalies.append({
                    "type": "negative_fatalities",
                    "count": negative_fatalities.sum(),
                    "severity": "high"
                })
        
        # Check for events with no location
        if "location" in df.columns:
            missing_location = df["location"].isna() | (df["location"] == "")
            if missing_location.any():
                anomalies.append({
                    "type": "missing_location",
                    "count": missing_location.sum(),
                    "severity": "medium"
                })
        
        return anomalies
    
    def _analyze_column_relationships(self, df: pd.DataFrame) -> Dict:
        """Analyze relationships between columns"""
        
        relationships = {}
        
        # Correlation analysis for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) > 1:
            correlation_matrix = df[numeric_columns].corr()
            relationships["correlations"] = correlation_matrix.to_dict()
        
        return relationships
    
    def _identify_data_patterns(self, df: pd.DataFrame) -> Dict:
        """Identify patterns in data"""
        
        patterns = {}
        
        # Temporal patterns
        if "event_date" in df.columns:
            dates = pd.to_datetime(df["event_date"])
            patterns["temporal"] = {
                "events_by_month": dates.dt.month.value_counts().to_dict(),
                "events_by_day_of_week": dates.dt.dayofweek.value_counts().to_dict(),
                "events_by_year": dates.dt.year.value_counts().to_dict()
            }
        
        # Geographic patterns
        if "location" in df.columns:
            patterns["geographic"] = {
                "events_by_location": df["location"].value_counts().head(20).to_dict()
            }
        
        return patterns
    
    def _analyze_quality_trends(self, quality_trends: Dict) -> Dict:
        """Analyze trends in quality metrics"""
        
        trend_analysis = {}
        
        for metric_category, metrics in quality_trends["metrics"].items():
            trend_analysis[metric_category] = {}
            
            for metric_name, values in metrics.items():
                if len(values) > 1:
                    # Calculate trend direction
                    recent_avg = np.mean(values[-3:])  # Last 3 periods
                    earlier_avg = np.mean(values[:3])   # First 3 periods
                    
                    if recent_avg > earlier_avg * 1.1:
                        trend_direction = "improving"
                    elif recent_avg < earlier_avg * 0.9:
                        trend_direction = "declining"
                    else:
                        trend_direction = "stable"
                    
                    trend_analysis[metric_category][metric_name] = {
                        "direction": trend_direction,
                        "recent_average": float(recent_avg),
                        "earlier_average": float(earlier_avg),
                        "change_percent": float((recent_avg - earlier_avg) / earlier_avg * 100) if earlier_avg != 0 else 0
                    }
        
        return trend_analysis