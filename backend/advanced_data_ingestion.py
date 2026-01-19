import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Optional
import json
from datetime import datetime, timedelta
import re

class AdvancedDataIngestion:
    """Enhanced data ingestion from multiple sources"""
    
    def __init__(self):
        self.acled_api_key = None
        self.twitter_api_key = None
        self.satellite_api_key = None
        self.base_urls = {
            "acled": "https://api.acleddata.com/v2",
            "twitter": "https://api.twitter.com/2",
            "sentinel": "https://scihub.copernicus.eu/dhus"
        }
    
    def fetch_social_media_data(self, 
                               keywords: List[str] = ["conflict", "violence", "protest"],
                               location: str = "DR Congo",
                               days_back: int = 7) -> pd.DataFrame:
        """Fetch conflict-related social media data"""
        
        # Sample social media data for testing
        sample_data = []
        base_date = datetime.now() - timedelta(days=days_back)
        
        for i in range(50):  # Generate 50 sample posts
            post_date = base_date + timedelta(hours=i*3)
            keyword = keywords[i % len(keywords)]
            
            sample_data.append({
                "post_id": f"post_{i}",
                "timestamp": post_date.isoformat(),
                "platform": "twitter",
                "content": f"Reports of {keyword} near {location}. Situation developing.",
                "user_location": location,
                "likes": np.random.randint(10, 1000),
                "retweets": np.random.randint(5, 500),
                "sentiment_score": np.random.uniform(-1, 1),
                "confidence": np.random.uniform(0.7, 1.0),
                "verified": np.random.choice([True, False], p=[0.3, 0.7])
            })
        
        df = pd.DataFrame(sample_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        return df
    
    def fetch_satellite_imagery_data(self,
                                    region: str = "eastern_drc",
                                    days_back: int = 30) -> Dict:
        """Fetch satellite imagery analysis for conflict indicators"""
        
        # Sample satellite analysis data
        sample_analysis = {
            "region": region,
            "analysis_period": f"{days_back} days",
            "imagery_dates": [
                (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") 
                for i in range(days_back, 0, -5)
            ],
            "indicators": {
                "new_settlements": np.random.randint(5, 25),
                "destroyed_infrastructure": np.random.randint(2, 15),
                "military_vehicle_movement": np.random.randint(10, 50),
                "fire_incidents": np.random.randint(3, 20),
                "population_displacement": np.random.randint(1000, 10000)
            },
            "confidence_score": np.random.uniform(0.75, 0.95),
            "high_risk_areas": [
                {"lat": -1.6833 + np.random.uniform(-0.5, 0.5), 
                 "lon": 29.2333 + np.random.uniform(-0.5, 0.5),
                 "risk_type": "military_activity"},
                {"lat": -2.5167 + np.random.uniform(-0.5, 0.5), 
                 "lon": 28.8667 + np.random.uniform(-0.5, 0.5),
                 "risk_type": "displacement"}
            ],
            "change_detection": {
                "significant_changes": np.random.randint(5, 30),
                "change_types": ["construction", "destruction", "movement", "fire"],
                "analysis_confidence": np.random.uniform(0.8, 0.95)
            }
        }
        
        return sample_analysis
    
    def fetch_economic_indicators(self,
                                country: str = "DR Congo",
                                indicators: List[str] = None) -> pd.DataFrame:
        """Fetch economic indicators that may correlate with conflict"""
        
        if indicators is None:
            indicators = ["gdp_growth", "inflation", "unemployment", "food_price_index"]
        
        # Sample economic data
        dates = pd.date_range(start="2020-01-01", end=datetime.now(), freq="M")
        economic_data = []
        
        for date in dates:
            row = {"date": date}
            for indicator in indicators:
                # Generate realistic economic data with some volatility
                if indicator == "gdp_growth":
                    value = np.random.normal(3.5, 2.0)  # GDP growth around 3.5%
                elif indicator == "inflation":
                    value = np.random.normal(15, 5)  # Inflation around 15%
                elif indicator == "unemployment":
                    value = np.random.normal(12, 3)  # Unemployment around 12%
                elif indicator == "food_price_index":
                    value = np.random.normal(100, 20)  # Food price index
                else:
                    value = np.random.normal(50, 10)
                
                row[indicator] = max(0, value)  # Ensure non-negative
            
            economic_data.append(row)
        
        return pd.DataFrame(economic_data)
    
    def fetch_climate_data(self,
                          region: str = "eastern_drc",
                          days_back: int = 365) -> pd.DataFrame:
        """Fetch climate data that may influence conflict patterns"""
        
        # Sample climate data
        dates = pd.date_range(start=datetime.now() - timedelta(days=days_back), 
                             end=datetime.now(), freq="D")
        climate_data = []
        
        for date in dates:
            # Generate realistic climate data with seasonal patterns
            day_of_year = date.timetuple().tm_yday
            seasonal_temp = 25 + 10 * np.sin(2 * np.pi * day_of_year / 365)  # Seasonal temperature variation
            
            climate_data.append({
                "date": date,
                "temperature": seasonal_temp + np.random.normal(0, 3),
                "precipitation": max(0, np.random.exponential(5)),  # Exponential distribution for rainfall
                "humidity": np.random.uniform(60, 90),
                "drought_index": np.random.uniform(0, 1),
                "vegetation_index": np.random.uniform(0.3, 0.8)
            })
        
        return pd.DataFrame(climate_data)
    
    def analyze_sentiment_trends(self, social_media_df: pd.DataFrame) -> Dict:
        """Analyze sentiment trends from social media data"""
        
        if social_media_df.empty:
            return {"error": "No social media data available"}
        
        # Daily sentiment analysis
        daily_sentiment = social_media_df.groupby(
            social_media_df["timestamp"].dt.date
        ).agg({
            "sentiment_score": ["mean", "std", "count"],
            "likes": "sum",
            "retweets": "sum"
        }).reset_index()
        
        # Flatten column names
        daily_sentiment.columns = ["date", "sentiment_mean", "sentiment_std", 
                                  "post_count", "total_likes", "total_retweets"]
        
        # Sentiment trend analysis
        sentiment_trend = {
            "overall_sentiment": daily_sentiment["sentiment_mean"].mean(),
            "sentiment_volatility": daily_sentiment["sentiment_std"].mean(),
            "total_posts": len(social_media_df),
            "engagement_rate": (social_media_df["likes"].sum() + social_media_df["retweets"].sum()) / len(social_media_df),
            "verified_posts_ratio": social_media_df["verified"].mean(),
            "daily_sentiment": daily_sentiment.to_dict("records")
        }
        
        # Detect sentiment anomalies
        sentiment_mean = daily_sentiment["sentiment_mean"].mean()
        sentiment_std = daily_sentiment["sentiment_mean"].std()
        
        anomaly_threshold = sentiment_mean - 2 * sentiment_std  # 2 standard deviations below mean
        negative_sentiment_days = daily_sentiment[
            daily_sentiment["sentiment_mean"] < anomaly_threshold
        ]
        
        sentiment_trend["sentiment_anomalies"] = {
            "threshold": anomaly_threshold,
            "negative_days": len(negative_sentiment_days),
            "anomaly_dates": negative_sentiment_days["date"].dt.strftime("%Y-%m-%d").tolist()
        }
        
        return sentiment_trend
    
    def correlate_conflict_drivers(self, 
                                 conflict_df: pd.DataFrame,
                                 economic_df: pd.DataFrame,
                                 climate_df: pd.DataFrame,
                                 social_media_df: pd.DataFrame) -> Dict:
        """Correlate conflict events with potential drivers"""
        
        correlations = {}
        
        # Prepare conflict data (daily aggregation)
        conflict_daily = conflict_df.groupby(
            conflict_df["event_date"].dt.date
        ).agg({
            "event_id": "count",
            "fatalities": "sum"
        }).reset_index()
        conflict_daily.columns = ["date", "conflict_events", "conflict_fatalities"]
        conflict_daily["date"] = pd.to_datetime(conflict_daily["date"])
        
        # Economic correlations
        if not economic_df.empty:
            economic_daily = economic_df.set_index("date").resample("D").ffill().reset_index()
            
            for indicator in ["gdp_growth", "inflation", "unemployment", "food_price_index"]:
                if indicator in economic_daily.columns:
                    merged = conflict_daily.merge(economic_daily[["date", indicator]], on="date", how="inner")
                    if len(merged) > 10:
                        correlation = merged["conflict_events"].corr(merged[indicator])
                        correlations[f"economic_{indicator}"] = {
                            "correlation": correlation,
                            "strength": "strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.3 else "weak",
                            "direction": "positive" if correlation > 0 else "negative"
                        }
        
        # Climate correlations
        if not climate_df.empty:
            for climate_var in ["temperature", "precipitation", "drought_index"]:
                if climate_var in climate_df.columns:
                    merged = conflict_daily.merge(climate_df[["date", climate_var]], on="date", how="inner")
                    if len(merged) > 10:
                        correlation = merged["conflict_events"].corr(merged[climate_var])
                        correlations[f"climate_{climate_var}"] = {
                            "correlation": correlation,
                            "strength": "strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.3 else "weak",
                            "direction": "positive" if correlation > 0 else "negative"
                        }
        
        # Social media correlations
        if not social_media_df.empty:
            social_daily = social_media_df.groupby(
                social_media_df["timestamp"].dt.date
            ).agg({
                "sentiment_score": "mean",
                "post_id": "count"
            }).reset_index()
            social_daily.columns = ["date", "sentiment", "social_posts"]
            social_daily["date"] = pd.to_datetime(social_daily["date"])
            
            merged = conflict_daily.merge(social_daily, on="date", how="inner")
            if len(merged) > 10:
                sentiment_correlation = merged["conflict_events"].corr(merged["sentiment"])
                posts_correlation = merged["conflict_events"].corr(merged["social_posts"])
                
                correlations["social_sentiment"] = {
                    "correlation": sentiment_correlation,
                    "strength": "strong" if abs(sentiment_correlation) > 0.7 else "moderate" if abs(sentiment_correlation) > 0.3 else "weak",
                    "direction": "positive" if sentiment_correlation > 0 else "negative"
                }
                
                correlations["social_volume"] = {
                    "correlation": posts_correlation,
                    "strength": "strong" if abs(posts_correlation) > 0.7 else "moderate" if abs(posts_correlation) > 0.3 else "weak",
                    "direction": "positive" if posts_correlation > 0 else "negative"
                }
        
        # Identify strongest correlations
        sorted_correlations = sorted(
            [(k, v) for k, v in correlations.items() if "correlation" in v],
            key=lambda x: abs(x[1]["correlation"]),
            reverse=True
        )
        
        return {
            "correlations": correlations,
            "strongest_correlations": sorted_correlations[:5],
            "analysis_summary": {
                "total_factors_analyzed": len(correlations),
                "significant_correlations": len([c for c in correlations.values() if abs(c.get("correlation", 0)) > 0.3]),
                "strong_correlations": len([c for c in correlations.values() if abs(c.get("correlation", 0)) > 0.7])
            }
        }
    
    def generate_multi_source_report(self, 
                                   conflict_df: pd.DataFrame,
                                   economic_df: pd.DataFrame = None,
                                   climate_df: pd.DataFrame = None,
                                   social_media_df: pd.DataFrame = None,
                                   satellite_data: Dict = None) -> Dict:
        """Generate comprehensive multi-source analysis report"""
        
        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "data_sources": ["conflict_events"],
                "analysis_period": f"{conflict_df['event_date'].min().strftime('%Y-%m-%d')} to {conflict_df['event_date'].max().strftime('%Y-%m-%d')}"
            }
        }
        
        # Basic conflict statistics
        report["conflict_summary"] = {
            "total_events": len(conflict_df),
            "total_fatalities": int(conflict_df["fatalities"].sum()),
            "unique_locations": conflict_df["location"].nunique(),
            "unique_actors": conflict_df["actor1"].nunique(),
            "event_types": conflict_df["event_type"].value_counts().to_dict()
        }
        
        # Economic analysis
        if economic_df is not None and not economic_df.empty:
            report["economic_analysis"] = {
                "data_points": len(economic_df),
                "indicators_available": list(economic_df.columns),
                "latest_values": economic_df.iloc[-1].to_dict()
            }
            report["report_metadata"]["data_sources"].append("economic_indicators")
        
        # Climate analysis
        if climate_df is not None and not climate_df.empty:
            report["climate_analysis"] = {
                "data_points": len(climate_df),
                "variables_available": list(climate_df.columns),
                "average_conditions": climate_df.mean().to_dict()
            }
            report["report_metadata"]["data_sources"].append("climate_data")
        
        # Social media analysis
        if social_media_df is not None and not social_media_df.empty:
            sentiment_analysis = self.analyze_sentiment_trends(social_media_df)
            report["social_media_analysis"] = sentiment_analysis
            report["report_metadata"]["data_sources"].append("social_media")
        
        # Satellite analysis
        if satellite_data is not None:
            report["satellite_analysis"] = satellite_data
            report["report_metadata"]["data_sources"].append("satellite_imagery")
        
        # Correlation analysis
        correlation_analysis = self.correlate_conflict_drivers(
            conflict_df, economic_df, climate_df, social_media_df
        )
        report["correlation_analysis"] = correlation_analysis
        
        # Key insights
        report["key_insights"] = self._generate_key_insights(report)
        
        return report
    
    def _generate_key_insights(self, report: Dict) -> List[str]:
        """Generate key insights from multi-source analysis"""
        insights = []
        
        # Conflict insights
        conflict_summary = report.get("conflict_summary", {})
        if conflict_summary.get("total_fatalities", 0) > 1000:
            insights.append("High fatality count indicates severe conflict intensity")
        
        if conflict_summary.get("unique_locations", 0) > 50:
            insights.append("Conflict is geographically widespread across multiple locations")
        
        # Economic insights
        economic_analysis = report.get("economic_analysis", {})
        if economic_analysis:
            insights.append("Economic factors available for correlation analysis")
        
        # Social media insights
        social_analysis = report.get("social_media_analysis", {})
        if social_analysis:
            overall_sentiment = social_analysis.get("overall_sentiment", 0)
            if overall_sentiment < -0.3:
                insights.append("Negative sentiment detected in social media discourse")
            elif overall_sentiment > 0.3:
                insights.append("Positive sentiment detected in social media discourse")
        
        # Correlation insights
        correlation_analysis = report.get("correlation_analysis", {})
        strongest = correlation_analysis.get("strongest_correlations", [])
        if strongest and abs(strongest[0][1].get("correlation", 0)) > 0.5:
            factor = strongest[0][0]
            strength = strongest[0][1].get("strength", "moderate")
            insights.append(f"Strong {strength} correlation found between conflict and {factor}")
        
        # Satellite insights
        satellite_analysis = report.get("satellite_analysis", {})
        if satellite_analysis:
            indicators = satellite_analysis.get("indicators", {})
            if indicators.get("military_vehicle_movement", 0) > 30:
                insights.append("High military vehicle movement detected in satellite imagery")
            if indicators.get("population_displacement", 0) > 5000:
                insights.append("Significant population displacement observed")
        
        return insights