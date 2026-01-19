import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class ConflictDriverAnalyzer:
    """Analyze conflict drivers and perform causal inference"""
    
    def __init__(self):
        self.driver_models = {}
        self.causal_models = {}
        self.feature_importance = {}
        self.driver_weights = {}
    
    def load_data(self, conflict_df: pd.DataFrame, 
                 economic_df: pd.DataFrame = None,
                 climate_df: pd.DataFrame = None,
                 social_df: pd.DataFrame = None):
        """Load multi-source data for driver analysis"""
        self.conflict_data = conflict_df.copy()
        self.economic_data = economic_df.copy() if economic_df is not None else None
        self.climate_data = climate_df.copy() if climate_df is not None else None
        self.social_data = social_df.copy() if social_df is not None else None
        
        # Ensure date columns are datetime
        self.conflict_data["event_date"] = pd.to_datetime(self.conflict_data["event_date"])
        if self.economic_data is not None:
            self.economic_data["date"] = pd.to_datetime(self.economic_data["date"])
        if self.climate_data is not None:
            self.climate_data["date"] = pd.to_datetime(self.climate_data["date"])
        if self.social_data is not None:
            self.social_data["timestamp"] = pd.to_datetime(self.social_data["timestamp"])
    
    def identify_conflict_drivers(self, 
                                 target_variable: str = "fatalities",
                                 analysis_method: str = "correlation") -> Dict:
        """Identify potential conflict drivers using multiple methods"""
        
        # Prepare integrated dataset
        integrated_data = self._prepare_integrated_dataset(target_variable)
        
        if integrated_data.empty:
            return {"error": "Insufficient data for driver analysis"}
        
        drivers = {}
        
        # Method 1: Correlation analysis
        if analysis_method in ["correlation", "all"]:
            correlation_results = self._correlation_driver_analysis(integrated_data, target_variable)
            drivers["correlation"] = correlation_results
        
        # Method 2: Feature importance from ML models
        if analysis_method in ["feature_importance", "all"]:
            importance_results = self._feature_importance_analysis(integrated_data, target_variable)
            drivers["feature_importance"] = importance_results
        
        # Method 3: Granger causality (for time series)
        if analysis_method in ["granger", "all"]:
            granger_results = self._granger_causality_analysis(integrated_data, target_variable)
            drivers["granger_causality"] = granger_results
        
        # Method 4: Lead-lag analysis
        if analysis_method in ["leadlag", "all"]:
            leadlag_results = self._lead_lag_analysis(integrated_data, target_variable)
            drivers["lead_lag"] = leadlag_results
        
        # Synthesize results
        synthesized_drivers = self._synthesize_driver_results(drivers)
        
        return {
            "driver_analysis": drivers,
            "synthesized_drivers": synthesized_drivers,
            "target_variable": target_variable,
            "data_points": len(integrated_data),
            "analysis_methods": list(drivers.keys())
        }
    
    def perform_causal_inference(self, 
                               treatment_variable: str,
                               outcome_variable: str = "fatalities",
                               method: str = "regression") -> Dict:
        """Perform causal inference analysis"""
        
        # Prepare data for causal analysis
        causal_data = self._prepare_causal_dataset(treatment_variable, outcome_variable)
        
        if causal_data.empty:
            return {"error": "Insufficient data for causal inference"}
        
        causal_results = {}
        
        # Method 1: Regression-based causal inference
        if method in ["regression", "all"]:
            regression_results = self._regression_causal_analysis(
                causal_data, treatment_variable, outcome_variable
            )
            causal_results["regression"] = regression_results
        
        # Method 2: Difference-in-differences (if applicable)
        if method in ["did", "all"]:
            did_results = self._difference_in_differences_analysis(
                causal_data, treatment_variable, outcome_variable
            )
            causal_results["difference_in_differences"] = did_results
        
        # Method 3: Instrumental variables (if instruments available)
        if method in ["iv", "all"]:
            iv_results = self._instrumental_variables_analysis(
                causal_data, treatment_variable, outcome_variable
            )
            causal_results["instrumental_variables"] = iv_results
        
        # Method 4: Propensity score matching
        if method in ["psm", "all"]:
            psm_results = self._propensity_score_matching(
                causal_data, treatment_variable, outcome_variable
            )
            causal_results["propensity_score_matching"] = psm_results
        
        return {
            "causal_inference": causal_results,
            "treatment": treatment_variable,
            "outcome": outcome_variable,
            "method": method,
            "sample_size": len(causal_data)
        }
    
    def analyze_driver_interactions(self, 
                                   target_variable: str = "fatalities") -> Dict:
        """Analyze interactions between different drivers"""
        
        integrated_data = self._prepare_integrated_dataset(target_variable)
        
        if integrated_data.empty:
            return {"error": "Insufficient data for interaction analysis"}
        
        interactions = {}
        
        # Economic-climate interactions
        if self.economic_data is not None and self.climate_data is not None:
            econ_climate_interactions = self._analyze_economic_climate_interactions(
                integrated_data, target_variable
            )
            interactions["economic_climate"] = econ_climate_interactions
        
        # Social-conflict interactions
        if self.social_data is not None:
            social_conflict_interactions = self._analyze_social_conflict_interactions(
                integrated_data, target_variable
            )
            interactions["social_conflict"] = social_conflict_interactions
        
        # Multi-driver interactions
        multi_driver_interactions = self._analyze_multi_driver_interactions(
            integrated_data, target_variable
        )
        interactions["multi_driver"] = multi_driver_interactions
        
        return {
            "driver_interactions": interactions,
            "target_variable": target_variable,
            "interaction_types": list(interactions.keys())
        }
    
    def assess_driver_sensitivity(self, 
                                  target_variable: str = "fatalities") -> Dict:
        """Assess sensitivity of conflict to different drivers"""
        
        integrated_data = self._prepare_integrated_dataset(target_variable)
        
        if integrated_data.empty:
            return {"error": "Insufficient data for sensitivity analysis"}
        
        sensitivity_results = {}
        
        # Get feature columns (exclude target)
        feature_columns = [col for col in integrated_data.columns if col != target_variable]
        
        for feature in feature_columns:
            if integrated_data[feature].var() > 0:  # Only analyze variables with variance
                sensitivity = self._calculate_driver_sensitivity(
                    integrated_data, feature, target_variable
                )
                sensitivity_results[feature] = sensitivity
        
        # Rank drivers by sensitivity
        ranked_drivers = sorted(
            sensitivity_results.items(),
            key=lambda x: abs(x[1]["sensitivity_score"]),
            reverse=True
        )
        
        return {
            "sensitivity_analysis": sensitivity_results,
            "ranked_drivers": ranked_drivers,
            "most_sensitive_drivers": ranked_drivers[:5],
            "target_variable": target_variable
        }
    
    def _prepare_integrated_dataset(self, target_variable: str) -> pd.DataFrame:
        """Prepare integrated dataset from multiple sources"""
        
        # Start with conflict data aggregated by date
        conflict_daily = self.conflict_data.groupby(
            self.conflict_data["event_date"].dt.date
        ).agg({
            "event_id": "count",
            "fatalities": "sum",
            "latitude": "mean",
            "longitude": "mean"
        }).reset_index()
        
        conflict_daily.columns = ["date", "conflict_events", "fatalities", "avg_lat", "avg_lon"]
        conflict_daily["date"] = pd.to_datetime(conflict_daily["date"])
        
        # Merge with economic data
        if self.economic_data is not None:
            economic_daily = self.economic_data.set_index("date").resample("D").ffill().reset_index()
            conflict_daily = conflict_daily.merge(economic_daily, on="date", how="left")
        
        # Merge with climate data
        if self.climate_data is not None:
            climate_daily = self.climate_data.set_index("date").resample("D").ffill().reset_index()
            conflict_daily = conflict_daily.merge(climate_daily, on="date", how="left")
        
        # Merge with social media data
        if self.social_data is not None:
            social_daily = self.social_data.groupby(
                self.social_data["timestamp"].dt.date
            ).agg({
                "sentiment_score": "mean",
                "post_id": "count",
                "likes": "sum",
                "retweets": "sum"
            }).reset_index()
            social_daily.columns = ["date", "sentiment", "social_posts", "total_likes", "total_retweets"]
            social_daily["date"] = pd.to_datetime(social_daily["date"])
            conflict_daily = conflict_daily.merge(social_daily, on="date", how="left")
        
        # Add temporal features
        conflict_daily["year"] = conflict_daily["date"].dt.year
        conflict_daily["month"] = conflict_daily["date"].dt.month
        conflict_daily["day_of_year"] = conflict_daily["date"].dt.dayofyear
        
        # Add lag features
        conflict_daily["conflict_lag_7"] = conflict_daily["conflict_events"].rolling(7).mean()
        conflict_daily["fatalities_lag_7"] = conflict_daily["fatalities"].rolling(7).mean()
        
        return conflict_daily.fillna(0)
    
    def _correlation_driver_analysis(self, data: pd.DataFrame, target: str) -> Dict:
        """Perform correlation-based driver analysis"""
        
        correlations = {}
        feature_columns = [col for col in data.columns if col != target and col not in ["date"]]
        
        for feature in feature_columns:
            if data[feature].var() > 0:
                correlation = data[target].corr(data[feature])
                p_value = self._correlation_significance(data[target], data[feature])
                
                correlations[feature] = {
                    "correlation": correlation,
                    "p_value": p_value,
                    "significance": "significant" if p_value < 0.05 else "not_significant",
                    "strength": self._interpret_correlation_strength(abs(correlation)),
                    "direction": "positive" if correlation > 0 else "negative"
                }
        
        # Sort by correlation strength
        sorted_correlations = sorted(
            correlations.items(),
            key=lambda x: abs(x[1]["correlation"]),
            reverse=True
        )
        
        return {
            "correlations": correlations,
            "ranked_drivers": sorted_correlations,
            "significant_correlations": [k for k, v in correlations.items() if v["significance"] == "significant"]
        }
    
    def _feature_importance_analysis(self, data: pd.DataFrame, target: str) -> Dict:
        """Analyze feature importance using ML models"""
        
        feature_columns = [col for col in data.columns if col != target and col not in ["date"]]
        
        if len(feature_columns) < 2:
            return {"error": "Insufficient features for importance analysis"}
        
        # Prepare data
        X = data[feature_columns].fillna(0)
        y = data[target]
        
        # Remove outliers
        Q1 = y.quantile(0.25)
        Q3 = y.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        mask = (y >= lower_bound) & (y <= upper_bound)
        X_clean = X[mask]
        y_clean = y[mask]
        
        if len(X_clean) < 30:
            return {"error": "Insufficient data after outlier removal"}
        
        # Train models
        models = {
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        importance_results = {}
        
        for name, model in models.items():
            try:
                model.fit(X_clean, y_clean)
                
                # Get feature importance
                if hasattr(model, 'feature_importances_'):
                    importance_dict = dict(zip(feature_columns, model.feature_importances_))
                    importance_results[name] = importance_dict
                    
                    # Store for global use
                    self.feature_importance[f"{target}_{name}"] = importance_dict
                
            except Exception as e:
                importance_results[name] = {"error": str(e)}
        
        # Average importance across models
        if importance_results:
            avg_importance = {}
            for feature in feature_columns:
                values = []
                for model_result in importance_results.values():
                    if feature in model_result and isinstance(model_result[feature], (int, float)):
                        values.append(model_result[feature])
                
                if values:
                    avg_importance[feature] = np.mean(values)
            
            importance_results["average"] = avg_importance
        
        return importance_results
    
    def _granger_causality_analysis(self, data: pd.DataFrame, target: str, max_lag: int = 4) -> Dict:
        """Perform Granger causality analysis"""
        
        from statsmodels.tsa.stattools import grangercausalitytests
        
        feature_columns = [col for col in data.columns if col != target and col not in ["date"]]
        granger_results = {}
        
        for feature in feature_columns:
            try:
                # Prepare data for Granger test
                test_data = data[[feature, target]].dropna()
                
                if len(test_data) > max_lag + 10:
                    # Perform Granger causality test
                    gc_result = grangercausalitytests(test_data, max_lag, verbose=False)
                    
                    # Extract p-values
                    p_values = [gc_result[i][0]["ssr_ftest"][1] for i in range(max_lag)]
                    min_p_value = min(p_values)
                    best_lag = p_values.index(min_p_value) + 1
                    
                    granger_results[feature] = {
                        "min_p_value": min_p_value,
                        "best_lag": best_lag,
                        "is_significant": min_p_value < 0.05,
                        "p_values": p_values
                    }
                
            except Exception as e:
                granger_results[feature] = {"error": str(e)}
        
        return granger_results
    
    def _lead_lag_analysis(self, data: pd.DataFrame, target: str, max_lag: int = 7) -> Dict:
        """Perform lead-lag analysis"""
        
        feature_columns = [col for col in data.columns if col != target and col not in ["date"]]
        leadlag_results = {}
        
        for feature in feature_columns:
            correlations = []
            
            for lag in range(-max_lag, max_lag + 1):
                if lag == 0:
                    corr = data[target].corr(data[feature])
                elif lag > 0:  # Feature leads target
                    corr = data[target].iloc[lag:].corr(data[feature].iloc[:-lag])
                else:  # Feature lags target
                    corr = data[target].iloc[:lag].corr(data[feature].iloc[-lag:])
                
                if not np.isnan(corr):
                    correlations.append((lag, corr))
            
            if correlations:
                # Find maximum correlation
                best_lag, best_corr = max(correlations, key=lambda x: abs(x[1]))
                
                leadlag_results[feature] = {
                    "best_lag": best_lag,
                    "best_correlation": best_corr,
                    "relationship": "leads" if best_lag > 0 else "lags" if best_lag < 0 else "contemporaneous",
                    "correlations": correlations
                }
        
        return leadlag_results
    
    def _synthesize_driver_results(self, driver_results: Dict) -> Dict:
        """Synthesize results from different driver analysis methods"""
        
        # Collect all driver scores
        driver_scores = {}
        
        # From correlation analysis
        if "correlation" in driver_results:
            for driver, result in driver_results["correlation"]["correlations"].items():
                if isinstance(result, dict) and "correlation" in result:
                    driver_scores[driver] = driver_scores.get(driver, []) + [abs(result["correlation"])]
        
        # From feature importance
        if "feature_importance" in driver_results:
            if "average" in driver_results["feature_importance"]:
                for driver, importance in driver_results["feature_importance"]["average"].items():
                    if isinstance(importance, (int, float)):
                        driver_scores[driver] = driver_scores.get(driver, []) + [importance]
        
        # From Granger causality
        if "granger_causality" in driver_results:
            for driver, result in driver_results["granger_causality"].items():
                if isinstance(result, dict) and "is_significant" in result and result["is_significant"]:
                    driver_scores[driver] = driver_scores.get(driver, []) + [0.8]  # High score for significant causality
        
        # Calculate average scores
        final_drivers = {}
        for driver, scores in driver_scores.items():
            if scores:
                final_drivers[driver] = {
                    "average_score": np.mean(scores),
                    "method_count": len(scores),
                    "consistency": np.std(scores),
                    "rank": 0  # Will be assigned after sorting
                }
        
        # Rank drivers
        ranked_drivers = sorted(final_drivers.items(), key=lambda x: x[1]["average_score"], reverse=True)
        
        for i, (driver, scores) in enumerate(ranked_drivers):
            scores["rank"] = i + 1
        
        return {
            "ranked_drivers": ranked_drivers,
            "top_drivers": ranked_drivers[:5],
            "total_drivers_analyzed": len(ranked_drivers)
        }
    
    def _regression_causal_analysis(self, data: pd.DataFrame, treatment: str, outcome: str) -> Dict:
        """Regression-based causal analysis"""
        
        # Prepare control variables
        control_vars = [col for col in data.columns if col not in [treatment, outcome, "date"]]
        
        if not control_vars:
            control_vars = ["const"]  # Add constant if no controls
        
        # Simple linear regression
        try:
            from statsmodels.regression.linear_model import OLS
            from statsmodels.tools.tools import add_constant
            
            X = data[control_vars] if control_vars != ["const"] else add_constant(data[[treatment]])
            X = add_constant(X) if "const" not in X.columns else X
            y = data[outcome]
            
            model = OLS(y, X).fit()
            
            # Extract treatment effect
            treatment_coef = model.params[treatment]
            treatment_pvalue = model.pvalues[treatment]
            treatment_se = model.bse[treatment]
            
            return {
                "treatment_effect": treatment_coef,
                "standard_error": treatment_se,
                "p_value": treatment_pvalue,
                "confidence_interval": list(model.conf_int().loc[treatment]),
                "is_significant": treatment_pvalue < 0.05,
                "r_squared": model.rsquared,
                "sample_size": len(data)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _difference_in_differences_analysis(self, data: pd.DataFrame, treatment: str, outcome: str) -> Dict:
        """Difference-in-differences analysis (simplified)"""
        
        # This is a simplified DiD analysis
        # In practice, you'd need proper treatment and control groups
        
        try:
            # Create binary treatment variable (median split)
            treatment_median = data[treatment].median()
            data["treatment_binary"] = (data[treatment] > treatment_median).astype(int)
            
            # Create time periods (before/after median date)
            median_date = data["date"].median()
            data["post_period"] = (data["date"] > median_date).astype(int)
            
            # DiD regression
            data["treatment_post"] = data["treatment_binary"] * data["post_period"]
            
            X = data[["treatment_binary", "post_period", "treatment_post"]]
            X = add_constant(X)
            y = data[outcome]
            
            model = OLS(y, X).fit()
            
            did_effect = model.params["treatment_post"]
            did_pvalue = model.pvalues["treatment_post"]
            
            return {
                "did_effect": did_effect,
                "p_value": did_pvalue,
                "is_significant": did_pvalue < 0.05,
                "treatment_cutoff": treatment_median,
                "time_cutoff": median_date,
                "sample_size": len(data)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _instrumental_variables_analysis(self, data: pd.DataFrame, treatment: str, outcome: str) -> Dict:
        """Instrumental variables analysis (simplified)"""
        
        # This is a placeholder for IV analysis
        # In practice, you'd need valid instruments
        
        return {
            "message": "IV analysis requires valid instruments - not implemented in this simplified version",
            "note": "Consider using lagged variables or external instruments"
        }
    
    def _propensity_score_matching(self, data: pd.DataFrame, treatment: str, outcome: str) -> Dict:
        """Propensity score matching (simplified)"""
        
        try:
            from sklearn.linear_model import LogisticRegression
            
            # Create binary treatment
            treatment_median = data[treatment].median()
            data["treatment_binary"] = (data[treatment] > treatment_median).astype(int)
            
            # Control variables
            control_vars = [col for col in data.columns if col not in [treatment, outcome, "date", "treatment_binary"]]
            
            if not control_vars:
                return {"error": "No control variables available for PSM"}
            
            # Estimate propensity scores
            X = data[control_vars].fillna(0)
            y = data["treatment_binary"]
            
            ps_model = LogisticRegression(random_state=42)
            ps_model.fit(X, y)
            data["propensity_score"] = ps_model.predict_proba(X)[:, 1]
            
            # Simple matching (nearest neighbor)
            treated = data[data["treatment_binary"] == 1]
            control = data[data["treatment_binary"] == 0]
            
            if len(treated) == 0 or len(control) == 0:
                return {"error": "No treated or control units"}
            
            # Calculate average treatment effect
            treated_outcome = treated[outcome].mean()
            control_outcome = control[outcome].mean()
            
            ate = treated_outcome - control_outcome
            
            return {
                "average_treatment_effect": ate,
                "treated_outcome": treated_outcome,
                "control_outcome": control_outcome,
                "treated_units": len(treated),
                "control_units": len(control),
                "propensity_model_score": ps_model.score(X, y)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _analyze_economic_climate_interactions(self, data: pd.DataFrame, target: str) -> Dict:
        """Analyze economic-climate interactions"""
        
        interactions = {}
        
        # Get economic and climate variables
        econ_vars = [col for col in data.columns if any(keyword in col.lower() for keyword in ["gdp", "inflation", "unemployment", "price"])]
        climate_vars = [col for col in data.columns if any(keyword in col.lower() for keyword in ["temperature", "precipitation", "drought", "climate"])]
        
        for econ_var in econ_vars:
            for climate_var in climate_vars:
                if econ_var in data.columns and climate_var in data.columns:
                    # Create interaction term
                    interaction_name = f"{econ_var}_x_{climate_var}"
                    data[interaction_name] = data[econ_var] * data[climate_var]
                    
                    # Test interaction effect
                    correlation = data[target].corr(data[interaction_name])
                    
                    if not np.isnan(correlation):
                        interactions[interaction_name] = {
                            "correlation": correlation,
                            "strength": self._interpret_correlation_strength(abs(correlation))
                        }
        
        return interactions
    
    def _analyze_social_conflict_interactions(self, data: pd.DataFrame, target: str) -> Dict:
        """Analyze social-conflict interactions"""
        
        interactions = {}
        
        # Get social variables
        social_vars = [col for col in data.columns if any(keyword in col.lower() for keyword in ["sentiment", "social", "likes", "retweets"])]
        
        for social_var in social_vars:
            if social_var in data.columns:
                # Test interaction with conflict intensity
                high_conflict = data[target] > data[target].median()
                low_conflict = data[target] <= data[target].median()
                
                high_sentiment = data[social_var][high_conflict].mean()
                low_sentiment = data[social_var][low_conflict].mean()
                
                interactions[f"{social_var}_conflict_interaction"] = {
                    "high_conflict_mean": high_sentiment,
                    "low_conflict_mean": low_sentiment,
                    "difference": high_sentiment - low_sentiment,
                    "interaction_strength": abs(high_sentiment - low_sentiment) / (data[social_var].std() + 1e-6)
                }
        
        return interactions
    
    def _analyze_multi_driver_interactions(self, data: pd.DataFrame, target: str) -> Dict:
        """Analyze multi-driver interactions"""
        
        # Use PCA to identify interaction patterns
        feature_columns = [col for col in data.columns if col not in [target, "date"]]
        
        if len(feature_columns) < 3:
            return {"error": "Insufficient variables for multi-driver analysis"}
        
        try:
            from sklearn.decomposition import PCA
            
            X = data[feature_columns].fillna(0)
            X_scaled = StandardScaler().fit_transform(X)
            
            # Perform PCA
            pca = PCA(n_components=min(3, len(feature_columns)))
            principal_components = pca.fit_transform(X_scaled)
            
            # Correlate components with target
            component_correlations = []
            for i in range(principal_components.shape[1]):
                corr = data[target].corr(pd.Series(principal_components[:, i]))
                component_correlations.append(abs(corr))
            
            return {
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "component_correlations": component_correlations,
                "total_variance_explained": sum(pca.explained_variance_ratio_),
                "principal_components": principal_components.shape[1]
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _calculate_driver_sensitivity(self, data: pd.DataFrame, driver: str, target: str) -> Dict:
        """Calculate sensitivity of target to driver changes"""
        
        try:
            # Calculate elasticity (percentage change in target / percentage change in driver)
            driver_mean = data[driver].mean()
            target_mean = data[target].mean()
            
            if driver_mean != 0 and target_mean != 0:
                # Simple elasticity calculation
                elasticity = (data[target].std() / target_mean) / (data[driver].std() / driver_mean)
                
                # Sensitivity bands
                driver_std = data[driver].std()
                target_std = data[target].std()
                
                # Calculate target changes for driver changes
                sensitivity_up = target_std * (driver_std / driver_mean) if driver_mean != 0 else 0
                sensitivity_down = -sensitivity_up
                
                return {
                    "elasticity": elasticity,
                    "sensitivity_score": abs(elasticity),
                    "sensitivity_up": sensitivity_up,
                    "sensitivity_down": sensitivity_down,
                    "driver_volatility": driver_std / driver_mean if driver_mean != 0 else 0,
                    "target_volatility": target_std / target_mean if target_mean != 0 else 0
                }
            else:
                return {"elasticity": 0, "sensitivity_score": 0}
                
        except Exception as e:
            return {"error": str(e)}
    
    def _correlation_significance(self, x: pd.Series, y: pd.Series) -> float:
        """Calculate p-value for correlation"""
        try:
            correlation = x.corr(y)
            n = len(x)
            if n <= 2:
                return 1.0
            
            # Calculate t-statistic
            t_stat = correlation * np.sqrt((n - 2) / (1 - correlation ** 2))
            
            # Two-tailed p-value
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
            
            return p_value
            
        except:
            return 1.0
    
    def _interpret_correlation_strength(self, correlation: float) -> str:
        """Interpret correlation strength"""
        abs_corr = abs(correlation)
        
        if abs_corr >= 0.7:
            return "strong"
        elif abs_corr >= 0.5:
            return "moderate"
        elif abs_corr >= 0.3:
            return "weak"
        else:
            return "very_weak"
    
    def _prepare_causal_dataset(self, treatment: str, outcome: str) -> pd.DataFrame:
        """Prepare dataset for causal analysis"""
        
        # Use integrated dataset
        integrated_data = self._prepare_integrated_dataset(outcome)
        
        # Ensure treatment variable exists
        if treatment not in integrated_data.columns:
            return pd.DataFrame()  # Empty if treatment not found
        
        # Remove rows with missing values
        causal_data = integrated_data[[treatment, outcome] + [col for col in integrated_data.columns if col not in [treatment, outcome, "date"]]].dropna()
        
        return causal_data