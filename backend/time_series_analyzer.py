import pandas as pd
import numpy as np
from scipy import stats
from scipy.fft import fft, fftfreq
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AdvancedTimeSeriesAnalyzer:
    """Advanced time series analysis for conflict data"""
    
    def __init__(self):
        self.data = None
        self.seasonal_components = {}
        self.trend_components = {}
        self.forecast_models = {}
    
    def load_data(self, df: pd.DataFrame):
        """Load time series data"""
        self.data = df.copy()
        self.data["event_date"] = pd.to_datetime(self.data["event_date"])
        self.data = self.data.sort_values("event_date")
    
    def decompose_time_series(self, 
                             variable: str = "fatalities",
                             freq: str = "W",
                             model: str = "additive") -> Dict:
        """Decompose time series into trend, seasonal, and residual components"""
        
        if self.data is None:
            raise ValueError("No data loaded")
        
        # Aggregate by frequency
        ts_data = self.data.set_index("event_date").resample(freq)[variable].sum()
        ts_data = ts_data.fillna(0)
        
        if len(ts_data) < 8:
            return {"error": "Insufficient data for decomposition"}
        
        # Simple decomposition using moving averages
        # Trend component
        trend_period = min(12, len(ts_data) // 3)
        trend = ts_data.rolling(window=trend_period, center=True).mean()
        
        # Detrended series
        detrended = ts_data - trend
        
        # Seasonal component
        seasonal_period = self._detect_seasonal_period(detrended.dropna())
        if seasonal_period > 1:
            seasonal = detrended.groupby(detrended.index % seasonal_period).transform("mean")
        else:
            seasonal = pd.Series(0, index=ts_data.index)
        
        # Residual component
        residual = ts_data - trend - seasonal
        
        # Store components
        self.seasonal_components[variable] = seasonal
        self.trend_components[variable] = trend
        
        # Calculate component strengths
        total_variance = ts_data.var()
        trend_strength = trend.var() / total_variance if total_variance > 0 else 0
        seasonal_strength = seasonal.var() / total_variance if total_variance > 0 else 0
        residual_strength = residual.var() / total_variance if total_variance > 0 else 0
        
        return {
            "original": ts_data.to_dict(),
            "trend": trend.fillna(0).to_dict(),
            "seasonal": seasonal.fillna(0).to_dict(),
            "residual": residual.fillna(0).to_dict(),
            "component_strengths": {
                "trend": trend_strength,
                "seasonal": seasonal_strength,
                "residual": residual_strength
            },
            "seasonal_period": seasonal_period,
            "decomposition_method": "moving_average"
        }
    
    def detect_seasonal_patterns(self, 
                                variable: str = "fatalities",
                                max_period: int = 52) -> Dict:
        """Detect seasonal patterns using multiple methods"""
        
        if self.data is None:
            raise ValueError("No data loaded")
        
        # Prepare time series
        ts_data = self.data.set_index("event_date").resample("W")[variable].sum()
        ts_data = ts_data.fillna(0)
        
        if len(ts_data) < 12:
            return {"error": "Insufficient data for seasonality detection"}
        
        # Method 1: Autocorrelation analysis
        autocorr_results = self._autocorrelation_analysis(ts_data, max_period)
        
        # Method 2: Fourier analysis
        fourier_results = self._fourier_analysis(ts_data)
        
        # Method 3: Periodogram
        periodogram_results = self._periodogram_analysis(ts_data)
        
        # Combine results
        detected_periods = []
        
        # From autocorrelation
        if autocorr_results["significant_periods"]:
            detected_periods.extend(autocorr_results["significant_periods"])
        
        # From Fourier analysis
        if fourier_results["dominant_frequencies"]:
            for freq in fourier_results["dominant_frequencies"]:
                period = int(1 / freq) if freq > 0 else None
                if period and 2 <= period <= max_period:
                    detected_periods.append(period)
        
        # Remove duplicates and sort
        detected_periods = sorted(list(set(detected_periods)))
        
        # Seasonality strength
        seasonality_strength = self._calculate_seasonality_strength(ts_data, detected_periods)
        
        return {
            "detected_periods": detected_periods,
            "seasonality_strength": seasonality_strength,
            "autocorrelation_analysis": autocorr_results,
            "fourier_analysis": fourier_results,
            "periodogram_analysis": periodogram_results,
            "is_seasonal": len(detected_periods) > 0 and seasonality_strength > 0.3
        }
    
    def forecast_with_methods(self, 
                            variable: str = "fatalities",
                            periods_ahead: int = 8,
                            methods: List[str] = None) -> Dict:
        """Forecast using multiple time series methods"""
        
        if methods is None:
            methods = ["naive", "moving_average", "exponential_smoothing", "linear_trend"]
        
        if self.data is None:
            raise ValueError("No data loaded")
        
        # Prepare time series
        ts_data = self.data.set_index("event_date").resample("W")[variable].sum()
        ts_data = ts_data.fillna(0)
        
        if len(ts_data) < 12:
            return {"error": "Insufficient data for forecasting"}
        
        forecasts = {}
        
        # Method 1: Naive forecast (last observed value)
        if "naive" in methods:
            naive_forecast = [ts_data.iloc[-1]] * periods_ahead
            forecasts["naive"] = {
                "values": naive_forecast,
                "method": "naive",
                "confidence": "low"
            }
        
        # Method 2: Moving average
        if "moving_average" in methods:
            ma_window = min(4, len(ts_data) // 3)
            ma_forecast = [ts_data.tail(ma_window).mean()] * periods_ahead
            forecasts["moving_average"] = {
                "values": ma_forecast,
                "method": "moving_average",
                "window": ma_window,
                "confidence": "medium"
            }
        
        # Method 3: Exponential smoothing
        if "exponential_smoothing" in methods:
            es_forecast = self._exponential_smoothing_forecast(ts_data, periods_ahead)
            forecasts["exponential_smoothing"] = es_forecast
        
        # Method 4: Linear trend
        if "linear_trend" in methods:
            trend_forecast = self._linear_trend_forecast(ts_data, periods_ahead)
            forecasts["linear_trend"] = trend_forecast
        
        # Method 5: Seasonal naive (if seasonality detected)
        seasonal_analysis = self.detect_seasonal_patterns(variable)
        if seasonal_analysis["is_seasonal"] and "seasonal_naive" in methods:
            seasonal_forecast = self._seasonal_naive_forecast(ts_data, periods_ahead, seasonal_analysis)
            forecasts["seasonal_naive"] = seasonal_forecast
        
        # Ensemble forecast (average of all methods)
        if len(forecasts) > 1:
            ensemble_values = []
            for method_data in forecasts.values():
                ensemble_values.append(method_data["values"])
            
            ensemble_forecast = np.mean(ensemble_values, axis=0).tolist()
            forecasts["ensemble"] = {
                "values": ensemble_forecast,
                "method": "ensemble_average",
                "confidence": "high",
                "component_methods": list(forecasts.keys())
            }
        
        return {
            "forecasts": forecasts,
            "historical_data": ts_data.to_dict(),
            "forecast_periods": periods_ahead,
            "last_observed_date": ts_data.index[-1].strftime("%Y-%m-%d"),
            "variable": variable
        }
    
    def detect_structural_breaks(self, 
                                variable: str = "fatalities",
                                min_breaks: int = 1,
                                max_breaks: int = 5) -> Dict:
        """Detect structural breaks in time series"""
        
        if self.data is None:
            raise ValueError("No data loaded")
        
        # Prepare time series
        ts_data = self.data.set_index("event_date").resample("W")[variable].sum()
        ts_data = ts_data.fillna(0)
        
        if len(ts_data) < 20:
            return {"error": "Insufficient data for structural break detection"}
        
        # Simple break detection using mean shifts
        breaks = []
        break_scores = []
        
        # Test different break points
        for n_breaks in range(min_breaks, min(max_breaks + 1, len(ts_data) // 10)):
            break_candidates = self._find_break_points(ts_data, n_breaks)
            
            for candidate in break_candidates:
                score = self._calculate_break_score(ts_data, candidate)
                break_scores.append((candidate, score))
        
        # Select best breaks
        if break_scores:
            break_scores.sort(key=lambda x: x[1], reverse=True)
            selected_breaks = [break for break, score in break_scores[:max_breaks]]
            
            # Analyze break periods
            break_analysis = []
            for break_point in selected_breaks:
                if break_point < len(ts_data):
                    break_date = ts_data.index[break_point]
                    pre_break = ts_data.iloc[:break_point]
                    post_break = ts_data.iloc[break_point:]
                    
                    break_analysis.append({
                        "break_date": break_date.strftime("%Y-%m-%d"),
                        "break_index": break_point,
                        "pre_break_mean": pre_break.mean(),
                        "post_break_mean": post_break.mean(),
                        "mean_change": post_break.mean() - pre_break.mean(),
                        "pre_break_std": pre_break.std(),
                        "post_break_std": post_break.std(),
                        "significance": "high" if abs(post_break.mean() - pre_break.mean()) > 2 * pre_break.std() else "medium"
                    })
        else:
            break_analysis = []
        
        return {
            "structural_breaks": break_analysis,
            "total_breaks": len(break_analysis),
            "break_detected": len(break_analysis) > 0,
            "method": "mean_shift_detection"
        }
    
    def analyze_cycles_and_patterns(self, 
                                   variable: str = "fatalities") -> Dict:
        """Analyze cyclical patterns and recurring events"""
        
        if self.data is None:
            raise ValueError("No data loaded")
        
        # Prepare time series
        ts_data = self.data.set_index("event_date").resample("W")[variable].sum()
        ts_data = ts_data.fillna(0)
        
        if len(ts_data) < 24:
            return {"error": "Insufficient data for cycle analysis"}
        
        # Cycle detection
        cycle_analysis = {}
        
        # Business cycle type analysis
        cycle_analysis["business_cycles"] = self._analyze_business_cycles(ts_data)
        
        # Event cycle analysis (patterns in event occurrences)
        cycle_analysis["event_cycles"] = self._analyze_event_cycles()
        
        # Recurring pattern detection
        cycle_analysis["recurring_patterns"] = self._detect_recurring_patterns(ts_data)
        
        # Phase analysis
        cycle_analysis["phase_analysis"] = self._analyze_cycle_phases(ts_data)
        
        return cycle_analysis
    
    def _autocorrelation_analysis(self, ts_data: pd.Series, max_lag: int) -> Dict:
        """Calculate autocorrelation function"""
        autocorr_values = []
        lags = []
        
        for lag in range(1, min(max_lag, len(ts_data) // 2)):
            if len(ts_data) > lag:
                correlation = ts_data.autocorr(lag=lag)
                if not np.isnan(correlation):
                    autocorr_values.append(correlation)
                    lags.append(lag)
        
        # Find significant periods (peaks in autocorrelation)
        significant_periods = []
        if autocorr_values:
            threshold = 0.3  # Significance threshold
            for i, corr in enumerate(autocorr_values):
                if abs(corr) > threshold:
                    significant_periods.append(lags[i])
        
        return {
            "autocorrelations": dict(zip(lags, autocorr_values)),
            "significant_periods": significant_periods,
            "max_correlation": max(autocorr_values) if autocorr_values else 0
        }
    
    def _fourier_analysis(self, ts_data: pd.Series) -> Dict:
        """Perform Fourier analysis to detect periodicities"""
        # Remove mean
        ts_centered = ts_data - ts_data.mean()
        
        # FFT
        fft_values = fft(ts_centered.values)
        frequencies = fftfreq(len(ts_centered))
        
        # Power spectrum
        power = np.abs(fft_values) ** 2
        
        # Find dominant frequencies (excluding DC component)
        positive_freq_idx = frequencies > 0
        positive_frequencies = frequencies[positive_freq_idx]
        positive_power = power[positive_freq_idx]
        
        # Top frequencies
        top_freq_idx = np.argsort(positive_power)[-5:][::-1]
        dominant_frequencies = positive_frequencies[top_freq_idx].tolist()
        
        return {
            "dominant_frequencies": dominant_frequencies,
            "power_spectrum": dict(zip(frequencies, power)),
            "total_power": np.sum(power)
        }
    
    def _periodogram_analysis(self, ts_data: pd.Series) -> Dict:
        """Simple periodogram analysis"""
        # Calculate periodogram using FFT
        fft_values = fft(ts_data.values)
        frequencies = fftfreq(len(ts_data))
        power = np.abs(fft_values) ** 2
        
        # Convert to periods
        periods = 1 / np.abs(frequencies[frequencies != 0])
        power_values = power[frequencies != 0]
        
        # Find peaks
        peak_periods = []
        if len(periods) > 0:
            peak_threshold = np.mean(power_values) + 2 * np.std(power_values)
            peak_mask = power_values > peak_threshold
            peak_periods = periods[peak_mask].tolist()
        
        return {
            "peak_periods": peak_periods,
            "periodogram": dict(zip(periods, power_values))
        }
    
    def _detect_seasonal_period(self, ts_data: pd.Series) -> int:
        """Detect the dominant seasonal period"""
        max_period = min(52, len(ts_data) // 3)
        best_period = 1
        best_correlation = 0
        
        for period in range(2, max_period + 1):
            if len(ts_data) >= 2 * period:
                # Calculate correlation between lagged series
                correlation = ts_data.autocorr(lag=period)
                if not np.isnan(correlation) and abs(correlation) > abs(best_correlation):
                    best_correlation = correlation
                    best_period = period
        
        return best_period
    
    def _calculate_seasonality_strength(self, ts_data: pd.Series, periods: List[int]) -> float:
        """Calculate overall seasonality strength"""
        if not periods:
            return 0
        
        max_strength = 0
        for period in periods:
            if len(ts_data) >= 2 * period:
                correlation = abs(ts_data.autocorr(lag=period))
                max_strength = max(max_strength, correlation)
        
        return max_strength
    
    def _exponential_smoothing_forecast(self, ts_data: pd.Series, periods: int) -> Dict:
        """Simple exponential smoothing forecast"""
        alpha = 0.3  # Smoothing parameter
        
        # Calculate smoothed values
        smoothed = [ts_data.iloc[0]]
        for i in range(1, len(ts_data)):
            smoothed.append(alpha * ts_data.iloc[i] + (1 - alpha) * smoothed[-1])
        
        # Forecast (last smoothed value)
        forecast_values = [smoothed[-1]] * periods
        
        return {
            "values": forecast_values,
            "method": "exponential_smoothing",
            "alpha": alpha,
            "confidence": "medium"
        }
    
    def _linear_trend_forecast(self, ts_data: pd.Series, periods: int) -> Dict:
        """Linear trend forecast"""
        x = np.arange(len(ts_data))
        y = ts_data.values
        
        # Fit linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Forecast
        last_x = len(ts_data) - 1
        forecast_x = np.arange(last_x + 1, last_x + periods + 1)
        forecast_values = slope * forecast_x + intercept
        
        return {
            "values": forecast_values.tolist(),
            "method": "linear_trend",
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_value ** 2,
            "confidence": "medium" if r_value ** 2 > 0.3 else "low"
        }
    
    def _seasonal_naive_forecast(self, ts_data: pd.Series, periods: int, seasonal_analysis: Dict) -> Dict:
        """Seasonal naive forecast"""
        detected_periods = seasonal_analysis.get("detected_periods", [])
        
        if not detected_periods:
            # Fallback to naive forecast
            return {
                "values": [ts_data.iloc[-1]] * periods,
                "method": "seasonal_naive",
                "confidence": "low"
            }
        
        # Use shortest detected period
        period = min(detected_periods)
        
        # Get seasonal values
        seasonal_values = []
        for i in range(periods):
            lag_index = len(ts_data) - (period - (i % period))
            if lag_index >= 0:
                seasonal_values.append(ts_data.iloc[lag_index])
            else:
                seasonal_values.append(ts_data.mean())
        
        return {
            "values": seasonal_values,
            "method": "seasonal_naive",
            "seasonal_period": period,
            "confidence": "medium"
        }
    
    def _find_break_points(self, ts_data: pd.Series, n_breaks: int) -> List[int]:
        """Find potential break points using simple method"""
        break_points = []
        
        # Divide data into segments
        segment_size = len(ts_data) // (n_breaks + 1)
        
        for i in range(1, n_breaks + 1):
            break_point = i * segment_size
            if break_point < len(ts_data):
                break_points.append(break_point)
        
        return break_points
    
    def _calculate_break_score(self, ts_data: pd.Series, break_point: int) -> float:
        """Calculate score for a potential break point"""
        if break_point >= len(ts_data) or break_point == 0:
            return 0
        
        pre_break = ts_data.iloc[:break_point]
        post_break = ts_data.iloc[break_point:]
        
        # Score based on mean difference
        mean_diff = abs(post_break.mean() - pre_break.mean())
        pooled_std = np.sqrt(((len(pre_break) - 1) * pre_break.var() + 
                             (len(post_break) - 1) * post_break.var()) / 
                            (len(pre_break) + len(post_break) - 2))
        
        if pooled_std > 0:
            score = mean_diff / pooled_std
        else:
            score = 0
        
        return score
    
    def _analyze_business_cycles(self, ts_data: pd.Series) -> Dict:
        """Analyze business cycle patterns"""
        # Simple cycle analysis
        diff_series = ts_data.diff().dropna()
        
        # Find peaks and troughs
        peaks = []
        troughs = []
        
        for i in range(1, len(diff_series) - 1):
            if diff_series.iloc[i-1] < diff_series.iloc[i] > diff_series.iloc[i+1]:
                peaks.append(i)
            elif diff_series.iloc[i-1] > diff_series.iloc[i] < diff_series.iloc[i+1]:
                troughs.append(i)
        
        # Calculate cycle statistics
        cycle_lengths = []
        for i in range(len(peaks) - 1):
            cycle_lengths.append(peaks[i+1] - peaks[i])
        
        return {
            "peaks": peaks,
            "troughs": troughs,
            "cycle_lengths": cycle_lengths,
            "average_cycle_length": np.mean(cycle_lengths) if cycle_lengths else None
        }
    
    def _analyze_event_cycles(self) -> Dict:
        """Analyze cycles in event patterns"""
        if self.data is None:
            return {}
        
        # Event type cycles
        event_cycles = {}
        
        for event_type in self.data["event_type"].unique():
            type_data = self.data[self.data["event_type"] == event_type]
            if len(type_data) > 10:
                # Resample by week
                weekly_counts = type_data.set_index("event_date").resample("W").size()
                
                # Simple cycle detection
                if len(weekly_counts) > 20:
                    autocorr = weekly_counts.autocorr(lag=4)  # 4-week lag
                    event_cycles[event_type] = {
                        "autocorrelation_4w": autocorr,
                        "is_cyclical": abs(autocorr) > 0.3,
                        "frequency": len(type_data) / len(self.data)
                    }
        
        return event_cycles
    
    def _detect_recurring_patterns(self, ts_data: pd.Series) -> Dict:
        """Detect recurring patterns in time series"""
        # Simple pattern detection using template matching
        patterns = []
        
        # Look for repeating sequences
        min_pattern_length = 3
        max_pattern_length = min(12, len(ts_data) // 4)
        
        for pattern_length in range(min_pattern_length, max_pattern_length + 1):
            if len(ts_data) >= 2 * pattern_length:
                pattern = ts_data.iloc[:pattern_length].values
                
                # Search for this pattern in the rest of the series
                matches = 0
                for i in range(pattern_length, len(ts_data) - pattern_length + 1):
                    candidate = ts_data.iloc[i:i+pattern_length].values
                    correlation = np.corrcoef(pattern, candidate)[0, 1]
                    if correlation > 0.7:  # High correlation threshold
                        matches += 1
                
                if matches > 0:
                    patterns.append({
                        "pattern_length": pattern_length,
                        "matches": matches,
                        "pattern": pattern.tolist(),
                        "strength": matches / (len(ts_data) / pattern_length)
                    })
        
        # Sort by strength
        patterns.sort(key=lambda x: x["strength"], reverse=True)
        
        return {
            "recurring_patterns": patterns[:5],  # Top 5 patterns
            "total_patterns": len(patterns)
        }
    
    def _analyze_cycle_phases(self, ts_data: pd.Series) -> Dict:
        """Analyze different phases of cycles"""
        # Simple phase analysis based on moving averages
        short_ma = ts_data.rolling(window=4).mean()
        long_ma = ts_data.rolling(window=12).mean()
        
        # Determine phases
        phases = []
        for i in range(len(ts_data)):
            if pd.isna(short_ma.iloc[i]) or pd.isna(long_ma.iloc[i]):
                phases.append("unknown")
            elif short_ma.iloc[i] > long_ma.iloc[i]:
                if ts_data.iloc[i] > short_ma.iloc[i]:
                    phases.append("expansion")
                else:
                    phases.append("contraction")
            else:
                phases.append("recession")
        
        # Count phases
        phase_counts = pd.Series(phases).value_counts().to_dict()
        
        return {
            "phase_sequence": phases,
            "phase_distribution": phase_counts,
            "current_phase": phases[-1] if phases else "unknown"
        }