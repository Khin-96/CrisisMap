import pandas as pd
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class EventClassifier:
    """Classify and separate CAST and ACLED events"""
    
    # ACLED typical columns
    ACLED_COLUMNS = {
        'data', 'iso', 'event_id_cnty', 'event_date', 'year', 'time_precision',
        'event_type', 'sub_event_type', 'actor1', 'inter1', 'actor2', 'inter2',
        'interaction', 'region', 'country', 'admin1', 'admin2', 'admin3',
        'location', 'latitude', 'longitude', 'geo_precision', 'source',
        'source_scale', 'notes', 'fatalities', 'timestamp'
    }
    
    # CAST typical columns
    CAST_COLUMNS = {
        'country', 'admin1', 'admin2', 'month', 'year', 'period',
        'expected_forecast', 'low_forecast', 'high_forecast',
        'total_forecast', 'battles_forecast', 'erv_forecast', 'vac_forecast',
        'total_observed', 'battles_observed', 'erv_observed', 'vac_observed'
    }
    
    @staticmethod
    def classify_dataframe(df: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
        """
        Classify whether dataframe is ACLED or CAST format
        Returns: (event_type, classification_info)
        """
        df_columns = set(col.lower().strip() for col in df.columns)
        
        # Count matching columns
        acled_matches = len(df_columns & EventClassifier.ACLED_COLUMNS)
        cast_matches = len(df_columns & EventClassifier.CAST_COLUMNS)
        
        classification_info = {
            'acled_matches': acled_matches,
            'cast_matches': cast_matches,
            'total_columns': len(df_columns),
            'columns': list(df_columns)
        }
        
        # Determine type based on matches
        if acled_matches > cast_matches and acled_matches >= 5:
            return 'acled', classification_info
        elif cast_matches > acled_matches and cast_matches >= 5:
            return 'cast', classification_info
        elif acled_matches >= 5:
            return 'acled', classification_info
        elif cast_matches >= 5:
            return 'cast', classification_info
        else:
            return 'unknown', classification_info
    
    @staticmethod
    def normalize_acled(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize ACLED data to standard format"""
        df = df.copy()
        df.columns = [col.lower().strip() for col in df.columns]
        
        # Ensure required columns
        required = ['event_date', 'latitude', 'longitude', 'event_type', 'fatalities']
        for col in required:
            if col not in df.columns:
                if col == 'event_date' and 'data' in df.columns:
                    df['event_date'] = df['data']
                elif col == 'fatalities' and 'fatalities' not in df.columns:
                    df['fatalities'] = 0
                else:
                    raise ValueError(f"Missing required column: {col}")
        
        # Standardize data types
        df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        df['fatalities'] = pd.to_numeric(df['fatalities'], errors='coerce').fillna(0)
        
        # Add source
        df['data_source'] = 'acled'
        df['event_source'] = 'acled'
        
        return df
    
    @staticmethod
    def normalize_cast(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize CAST data to standard format"""
        df = df.copy()
        df.columns = [col.lower().strip() for col in df.columns]
        
        # CAST is forecast data, not event data
        # Convert to event-like format for storage
        
        # Standardize data types
        if 'period' in df.columns:
            df['event_date'] = pd.to_datetime(df['period'], errors='coerce')
        elif 'month' in df.columns and 'year' in df.columns:
            df['event_date'] = pd.to_datetime(
                df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2) + '-01',
                errors='coerce'
            )
        else:
            df['event_date'] = pd.to_datetime('today')
        
        # CAST doesn't have coordinates, use country center (placeholder)
        df['latitude'] = 0.0
        df['longitude'] = 0.0
        
        # Use forecast as "fatalities" for compatibility
        df['fatalities'] = pd.to_numeric(df.get('expected_forecast', 0), errors='coerce').fillna(0)
        
        # Add event type
        df['event_type'] = 'forecast'
        
        # Add source
        df['data_source'] = 'cast'
        df['event_source'] = 'cast'
        
        # Store forecast values
        df['forecast_total'] = pd.to_numeric(df.get('expected_forecast', 0), errors='coerce').fillna(0)
        df['forecast_low'] = pd.to_numeric(df.get('low_forecast', 0), errors='coerce').fillna(0)
        df['forecast_high'] = pd.to_numeric(df.get('high_forecast', 0), errors='coerce').fillna(0)
        
        return df
    
    @staticmethod
    def separate_events(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Separate mixed ACLED and CAST data
        Returns: (acled_df, cast_df)
        """
        acled_df = pd.DataFrame()
        cast_df = pd.DataFrame()
        
        # Try to identify by columns
        event_type, info = EventClassifier.classify_dataframe(df)
        
        if event_type == 'acled':
            acled_df = EventClassifier.normalize_acled(df)
        elif event_type == 'cast':
            cast_df = EventClassifier.normalize_cast(df)
        else:
            # Try to separate by column patterns
            df_lower = df.copy()
            df_lower.columns = [col.lower().strip() for col in df_lower.columns]
            
            acled_cols = df_lower.columns & EventClassifier.ACLED_COLUMNS
            cast_cols = df_lower.columns & EventClassifier.CAST_COLUMNS
            
            if len(acled_cols) > len(cast_cols):
                acled_df = EventClassifier.normalize_acled(df)
            elif len(cast_cols) > len(acled_cols):
                cast_df = EventClassifier.normalize_cast(df)
        
        return acled_df, cast_df
