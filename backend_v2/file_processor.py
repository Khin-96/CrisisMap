import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class FileProcessor:
    REQUIRED_COLUMNS = ['event_date', 'latitude', 'longitude', 'event_type', 'fatalities']
    
    @staticmethod
    def read_file(file_path: str) -> pd.DataFrame:
        """Read CSV or Excel file"""
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            logger.info(f"Read {len(df)} rows from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            raise
    
    @staticmethod
    def validate_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Validate and clean data"""
        validation_report = {
            'total_rows': len(df),
            'missing_columns': [],
            'rows_removed': 0,
            'issues': []
        }
        
        # Check required columns
        missing = [col for col in FileProcessor.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            validation_report['missing_columns'] = missing
            raise ValueError(f"Missing required columns: {missing}")
        
        initial_rows = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Remove rows with missing required values
        df = df.dropna(subset=FileProcessor.REQUIRED_COLUMNS)
        
        # Validate coordinates
        df = df[(df['latitude'].between(-90, 90)) & (df['longitude'].between(-180, 180))]
        
        # Validate fatalities (non-negative)
        df['fatalities'] = pd.to_numeric(df['fatalities'], errors='coerce')
        df = df[df['fatalities'] >= 0]
        
        # Validate dates
        df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
        df = df.dropna(subset=['event_date'])
        
        rows_removed = initial_rows - len(df)
        validation_report['rows_removed'] = rows_removed
        
        if rows_removed > 0:
            validation_report['issues'].append(f"Removed {rows_removed} invalid rows")
        
        logger.info(f"Validation complete: {len(df)} valid rows")
        return df, validation_report
    
    @staticmethod
    def enrich_data(df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features"""
        df = df.copy()
        
        # Add temporal features
        df['year'] = df['event_date'].dt.year
        df['month'] = df['event_date'].dt.month
        df['day_of_year'] = df['event_date'].dt.dayofyear
        df['day_of_week'] = df['event_date'].dt.dayofweek
        df['quarter'] = df['event_date'].dt.quarter
        
        # Add geographic features
        df['lat_rounded'] = df['latitude'].round(1)
        df['lng_rounded'] = df['longitude'].round(1)
        
        # Add severity classification
        df['severity'] = pd.cut(df['fatalities'], 
                               bins=[0, 1, 10, 50, float('inf')],
                               labels=['low', 'medium', 'high', 'critical'])
        
        # Add event type normalization
        df['event_type'] = df['event_type'].str.lower().str.strip()
        
        logger.info("Data enrichment complete")
        return df
    
    @staticmethod
    def generate_statistics(df: pd.DataFrame) -> Dict[str, Any]:
        """Generate data statistics"""
        return {
            'total_events': len(df),
            'total_fatalities': int(df['fatalities'].sum()),
            'avg_fatalities': float(df['fatalities'].mean()),
            'date_range': {
                'start': df['event_date'].min().isoformat(),
                'end': df['event_date'].max().isoformat()
            },
            'countries': df['country'].nunique() if 'country' in df.columns else 0,
            'locations': df['location'].nunique() if 'location' in df.columns else 0,
            'event_types': df['event_type'].unique().tolist(),
            'severity_distribution': df['severity'].value_counts().to_dict() if 'severity' in df.columns else {}
        }
