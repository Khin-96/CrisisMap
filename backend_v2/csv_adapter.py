import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
import re
import os

logger = logging.getLogger(__name__)

class CSVAdapter:
    """
    Adaptive CSV processor that handles different column formats and structures
    """
    
    # Column mapping patterns - maps common variations to standard names
    COLUMN_MAPPINGS = {
        'event_date': [
            'event_date', 'date', 'event_dt', 'occurrence_date', 'week', 'time'
        ],
        'latitude': [
            'latitude', 'lat', 'y', 'centroid_latitude', 'event_latitude'
        ],
        'longitude': [
            'longitude', 'lng', 'lon', 'x', 'centroid_longitude', 'event_longitude'
        ],
        'event_type': [
            'event_type', 'type', 'event_category', 'disorder_type', 'sub_event_type'
        ],
        'fatalities': [
            'fatalities', 'deaths', 'casualties', 'killed', 'dead'
        ],
        'location': [
            'location', 'place', 'city', 'admin1', 'admin2', 'locality'
        ],
        'actor1': [
            'actor1', 'primary_actor', 'perpetrator', 'group1'
        ],
        'actor2': [
            'actor2', 'secondary_actor', 'target', 'group2'
        ],
        'country': [
            'country', 'nation', 'state'
        ],
        'notes': [
            'notes', 'description', 'details', 'comments', 'summary'
        ],
        'period': [
            'period', 'time_period', 'forecast_period', 'date_period'
        ],
        'expected_forecast': [
            'expected_forecast', 'forecast', 'expected', 'prediction', 'expected_events'
        ],
        'low_forecast': [
            'low_forecast', 'low', 'minimum', 'lower_bound', 'low_scenario'
        ],
        'high_forecast': [
            'high_forecast', 'high', 'maximum', 'upper_bound', 'high_scenario'
        ],
        'admin1': [
            'admin1', 'province', 'state', 'region', 'administrative_division'
        ],
        'outcome': [
            'outcome', 'event_type_forecast', 'category', 'violence_type'
        ]
    }
    
    def __init__(self):
        self.detected_mappings = {}
        self.data_quality_report = {}
    
    def _read_file_robust(self, file_path: str) -> pd.DataFrame:
        """Helper to robustly read Excel or CSV files with multiple strategies"""
        df = None
        file_lower = file_path.lower()
        file_basename = os.path.basename(file_path)
        
        # Strategy 1: Try as Excel first if it looks like one or has no extension
        if file_lower.endswith(('.xlsx', '.xls', '.xlsm')) or "." not in file_basename:
            # We prioritize calamine for speed and robust 'Strict' format support
            for engine in ['calamine', 'openpyxl', None, 'xlrd']:
                try:
                    # For CAST files, we might need to check multiple sheets
                    if 'cast' in file_basename.lower():
                        xl = pd.ExcelFile(file_path, engine=engine)
                        sheet_name = 0
                        if 'Results' in xl.sheet_names:
                            sheet_name = 'Results'
                        elif 'Data' in xl.sheet_names:
                            sheet_name = 'Data'
                        
                        df = xl.parse(sheet_name)
                        logger.info(f"Successfully read {file_path} (sheet: {sheet_name}) as Excel with engine {engine}")
                        return df
                    
                    df = pd.read_excel(file_path, engine=engine)
                    logger.info(f"Successfully read {file_path} as Excel with engine {engine}")
                    return df
                except Exception as e:
                    logger.debug(f"Engine {engine} failed for {file_path}: {e}")
                    continue
        
        # Strategy 2: Try as CSV
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding, on_bad_lines='skip')
                if df is not None and len(df.columns) > 1:
                    logger.info(f"Successfully read {file_path} as CSV with encoding {encoding}")
                    return df
            except Exception:
                continue
        
        # Check for semi-colon delimiter
        if df is not None and len(df.columns) <= 1:
            try:
                df_semi = pd.read_csv(file_path, sep=';', on_bad_lines='skip')
                if len(df_semi.columns) > 1:
                    return df_semi
            except:
                pass
                
        if df is None:
            raise ValueError(f"Could not read file {file_path}. Please ensure it is a valid CSV or Excel file.")
        return df
    
    def analyze_csv(self, file_path: str, data_type: str = 'acled_events') -> Dict[str, Any]:
        """
        Analyze CSV file structure and detect column mappings
        """
        try:
            df = self._read_file_robust(file_path)
            
            # Detect column mappings
            self.detected_mappings = self._detect_column_mappings(df.columns.tolist())
            
            # Generate data quality report
            self.data_quality_report = self._analyze_data_quality(df)
            
            # Use dynamic required columns based on data_type
            missing_required = self._get_missing_required_columns(data_type)
            
            return {
                'total_rows': len(df),
                'columns': df.columns.tolist(),
                'detected_mappings': self.detected_mappings,
                'data_quality': self.data_quality_report,
                'sample_data': df.head().to_dict('records'),
                'missing_required': missing_required
            }
            
        except Exception as e:
            logger.error(f"Error analyzing file: {e}")
            raise
    
    def _detect_column_mappings(self, columns: List[str]) -> Dict[str, str]:
        """
        Detect which columns map to our standard schema
        """
        mappings = {}
        columns_lower = [col.lower().strip() for col in columns]
        
        for standard_col, variations in self.COLUMN_MAPPINGS.items():
            for variation in variations:
                for i, col_lower in enumerate(columns_lower):
                    if variation.lower() in col_lower or col_lower in variation.lower():
                        mappings[standard_col] = columns[i]
                        break
                if standard_col in mappings:
                    break
        
        return mappings
    
    def _analyze_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze data quality and completeness
        """
        quality_report = {
            'total_rows': len(df),
            'columns_analysis': {},
            'data_types': {},
            'missing_data': {},
            'potential_issues': []
        }
        
        for col in df.columns:
            col_data = df[col]
            quality_report['columns_analysis'][col] = {
                'non_null_count': col_data.notna().sum(),
                'null_count': col_data.isna().sum(),
                'unique_values': col_data.nunique(),
                'data_type': str(col_data.dtype)
            }
            
            # Check for potential coordinate columns
            if col.lower() in ['latitude', 'lat', 'centroid_latitude']:
                lat_range = col_data.dropna()
                if len(lat_range) > 0:
                    if lat_range.min() < -90 or lat_range.max() > 90:
                        quality_report['potential_issues'].append(
                            f"Latitude values in {col} outside valid range (-90, 90)"
                        )
            
            if col.lower() in ['longitude', 'lng', 'lon', 'centroid_longitude']:
                lng_range = col_data.dropna()
                if len(lng_range) > 0:
                    if lng_range.min() < -180 or lng_range.max() > 180:
                        quality_report['potential_issues'].append(
                            f"Longitude values in {col} outside valid range (-180, 180)"
                        )
        
        return quality_report
    
    def _get_missing_required_columns(self, data_type: str = 'acled_events') -> List[str]:
        """
        Get list of required columns that are missing based on data type
        """
        if data_type == 'cast_predictions':
            required = ['country', 'period', 'expected_forecast']
        else:
            # Default to ACLED events
            required = ['event_date', 'latitude', 'longitude', 'event_type', 'fatalities']
        
        return [col for col in required if col not in self.detected_mappings]
    
    def process_csv(self, file_path: str, custom_mappings: Dict[str, str] = None) -> pd.DataFrame:
        """
        Process CSV file and convert to standard format
        """
        try:
            # Read file robustly
            df = self._read_file_robust(file_path)
            
            # Use custom mappings if provided, otherwise use detected mappings
            mappings = custom_mappings or self.detected_mappings
            
            # Create standardized dataframe
            standardized_df = pd.DataFrame()
            
            # Map columns to standard format
            for standard_col, source_col in mappings.items():
                if source_col in df.columns:
                    standardized_df[standard_col] = df[source_col]
            
            # Process and clean data
            standardized_df = self._clean_and_process_data(standardized_df)
            
            # Add missing optional columns with defaults
            optional_columns = ['location', 'actor1', 'actor2', 'country', 'notes']
            for col in optional_columns:
                if col not in standardized_df.columns:
                    standardized_df[col] = None
            
            # Generate unique event IDs
            standardized_df['event_id'] = [
                f"evt_{datetime.now().strftime('%Y%m%d')}_{i:06d}" 
                for i in range(len(standardized_df))
            ]
            
            return standardized_df
            
        except Exception as e:
            logger.error(f"Error processing CSV: {e}")
            raise
    
    def _clean_and_process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and process the standardized data
        """
        # Clean event_date
        if 'event_date' in df.columns:
            df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
            df['event_date'] = df['event_date'].dt.strftime('%Y-%m-%d')
        
        # Clean coordinates
        if 'latitude' in df.columns:
            df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
            # Filter valid latitude range
            df = df[(df['latitude'] >= -90) & (df['latitude'] <= 90)]
        
        if 'longitude' in df.columns:
            df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
            # Filter valid longitude range
            df = df[(df['longitude'] >= -180) & (df['longitude'] <= 180)]
        
        # Clean fatalities
        if 'fatalities' in df.columns:
            df['fatalities'] = pd.to_numeric(df['fatalities'], errors='coerce').fillna(0)
            df['fatalities'] = df['fatalities'].astype(int)
        
        # Clean text fields
        text_columns = ['event_type', 'location', 'actor1', 'actor2', 'country', 'notes']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace('nan', None)
        
        # Remove rows with missing required data
        required_cols = ['event_date', 'latitude', 'longitude', 'event_type']
        for col in required_cols:
            if col in df.columns:
                df = df.dropna(subset=[col])
        
        return df
    
    def get_column_suggestions(self, columns: List[str]) -> Dict[str, List[str]]:
        """
        Get suggestions for column mappings based on column names
        """
        suggestions = {}
        columns_lower = [col.lower() for col in columns]
        
        for standard_col, variations in self.COLUMN_MAPPINGS.items():
            matches = []
            for col in columns:
                col_lower = col.lower()
                for variation in variations:
                    if variation in col_lower or col_lower in variation:
                        matches.append(col)
                        break
            suggestions[standard_col] = matches
        
        return suggestions
    
    def validate_processed_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the processed data and return validation report
        """
        validation_report = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check required columns
        required_cols = ['event_date', 'latitude', 'longitude', 'event_type', 'fatalities']
        missing_required = [col for col in required_cols if col not in df.columns]
        
        if missing_required:
            validation_report['is_valid'] = False
            validation_report['errors'].append(f"Missing required columns: {missing_required}")
        
        # Check data quality
        if len(df) == 0:
            validation_report['is_valid'] = False
            validation_report['errors'].append("No valid data rows after processing")
        
        # Generate statistics
        if len(df) > 0:
            validation_report['statistics'] = {
                'total_events': len(df),
                'date_range': {
                    'start': df['event_date'].min() if 'event_date' in df.columns else None,
                    'end': df['event_date'].max() if 'event_date' in df.columns else None
                },
                'countries': df['country'].nunique() if 'country' in df.columns else 0,
                'event_types': df['event_type'].nunique() if 'event_type' in df.columns else 0,
                'total_fatalities': df['fatalities'].sum() if 'fatalities' in df.columns else 0,
                'coordinate_bounds': {
                    'lat_min': df['latitude'].min() if 'latitude' in df.columns else None,
                    'lat_max': df['latitude'].max() if 'latitude' in df.columns else None,
                    'lng_min': df['longitude'].min() if 'longitude' in df.columns else None,
                    'lng_max': df['longitude'].max() if 'longitude' in df.columns else None
                }
            }
        
        return validation_report