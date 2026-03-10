import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import io
import uuid
from datetime import datetime
import logging
from motor.motor_asyncio import AsyncIOMotorDatabase

logger = logging.getLogger(__name__)

class CSVProcessor:
    """Enhanced CSV processing with validation and MongoDB integration"""
    
    REQUIRED_COLUMNS = [
        'event_date', 'latitude', 'longitude', 'event_type', 
        'fatalities', 'location', 'country'
    ]
    
    OPTIONAL_COLUMNS = [
        'actor1', 'actor2', 'notes', 'event_id'
    ]
    
    def __init__(self):
        self.validation_errors = []
    
    async def process_file(
        self, 
        file_content: bytes, 
        filename: str, 
        db: AsyncIOMotorDatabase
    ) -> Dict[str, Any]:
        """Process uploaded CSV/Excel file and store in MongoDB"""
        
        try:
            # Reset validation errors
            self.validation_errors = []
            
            # Read file based on extension
            if filename.endswith('.csv'):
                df = pd.read_csv(io.BytesIO(file_content))
            elif filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(io.BytesIO(file_content))
            else:
                return {
                    "success": False,
                    "message": "Unsupported file format",
                    "processed_rows": 0
                }
            
            logger.info(f"Loaded file {filename} with {len(df)} rows")
            
            # Validate and clean data
            df_cleaned = await self._validate_and_clean(df)
            
            if df_cleaned is None or len(df_cleaned) == 0:
                return {
                    "success": False,
                    "message": "No valid data found after validation",
                    "processed_rows": 0,
                    "validation_errors": self.validation_errors
                }
            
            # Generate dataset ID
            dataset_id = str(uuid.uuid4())
            
            # Prepare documents for MongoDB
            documents = []
            for _, row in df_cleaned.iterrows():
                doc = {
                    "dataset_id": dataset_id,
                    "event_id": row.get("event_id", str(uuid.uuid4())),
                    "event_date": row["event_date"],
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                    "event_type": row["event_type"],
                    "fatalities": int(row["fatalities"]),
                    "location": row["location"],
                    "country": row["country"],
                    "actor1": row.get("actor1"),
                    "actor2": row.get("actor2"),
                    "notes": row.get("notes"),
                    "created_at": datetime.utcnow(),
                    "source": "csv_upload",
                    "filename": filename
                }
                documents.append(doc)
            
            # Insert into MongoDB
            if documents:
                result = await db.events.insert_many(documents)
                logger.info(f"Inserted {len(result.inserted_ids)} documents")
                
                # Store dataset metadata
                dataset_meta = {
                    "dataset_id": dataset_id,
                    "filename": filename,
                    "total_rows": len(documents),
                    "created_at": datetime.utcnow(),
                    "validation_errors": self.validation_errors,
                    "status": "processed"
                }
                await db.datasets.insert_one(dataset_meta)
            
            return {
                "success": True,
                "message": f"Successfully processed {len(documents)} rows",
                "processed_rows": len(documents),
                "validation_errors": self.validation_errors if self.validation_errors else None,
                "dataset_id": dataset_id
            }
            
        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")
            return {
                "success": False,
                "message": f"Processing error: {str(e)}",
                "processed_rows": 0
            }
    
    async def _validate_and_clean(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Validate and clean the dataframe"""
        
        original_rows = len(df)
        logger.info(f"Starting validation of {original_rows} rows")
        
        # Check required columns
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            self.validation_errors.append(f"Missing required columns: {missing_cols}")
            return None
        
        # Create a copy for cleaning
        df_clean = df.copy()
        
        # Clean and validate each column
        df_clean = self._clean_dates(df_clean)
        df_clean = self._clean_coordinates(df_clean)
        df_clean = self._clean_fatalities(df_clean)
        df_clean = self._clean_text_fields(df_clean)
        
        # Remove rows with critical missing data
        initial_count = len(df_clean)
        df_clean = df_clean.dropna(subset=['event_date', 'latitude', 'longitude'])
        
        if len(df_clean) < initial_count:
            dropped = initial_count - len(df_clean)
            self.validation_errors.append(f"Dropped {dropped} rows with missing critical data")
        
        # Validate coordinate ranges
        df_clean = df_clean[
            (df_clean['latitude'].between(-90, 90)) & 
            (df_clean['longitude'].between(-180, 180))
        ]
        
        if len(df_clean) < initial_count:
            dropped = initial_count - len(df_clean)
            self.validation_errors.append(f"Dropped {dropped} rows with invalid coordinates")
        
        final_rows = len(df_clean)
        if final_rows < original_rows:
            self.validation_errors.append(
                f"Data quality: {final_rows}/{original_rows} rows passed validation"
            )
        
        logger.info(f"Validation complete: {final_rows} valid rows")
        return df_clean
    
    def _clean_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize date column"""
        try:
            df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
            
            # Count invalid dates
            invalid_dates = df['event_date'].isna().sum()
            if invalid_dates > 0:
                self.validation_errors.append(f"Found {invalid_dates} invalid dates")
            
            return df
        except Exception as e:
            self.validation_errors.append(f"Date parsing error: {str(e)}")
            return df
    
    def _clean_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate coordinate columns"""
        try:
            df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
            df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
            
            # Count invalid coordinates
            invalid_lat = df['latitude'].isna().sum()
            invalid_lon = df['longitude'].isna().sum()
            
            if invalid_lat > 0:
                self.validation_errors.append(f"Found {invalid_lat} invalid latitude values")
            if invalid_lon > 0:
                self.validation_errors.append(f"Found {invalid_lon} invalid longitude values")
            
            return df
        except Exception as e:
            self.validation_errors.append(f"Coordinate parsing error: {str(e)}")
            return df
    
    def _clean_fatalities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate fatalities column"""
        try:
            df['fatalities'] = pd.to_numeric(df['fatalities'], errors='coerce').fillna(0)
            df['fatalities'] = df['fatalities'].astype(int)
            
            # Ensure non-negative
            negative_fatalities = (df['fatalities'] < 0).sum()
            if negative_fatalities > 0:
                self.validation_errors.append(f"Found {negative_fatalities} negative fatality values, set to 0")
                df.loc[df['fatalities'] < 0, 'fatalities'] = 0
            
            return df
        except Exception as e:
            self.validation_errors.append(f"Fatalities parsing error: {str(e)}")
            return df
    
    def _clean_text_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize text fields"""
        try:
            text_columns = ['event_type', 'location', 'country', 'actor1', 'actor2', 'notes']
            
            for col in text_columns:
                if col in df.columns:
                    # Strip whitespace and standardize
                    df[col] = df[col].astype(str).str.strip()
                    df[col] = df[col].replace('nan', None)
                    df[col] = df[col].replace('', None)
            
            return df
        except Exception as e:
            self.validation_errors.append(f"Text cleaning error: {str(e)}")
            return df
    
    async def get_dataset_info(self, dataset_id: str, db: AsyncIOMotorDatabase) -> Dict[str, Any]:
        """Get information about a processed dataset"""
        try:
            dataset_meta = await db.datasets.find_one({"dataset_id": dataset_id})
            if not dataset_meta:
                return {"error": "Dataset not found"}
            
            # Get sample of events
            sample_events = await db.events.find(
                {"dataset_id": dataset_id}
            ).limit(5).to_list(length=5)
            
            return {
                "dataset_id": dataset_id,
                "metadata": dataset_meta,
                "sample_events": sample_events,
                "total_events": await db.events.count_documents({"dataset_id": dataset_id})
            }
            
        except Exception as e:
            logger.error(f"Error getting dataset info: {e}")
            return {"error": str(e)}