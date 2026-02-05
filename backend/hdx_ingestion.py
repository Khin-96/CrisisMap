import requests
import pandas as pd
import os
from typing import List, Dict, Optional
from datetime import datetime
from models import SessionLocal, HumanitarianIndicator, init_db

class HDXIngestor:
    """Ingest humanitarian data from HDX HAPI (Humanitarian API)"""
    
    def __init__(self):
        self.base_url = "https://hapi.humdata.org/api/v1"
        self.app_identifier = os.getenv("HDX_APP_IDENTIFIER")
        init_db()
        
    def _get_headers(self):
        # Use the App Identifier as User-Agent if available, per HDX best practices
        user_agent = self.app_identifier if self.app_identifier else "CrisisMap/2.0 (Analysis/Research)"
        return {
            "User-Agent": user_agent
        }

    def fetch_operational_presence(self, location_code: str = "COD") -> pd.DataFrame:
        """
        Fetch Operational Presence (3W) data.
        aggregates number of organizations by sector and admin1.
        
        Args:
            location_code: ISO3 code (e.g., 'COD' for DR Congo)
        """
        if not self.app_identifier:
            print("HDX_APP_IDENTIFIER is missing. Please set it in .env")
            return pd.DataFrame()

        endpoint = "/coordination-context/operational-presence"
        url = f"{self.base_url}{endpoint}"
        
        params = {
            "location_code": location_code,
            "output_format": "json",
            "offset": 0,
            "limit": 10000, # Max limit per docs
            "app_identifier": self.app_identifier
        }
        
        try:
            print(f"Fetching Operational Presence for {location_code}...")
            response = requests.get(url, params=params, headers=self._get_headers())
            response.raise_for_status()
            data = response.json()
            
            if not data:
                return pd.DataFrame()
                
            df = pd.DataFrame(data)
            return df
            
        except Exception as e:
            print(f"Error fetching operational presence: {e}")
            return pd.DataFrame()

    def fetch_population_data(self, location_code: str = "COD") -> pd.DataFrame:
        """
        Fetch Population data.
        
        Args:
            location_code: ISO3 code (e.g., 'COD' for DR Congo)
        """
        if not self.app_identifier:
            print("HDX_APP_IDENTIFIER is missing.")
            return pd.DataFrame()

        endpoint = "/population-social/population"
        url = f"{self.base_url}{endpoint}"
        
        params = {
            "location_code": location_code,
            "output_format": "json",
            "offset": 0,
            "limit": 10000,
            "app_identifier": self.app_identifier
        }
        
        try:
            print(f"Fetching Population Data for {location_code}...")
            response = requests.get(url, params=params, headers=self._get_headers())
            response.raise_for_status()
            data = response.json()
            
            if not data:
                return pd.DataFrame()
                
            df = pd.DataFrame(data)
            return df
            
        except Exception as e:
            print(f"Error fetching population data: {e}")
            return pd.DataFrame()

    def sync_data(self, country_iso3: str = "COD"):
        """Sync HAPI data to local DB"""
        
        # 1. Operational Presence -> Convert to 'org_count' indicators
        op_df = self.fetch_operational_presence(country_iso3)
        if not op_df.empty and "sector_name" in op_df.columns and "admin1_name" in op_df.columns:
            # Aggregate: Count unique organizations per Sector per Admin1
            agg_df = op_df.groupby(["admin1_name", "sector_name"])["org_name"].nunique().reset_index()
            agg_df.rename(columns={"org_name": "org_count"}, inplace=True)
            
            self._save_indicators_to_db(agg_df, "operational_presence", country_iso3)

        # 2. Population -> Convert to 'population' indicators
        pop_df = self.fetch_population_data(country_iso3)
        if not pop_df.empty and "population" in pop_df.columns:
            # Aggregate total population per Admin1 (summing over genders/ages/admin2 if present)
            # Check what columns we have. Usually 'admin1_name' and 'population'
            if "admin1_name" in pop_df.columns:
                 agg_pop = pop_df.groupby("admin1_name")["population"].sum().reset_index()
                 self._save_indicators_to_db(agg_pop, "population", country_iso3, unit="people")

    def _save_indicators_to_db(self, df: pd.DataFrame, indicator_type: str, source: str, unit: str = "count"):
        session = SessionLocal()
        
        for _, row in df.iterrows():
            try:
                # Handle different schemas from aggregations above
                if indicator_type == "operational_presence":
                    # Row: admin1_name, sector_name, org_count
                    if "org_count" not in row or row["org_count"] == 0:
                        continue
                    indicator = HumanitarianIndicator(
                        indicator_type=f"org_count_{row['sector_name'].lower().replace(' ', '_')}", # e.g. org_count_nutrition
                        region=row.get("admin1_name", "Unknown"), # Mapping admin1 to region field
                        admin1=row.get("admin1_name", "Unknown"),
                        admin2="All",
                        value=float(row["org_count"]),
                        unit="organizations",
                        source=f"HDX HAPI ({source})",
                        date=datetime.now()
                    )
                elif indicator_type == "population":
                    # Row: admin1_name, population
                    indicator = HumanitarianIndicator(
                        indicator_type="population",
                        region=row.get("admin1_name", "Unknown"),
                        admin1=row.get("admin1_name", "Unknown"),
                        admin2="All",
                        value=float(row["population"]),
                        unit=unit,
                        source=f"HDX HAPI ({source})",
                        date=datetime.now()
                    )
                else:
                    continue

                session.add(indicator)
            except Exception as e:
                print(f"Error saving row {row}: {e}")
                continue
                
        try:
            session.commit()
            print(f"Successfully synced {len(df)} {indicator_type} records.")
        except Exception as e:
            session.rollback()
            print(f"DB Commit Error: {e}")
        finally:
            session.close()

def run_regional_sync():
    ingestor = HDXIngestor()
    # Default to Great Lakes / DRC for now
    ingestor.sync_data("COD")

if __name__ == "__main__":
    run_regional_sync()
