import pandas as pd
import requests
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import json
import os
from sqlalchemy.orm import Session
from models import ConflictEvent, SessionLocal, init_db

class DataIngestion:
    """Handle data ingestion from ACLED and UN datasets with DB persistence"""
    
    def __init__(self):
        self.acled_api_key = os.getenv("ACLED_API_KEY")
        self.acled_email = os.getenv("ACLED_EMAIL")
        self.base_url = "https://api.acleddata.com/acled/read"
        init_db()
    
    def fetch_acled_data(self, 
                          country: Optional[str] = None,
                          region: Optional[str] = "Eastern Africa", # Great Lakes Focus
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None,
                          force_refresh: bool = False) -> pd.DataFrame:
        """Fetch conflict data from ACLED API or Local DB"""
        
        # Check DB first if not force_refresh
        if not force_refresh:
            db_data = self._fetch_from_db(country, start_date, end_date)
            if not db_data.empty:
                return db_data

        if not self.acled_api_key:
            # Fallback to realistic sample data if no key
            print("No ACLED_API_KEY found, generating realistic samples for the Great Lakes Region...")
            df = self._generate_sample_data(country)
            self._save_to_db(df)
            return df
        
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        
        params = {
            "key": self.acled_api_key,
            "email": self.acled_email,
            "region": region,
            "event_date": start_date,
            "event_date_where": ">=",
            "limit": 5000
        }
        
        if country:
            params["country"] = country

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") == 200:
                df = pd.DataFrame(data.get("data", []))
                if not df.empty:
                    df = self.validate_data(df)
                    self._save_to_db(df)
                return df
            else:
                print(f"ACLED API Error: {data.get('message')}")
                return self._generate_sample_data(country)
        except Exception as e:
            print(f"Failed to fetch from ACLED: {str(e)}")
            return self._generate_sample_data(country)

    def _fetch_from_db(self, country: Optional[str], start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
        session = SessionLocal()
        query = session.query(ConflictEvent)
        if country:
            query = query.filter(ConflictEvent.country == country)
        if start_date:
            query = query.filter(ConflictEvent.event_date >= datetime.strptime(start_date, "%Y-%m-%d"))
        if end_date:
            query = query.filter(ConflictEvent.event_date <= datetime.strptime(end_date, "%Y-%m-%d"))
        
        events = query.all()
        session.close()
        
        if not events:
            return pd.DataFrame()
            
        data = []
        for e in events:
            data.append({
                "event_id": e.event_id,
                "event_date": e.event_date,
                "location": e.location,
                "latitude": e.latitude,
                "longitude": e.longitude,
                "event_type": e.event_type,
                "actor1": e.actor1,
                "fatalities": e.fatalities,
                "country": e.country,
                "admin1": e.admin1,
                "admin2": e.admin2
            })
        return pd.DataFrame(data)

    def _save_to_db(self, df: pd.DataFrame):
        session = SessionLocal()
        for _, row in df.iterrows():
            # Check for existing
            exists = session.query(ConflictEvent).filter(ConflictEvent.event_id == str(row["event_id"])).first()
            if not exists:
                event = ConflictEvent(
                    event_id=str(row["event_id"]),
                    event_date=pd.to_datetime(row["event_date"]),
                    year=int(row.get("year", datetime.now().year)),
                    event_type=row["event_type"],
                    actor1=row["actor1"],
                    location=row.get("location", "Unknown"),
                    latitude=float(row["latitude"]),
                    longitude=float(row["longitude"]),
                    fatalities=int(row["fatalities"]),
                    country=row["country"],
                    admin1=row.get("admin1"),
                    admin2=row.get("admin2")
                )
                session.add(event)
        session.commit()
        session.close()
    
    def fetch_un_data(self, dataset_type: str = "displacement") -> pd.DataFrame:
        """Fetch UN displacement data (Placeholder with realistic mock)"""
        # In a real app, this would call UNOCHA or IOM APIs
        data = [
            {"date": "2023-01-01", "displaced_persons": 50000, "region": "North Kivu"},
            {"date": "2023-02-01", "displaced_persons": 55000, "region": "North Kivu"},
            {"date": "2023-03-01", "displaced_persons": 70000, "region": "North Kivu"},
        ]
        return pd.DataFrame(data)
    
    def import_from_csv(self, file_path: str) -> bool:
        """Import data from a local ACLED/HDX CSV file into the DB"""
        try:
            df = pd.read_csv(file_path)
            # ACLED CSV headers often differ slightly from API JSON
            header_map = {
                "data_id": "event_id",
                "event_date": "event_date",
                "event_type": "event_type",
                "actor1": "actor1",
                "location": "location",
                "latitude": "latitude",
                "longitude": "longitude",
                "fatalities": "fatalities",
                "country": "country",
                "admin1": "admin1",
                "admin2": "admin2"
            }
            # Rename if columns exist
            df = df.rename(columns={k: v for k, v in header_map.items() if k in df.columns})
            
            if not df.empty:
                df = self.validate_data(df)
                self._save_to_db(df)
                return True
            return False
        except Exception as e:
            print(f"CSV Import Error: {str(e)}")
            return False
    
    def _generate_sample_data(self, country: Optional[str] = None) -> pd.DataFrame:
        """Generate realistic Great Lakes sample data"""
        locations = [
            {"name": "Goma", "lat": -1.6833, "lon": 29.2333, "adm1": "North Kivu", "adm2": "Goma"},
            {"name": "Bukavu", "lat": -2.5167, "lon": 28.8667, "adm1": "South Kivu", "adm2": "Bukavu"},
            {"name": "Beni", "lat": 0.4833, "lon": 29.4667, "adm1": "North Kivu", "adm2": "Beni"},
            {"name": "Butembo", "lat": 0.1333, "lon": 29.2833, "adm1": "North Kivu", "adm2": "Butembo"},
            {"name": "Uvira", "lat": -3.4000, "lon": 29.1500, "adm1": "South Kivu", "adm2": "Uvira"},
            {"name": "Bunia", "lat": 1.5667, "lon": 30.2500, "adm1": "Ituri", "adm2": "Bunia"},
            {"name": "Fizi", "lat": -4.3000, "lon": 28.9500, "adm1": "South Kivu", "adm2": "Fizi"}
        ]
        
        event_types = ["Battle", "Violence against civilians", "Remote violence", "Riots", "Protests"]
        actors = ["M23", "ADF", "FARDC", "Mai-Mai", "CODECO", "Local militias"]
        
        data = []
        for i in range(200):
            location = locations[i % len(locations)]
            event_date = datetime.now() - timedelta(days=i*2)
            
            data.append({
                "event_id": str(100000 + i),
                "event_date": event_date.strftime("%Y-%m-%d"),
                "location": location["name"],
                "latitude": location["lat"] + (i % 5) * 0.005,
                "longitude": location["lon"] + (i % 5) * 0.005,
                "event_type": event_types[i % len(event_types)],
                "actor1": actors[i % len(actors)],
                "fatalities": max(0, (i % 15) - 3),
                "country": country or "Democratic Republic of Congo",
                "admin1": location["adm1"],
                "admin2": location["adm2"]
            })
        
        return pd.DataFrame(data)

class DataProcessor:
    """Process and analyze conflict data"""
    
    def __init__(self):
        self.data = None
    
    def load_data(self, df: pd.DataFrame):
        if df is not None and not df.empty:
            df = df.copy()
            df["event_date"] = pd.to_datetime(df["event_date"])
        self.data = df
    
    def calculate_trends(self, period: str = "monthly") -> List[Dict]:
        if self.data is None or self.data.empty:
            return []
        
        temp_df = self.data.copy()
        if period == "monthly":
            temp_df["period"] = temp_df["event_date"].dt.to_period("M").astype(str)
        elif period == "weekly":
            temp_df["period"] = temp_df["event_date"].dt.to_period("W").astype(str)
        else:
            temp_df["period"] = temp_df["event_date"].dt.strftime("%Y-%m-%d")
        
        trends = temp_df.groupby("period").agg({
            "event_id": "count",
            "fatalities": "sum"
        }).reset_index()
        
        trends.columns = ["period", "total_events", "total_fatalities"]
        return trends.to_dict("records")
    
    def identify_hotspots(self, threshold: int = 5) -> List[Dict]:
        if self.data is None or self.data.empty:
            return []
        
        # Cluster by admin2 (Territory) for more actionable insights
        hotspots = self.data.groupby(["admin2", "admin1"]).agg({
            "event_id": "count",
            "fatalities": "sum",
            "latitude": "mean",
            "longitude": "mean"
        }).reset_index()
        
        hotspots.columns = ["location", "province", "event_count", "total_fatalities", "latitude", "longitude"]
        hotspots = hotspots[hotspots["event_count"] >= threshold]
        return hotspots.sort_values("event_count", ascending=False).to_dict("records")
    
    def analyze_actors(self) -> List[Dict]:
        if self.data is None or self.data.empty:
            return []
        
        actor_analysis = self.data.groupby("actor1").agg({
            "event_id": "count",
            "fatalities": "sum"
        }).reset_index()
        
        actor_analysis.columns = ["actor", "event_count", "total_fatalities"]
        return actor_analysis.sort_values("event_count", ascending=False).to_dict("records")