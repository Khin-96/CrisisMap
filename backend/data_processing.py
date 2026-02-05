import pandas as pd
import requests
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import json
import os
from sqlalchemy.orm import Session
from models import ConflictEvent, HumanitarianIndicator, SessionLocal, init_db

class DataIngestion:
    """Handle data ingestion from ACLED and UN datasets with DB persistence"""
    
    REGION_MAP = {
        "Western Africa": 1,
        "Middle Africa": 2,
        "Eastern Africa": 3,
        "Southern Africa": 4,
        "Northern Africa": 5,
        "South Asia": 7,
        "Southeast Asia": 9,
        "Middle East": 11,
        "Europe": 12,
        "Caucasus and Central Asia": 13,
        "Central America": 14,
        "South America": 15,
        "Caribbean": 16,
        "East Asia": 17,
        "North America": 18,
        "Oceania": 19,
        "Antarctica": 20
    }
    
    def __init__(self):
        self.acled_username = os.getenv("ACLED_USERNAME")
        self.acled_password = os.getenv("ACLED_PASSWORD")
        # Fallback for compatibility or if user put key in password field
        if not self.acled_password:
             self.acled_password = os.getenv("ACLED_API_KEY")

        self.base_url = "https://acleddata.com/api/acled/read"
        self.token_url = "https://acleddata.com/oauth/token"
        self.access_token = None
        init_db()

    def _get_access_token(self) -> Optional[str]:
        if self.access_token:
            return self.access_token
            
        if not self.acled_username or not self.acled_password:
             # Fallback: check if we just have an API KEY from old config, 
             # but strictly speaking the new API needs proper credentials.
             print("ACLED Credentials (USERNAME/PASSWORD) missing.")
             return None

        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        payload = {
            'username': self.acled_username,
            'password': self.acled_password,
            'grant_type': 'password',
            'client_id': 'acled'
        }
        
        try:
            response = requests.post(self.token_url, headers=headers, data=payload)
            if response.status_code == 200:
                self.access_token = response.json().get('access_token')
                return self.access_token
            else:
                print(f"Failed to obtain ACLED token: {response.status_code} {response.text}")
                return None
        except Exception as e:
            print(f"Error connecting to ACLED auth: {e}")
            return None

    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean the dataframe"""
        # Map ACLED API columns to internal names if needed (e.g. event_id_cnty -> event_id)
        if "event_id_cnty" in df.columns:
            df = df.rename(columns={"event_id_cnty": "event_id"})
        return df
    
    def fetch_acled_data(self, 
                          country: Optional[str] = None,
                          region: Optional[str] = None, # Optional, defaults to None
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None,
                          force_refresh: bool = False) -> pd.DataFrame:
        """Fetch conflict data from ACLED API or Local DB"""
        
        # Check DB first if not force_refresh
        if not force_refresh:
            db_data = self._fetch_from_db(country, start_date, end_date)
            if not db_data.empty:
                return db_data

        token = self._get_access_token()
        if not token:
            # Fallback to realistic sample data if no key
            print("No ACLED Token available, generating realistic samples for the Great Lakes Region...")
            df = self._generate_sample_data(country)
            self._save_to_db(df)
            return df
        
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            
        params = {
            "event_date": start_date,
            "event_date_where": ">=",
            "limit": 5000
        }
        
        if region:
            region_id = self.REGION_MAP.get(region)
            if region_id:
                params["region"] = region_id
        
        if country:
            params["country"] = country

        headers = {"Authorization": f"Bearer {token}"}

        try:
            response = requests.get(self.base_url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") == 200:
                df = pd.DataFrame(data.get("data", []))
                if not df.empty:
                    df = self.validate_data(df)
                    self._save_to_db(df)
                return df
            else:
                print(f"ACLED API Error: {data.get('message', 'Unknown error')}")
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
        """Fetch UN displacement/humanitarian data from local DB"""
        session = SessionLocal()
        try:
            query = session.query(HumanitarianIndicator)
            if dataset_type:
                # Filter by simple text match for now
                query = query.filter(HumanitarianIndicator.indicator_type.contains(dataset_type))
            
            results = query.all()
            if not results:
                 # Fallback to mock if empty
                 return self._generate_mock_un_data()
                 
            data = []
            for r in results:
                data.append({
                    "date": r.date,
                    "region": r.region,
                    "admin1": r.admin1,
                    "value": r.value,
                    "unit": r.unit,
                    "type": r.indicator_type,
                    "source": r.source
                })
            return pd.DataFrame(data)
        except Exception as e:
            print(f"Error fetching UN data: {e}")
            return self._generate_mock_un_data()
        finally:
            session.close()

    def _generate_mock_un_data(self):
        data = [
            {"date": "2023-01-01", "value": 50000, "region": "North Kivu", "type": "displacement"},
            {"date": "2023-02-01", "value": 55000, "region": "North Kivu", "type": "displacement"},
            {"date": "2023-03-01", "value": 70000, "region": "North Kivu", "type": "displacement"},
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
        """Generate realistic sample data for Great Lakes Region"""
        country = country or "Democratic Republic of Congo"
        
        # Define locations per country
        locations_db = {
            "Democratic Republic of Congo": [
                {"name": "Goma", "lat": -1.6833, "lon": 29.2333, "adm1": "North Kivu", "adm2": "Goma"},
                {"name": "Bukavu", "lat": -2.5167, "lon": 28.8667, "adm1": "South Kivu", "adm2": "Bukavu"},
                {"name": "Beni", "lat": 0.4833, "lon": 29.4667, "adm1": "North Kivu", "adm2": "Beni"},
                {"name": "Bunia", "lat": 1.5667, "lon": 30.2500, "adm1": "Ituri", "adm2": "Bunia"}
            ],
            "Uganda": [
                {"name": "Kampala", "lat": 0.3476, "lon": 32.5825, "adm1": "Central", "adm2": "Kampala"},
                {"name": "Gulu", "lat": 2.7724, "lon": 32.2881, "adm1": "Northern", "adm2": "Gulu"},
                {"name": "Kasese", "lat": 0.1833, "lon": 30.0833, "adm1": "Western", "adm2": "Kasese"}
            ],
            "Rwanda": [
                {"name": "Kigali", "lat": -1.9441, "lon": 30.0619, "adm1": "Kigali", "adm2": "Gasabo"},
                {"name": "Rubavu", "lat": -1.6761, "lon": 29.2638, "adm1": "Western", "adm2": "Rubavu"},
                {"name": "Musanze", "lat": -1.5000, "lon": 29.6333, "adm1": "Northern", "adm2": "Musanze"}
            ],
            "Burundi": [
                {"name": "Bujumbura", "lat": -3.3614, "lon": 29.3599, "adm1": "Bujumbura Mairie", "adm2": "Bujumbura"},
                {"name": "Gitega", "lat": -3.4264, "lon": 29.9308, "adm1": "Gitega", "adm2": "Gitega"},
                {"name": "Cibitoke", "lat": -2.8833, "lon": 29.1167, "adm1": "Cibitoke", "adm2": "Cibitoke"}
            ],
            "South Sudan": [
                {"name": "Juba", "lat": 4.8594, "lon": 31.5713, "adm1": "Central Equatoria", "adm2": "Juba"},
                {"name": "Malakal", "lat": 9.5334, "lon": 31.6605, "adm1": "Upper Nile", "adm2": "Malakal"},
                {"name": "Bor", "lat": 6.2092, "lon": 31.5589, "adm1": "Jonglei", "adm2": "Bor"}
            ]
        }
        
        # Select locations (fallback to DRC if unknown)
        locations = locations_db.get(country, locations_db["Democratic Republic of Congo"])
        
        event_types = ["Battle", "Violence against civilians", "Remote violence", "Riots", "Protests"]
        
        # Country specific actors
        actors_db = {
            "Democratic Republic of Congo": ["M23", "ADF", "FARDC", "Mai-Mai", "CODECO"],
            "Uganda": ["ADF", "UPDF", "Police Forces of Uganda", "Rioters"],
            "Rwanda": ["RDF", "Police Forces of Rwanda", "FDLR", "Unidentified Armed Group"],
            "Burundi": ["FDNB", "Imbonerakure", "RED-Tabara", "Police Forces of Burundi"],
            "South Sudan": ["SSPDF", "SPLA-IO", "NAS", "White Army", "Unknown Militia"]
        }
        actors = actors_db.get(country, ["Military Forces", "Rebel Militia", "Protesters", "Rioters"])
        
        data = []
        # Generate 600 events to ensure sufficient history for ML
        for i in range(600):
            location = locations[i % len(locations)]
            # Spread events over last 365 days
            days_ago = (i * 0.6) # approx 2 events per day spread
            event_date = datetime.now() - timedelta(days=days_ago)
            
            # Add some randomness to lat/lon for spatial clustering
            lat_offset = (i % 7 - 3) * 0.01
            lon_offset = (i % 5 - 2) * 0.01
            
            data.append({
                "event_id": str(100000 + i),
                "event_date": event_date.strftime("%Y-%m-%d"),
                "location": location["name"],
                "latitude": location["lat"] + lat_offset,
                "longitude": location["lon"] + lon_offset,
                "event_type": event_types[i % len(event_types)],
                "actor1": actors[i % len(actors)],
                "fatalities": max(0, int((i % 20) * 0.5)), # Periodic fatality spikes
                "country": country,
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