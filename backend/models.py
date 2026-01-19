from sqlalchemy import Column, Integer, String, Float, DateTime, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

Base = declarative_base()

class ConflictEvent(Base):
    __tablename__ = "events"
    
    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(String, unique=True, index=True)
    event_date = Column(DateTime, index=True)
    year = Column(Integer)
    event_type = Column(String, index=True)
    sub_event_type = Column(String)
    actor1 = Column(String, index=True)
    assoc_actor_1 = Column(String)
    actor2 = Column(String, index=True)
    assoc_actor_2 = Column(String)
    region = Column(String)
    country = Column(String, index=True)
    admin1 = Column(String) # Province
    admin2 = Column(String) # District/Territory
    location = Column(String)
    latitude = Column(Float)
    longitude = Column(Float)
    fatalities = Column(Integer)
    source = Column(String)
    notes = Column(Text)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

class AnalysisResult(Base):
    __tablename__ = "analysis_results"
    
    id = Column(Integer, primary_key=True)
    analysis_type = Column(String, index=True) # "trends", "hotspots", "prediction"
    country = Column(String)
    result_data = Column(Text) # JSON string
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class HumanitarianIndicator(Base):
    __tablename__ = "humanitarian_indicators"
    
    id = Column(Integer, primary_key=True, index=True)
    indicator_type = Column(String, index=True) # "displacement", "food_security", "health"
    region = Column(String, index=True)
    admin1 = Column(String)
    admin2 = Column(String)
    value = Column(Float)
    unit = Column(String)
    source = Column(String)
    date = Column(DateTime, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

# Database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./crisismap.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)
