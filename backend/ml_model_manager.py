import os
import pickle
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import mlflow
import mlflow.sklearn
from pymongo import MongoClient
from bson import ObjectId
import logging

logger = logging.getLogger(__name__)

class MLModelManager:
    def __init__(self, mongo_uri: str = "mongodb://localhost:27017", db_name: str = "crisismap"):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.models_collection = self.db.models
        self.datasets_collection = self.db.datasets
        self.predictions_collection = self.db.predictions
        
        # Initialize MLflow
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("crisismap_models")
        