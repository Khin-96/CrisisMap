import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    MONGODB_URL = os.getenv('MONGODB_URL', 'mongodb://admin:crisismap2024@localhost:27017/crisismap?authSource=admin')
    MONGODB_DATABASE = os.getenv('MONGODB_DATABASE', 'crisismap')
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
    
    UPLOAD_DIR = 'uploads'
    MODELS_DIR = 'models'
    
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
    
    # ML Config
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # API Config
    API_HOST = '0.0.0.0'
    API_PORT = 8000
    DEBUG = os.getenv('DEBUG', 'False') == 'True'
