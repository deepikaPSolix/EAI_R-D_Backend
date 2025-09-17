import os

class Config:
    SECRET_KEY = 'your-secret-key'
    CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://redis:6379/0')
    CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://redis:6379/0')

    # Use environment specific base directory or default to project structure
    # This allows paths to work in both Docker (/myapp/cache) and local dev (./cache)
    BASE_DIR = os.getenv('BASE_DIR', '/myapp/cache')
    
    # Determine if we're running locally or in Docker
    if not os.path.exists(BASE_DIR) and os.path.exists('./cache'):
        # If the Docker path doesn't exist but local './cache' does, we're running locally
        BASE_DIR = os.path.abspath('./cache')
    
    # Cache directory for local storage
    CACHE_DIR = BASE_DIR
    
    # File storage paths
    GRAPH_DOC_UPLOAD = os.path.join(BASE_DIR, 'graph-file-uploads')
    SQL_UPLOAD_DIR_PATH = os.path.join(BASE_DIR, 'sql-uploads')
    UPLOAD_DIR_PATH = os.path.join(BASE_DIR, 'uploads')
    ML_DIR_PATH = os.path.join(BASE_DIR, 'ml-model')
