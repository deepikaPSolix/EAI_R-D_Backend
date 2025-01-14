import os

class Config:
    SECRET_KEY = 'your-secret-key'
    CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://redis:6379/0')
    CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://redis:6379/0')

    BASE_DIR = '/myapp/cache'
    UPLOAD_DIR_PATH = os.path.join(BASE_DIR, 'uploads')
    ML_DIR_PATH = os.path.join(BASE_DIR, 'ml-model')
