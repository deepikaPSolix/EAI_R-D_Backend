import multiprocessing as mp
try:
    mp.set_start_method('spawn',force=True)
except RuntimeError:
    pass

from app import create_app

app = create_app()
celery = app.extensions['celery']

if __name__ == '__main__':
    celery.start()
