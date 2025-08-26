# MUST be first, before any other imports
import multiprocessing as mp
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

# Enforce spawn for PyTorch as well (safe if torch not installed)
try:
    import torch.multiprocessing as tmp
    tmp.set_start_method("spawn", force=True)
except Exception:
    pass

from app import create_app

app = create_app()
celery = app.extensions['celery']

if __name__ == '__main__':
    celery.start()
