import sys
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
from flask import Flask
from celery import Celery, Task
import logging

# Fix for transformers library calling sys.stdout.isatty() on filtered streams
if not hasattr(sys.stdout, 'isatty'):
    sys.stdout.isatty = lambda: False
if not hasattr(sys.stderr, 'isatty'):
    sys.stderr.isatty = lambda: False

from sentence_transformers import SentenceTransformer

st_model = SentenceTransformer("all-MiniLM-L6-v2")
logging.getLogger('LiteLLM').setLevel(logging.ERROR)
logging.getLogger('together').setLevel(logging.ERROR)
logging.getLogger('ppocr').setLevel(logging.ERROR)

load_dotenv()

def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object('app.config.Config')
    make_celery(app)
    configure_logging(app)
    from app.routes import main as main_blueprint
    app.register_blueprint(main_blueprint)
    return app


def make_celery(app: Flask) -> Celery:
    class FlaskTask(Task):
        def __call__(self, *args: object, **kwargs: object) -> object:
            with app.app_context():
                return self.run(*args, **kwargs)

    celery_app = Celery(app.name, task_cls=FlaskTask)
    celery_app.config_from_object('app.config.Config')
    celery_app.conf.task_queues = {
        'cpu_queue': {'exchange': 'cpu', 'routing_key': 'cpu'},
        'gpu_queue': {'exchange': 'gpu', 'routing_key': 'gpu'},
    }
    celery_app.conf.task_default_queue = 'cpu_queue'
    celery_app.conf.task_default_exchange = 'cpu'
    celery_app.conf.task_default_routing_key = 'cpu'
    celery_app.set_default()
    app.extensions["celery"] = celery_app
    return celery_app


def configure_logging(app: Flask):
    # Remove the default Flask logger to prevent duplicate messages
    app.logger.handlers.clear()

    # Create log directory if not exists
    import os
    log_dir = "cache/logs"
    os.makedirs(log_dir, exist_ok=True)

    # File handler with rotation
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, 'app.log'),
        maxBytes=1000000,  # 1MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter('[%(asctime)s] - %(name)s - %(levelname)s - %(funcName)s - %(message)s')
    )

    # Stream handler (console logging)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter('[%(asctime)s] - %(name)s - %(levelname)s - %(funcName)s - %(message)s')
    )

    # Add handlers to the Flask app logger
    app.logger.addHandler(file_handler)
    app.logger.addHandler(console_handler)

    app.logger.setLevel(logging.INFO)  # Log level for the app

    # Suppress noisy polling endpoints from Werkzeug request log
    class SuppressPollingFilter(logging.Filter):
        def filter(self, record):
            msg = record.getMessage()
            return "rag2EvalApi" not in msg

    logging.getLogger('werkzeug').addFilter(SuppressPollingFilter())
