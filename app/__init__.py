from dotenv import load_dotenv
from flask import Flask
from celery import Celery, Task

load_dotenv()

def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object('app.config.Config')
    make_celery(app)
    from app.routes import main as main_blueprint
    app.register_blueprint(main_blueprint)
    return app

# def make_celery(app):
#     celery = Celery(
#         app.import_name,
#         backend=app.config['CELERY_RESULT_BACKEND'],
#         broker=app.config['CELERY_BROKER_URL']
#     )
#     celery.conf.update(app.config)
#     class ContextTask(celery.Task):
#         def __call__(self, *args, **kwargs):
#             with app.app_context():
#                 return self.run(*args, **kwargs)

#     celery.Task = ContextTask
#     return celery


def make_celery(app: Flask) -> Celery:
    class FlaskTask(Task):
        def __call__(self, *args: object, **kwargs: object) -> object:
            with app.app_context():
                return self.run(*args, **kwargs)

    celery_app = Celery(app.name, task_cls=FlaskTask)
    celery_app.config_from_object('app.config.Config')
    celery_app.set_default()
    app.extensions["celery"] = celery_app
    return celery_app
