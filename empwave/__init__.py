"""Empwave Flask application factory."""

from pathlib import Path

from flask import Flask


def create_app():
    project_root = Path(__file__).resolve().parent.parent
    app = Flask(
        __name__,
        template_folder=str(project_root / "templates"),
        static_folder=str(project_root / "static"),
    )

    from empwave.routes import web

    app.register_blueprint(web)
    return app
