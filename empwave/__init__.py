"""Empwave Flask application factory."""

import os
from pathlib import Path

from flask import Flask


def create_app():
    project_root = Path(__file__).resolve().parent.parent
    app = Flask(
        __name__,
        template_folder=str(project_root / "templates"),
        static_folder=str(project_root / "static"),
    )
    app.config.setdefault(
        "FEEDBACK_DB_PATH",
        str(project_root / "data" / "feedback" / "empwave_feedback.sqlite3"),
    )
    app.config.setdefault(
        "ENABLE_EMOTION_FEEDBACK",
        os.getenv("EMPWAVE_ENABLE_EMOTION_FEEDBACK") == "1",
    )

    from empwave.routes import web

    app.register_blueprint(web)

    @app.after_request
    def prevent_stale_frontend(response):
        response.headers["Cache-Control"] = (
            "no-store, no-cache, must-revalidate, max-age=0"
        )
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    return app
