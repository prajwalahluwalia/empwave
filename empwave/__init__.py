"""Empwave Flask application factory."""

import os
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS


def create_app(test_config=None):
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
    app.config.setdefault(
        "ALLOWED_ORIGIN",
        os.getenv("ALLOWED_ORIGIN", "").rstrip("/"),
    )
    if test_config:
        app.config.update(test_config)

    from empwave.routes import web
    from empwave.services.intent_classifier import get_classifier

    app.register_blueprint(web)
    classifier = get_classifier()
    app.extensions["empwave_classifier"] = classifier

    allowed_origin = app.config["ALLOWED_ORIGIN"]
    if allowed_origin:
        CORS(
            app,
            resources={
                r"/simulate": {
                    "origins": [allowed_origin],
                    "methods": ["POST", "OPTIONS"],
                    "allow_headers": ["Content-Type"],
                }
            },
        )

    @app.before_request
    def enforce_simulate_origin():
        origin = request.headers.get("Origin")
        configured_origin = app.config["ALLOWED_ORIGIN"]
        if (
            request.path == "/simulate"
            and origin
            and configured_origin
            and origin.rstrip("/") != configured_origin
        ):
            return jsonify(
                message="This origin is not allowed to call /simulate."
            ), 403

    @app.after_request
    def prevent_stale_frontend(response):
        response.headers["Cache-Control"] = (
            "no-store, no-cache, must-revalidate, max-age=0"
        )
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    return app
