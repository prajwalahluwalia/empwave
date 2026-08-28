from datetime import datetime, timezone

from flask import Blueprint, current_app, jsonify, render_template, request

from empwave.services.intent_classifier import get_classifier


web = Blueprint("web", __name__)


@web.get("/")
def index():
    return render_template("index.html")


@web.post("/api/process-speech")
def process_speech():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(
            status="error",
            message="Request body must be a JSON object.",
        ), 400

    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        return jsonify(
            status="error",
            message="A non-empty text value is required.",
        ), 400

    text = text.strip()
    if len(text) > 1000:
        return jsonify(
            status="error",
            message="Text must be 1000 characters or fewer.",
        ), 400

    try:
        analysis = get_classifier().classify(text)
    except Exception:
        current_app.logger.exception("Semantic NLP classification failed")
        return jsonify(
            status="error",
            message="The local NLP model could not analyze this text.",
        ), 503

    return jsonify(
        status="success",
        text=text,
        timestamp=datetime.now(timezone.utc).isoformat(),
        **analysis,
    )
