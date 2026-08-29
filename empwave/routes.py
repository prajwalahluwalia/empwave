from datetime import datetime, timezone

from flask import Blueprint, current_app, jsonify, render_template, request

from empwave.services.feedback_store import get_feedback_store
from empwave.services.intent_classifier import (
    SUPPORTED_EMOTIONS,
    get_classifier,
)


web = Blueprint("web", __name__)


@web.get("/")
def index():
    return render_template(
        "index.html",
        emotion_feedback_enabled=current_app.config[
            "ENABLE_EMOTION_FEEDBACK"
        ],
    )


@web.get("/api/emotions")
def emotions():
    return jsonify(
        status="success",
        emotions=list(SUPPORTED_EMOTIONS),
    )


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
        supported_emotions=list(SUPPORTED_EMOTIONS),
        **analysis,
    )


@web.post("/api/emotion-feedback")
def emotion_feedback():
    if not current_app.config["ENABLE_EMOTION_FEEDBACK"]:
        return jsonify(
            status="error",
            message="Emotion feedback is currently disabled.",
        ), 404

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(
            status="error",
            message="Request body must be a JSON object.",
        ), 400

    text = data.get("text")
    selected_emotions = data.get("selected_emotions")
    consent = data.get("consent")
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
    if consent is not True:
        return jsonify(
            status="error",
            message="Explicit consent is required to store this feedback.",
        ), 400
    if not isinstance(selected_emotions, list) or not all(
        isinstance(emotion, str) for emotion in selected_emotions
    ):
        return jsonify(
            status="error",
            message="selected_emotions must be a list of emotion IDs.",
        ), 400

    selected_emotions = [
        emotion.strip().lower() for emotion in selected_emotions
    ]
    unsupported = sorted(
        set(selected_emotions) - set(SUPPORTED_EMOTIONS)
    )
    if unsupported:
        return jsonify(
            status="error",
            message=f"Unsupported emotions: {', '.join(unsupported)}.",
        ), 400

    try:
        analysis = get_classifier().classify(text)
        result = get_feedback_store(
            current_app.config["FEEDBACK_DB_PATH"]
        ).submit(
            text=text,
            selected_emotions=selected_emotions,
            predicted_emotions=analysis["emotions"],
            model_name=analysis["model"],
        )
    except ValueError as error:
        return jsonify(status="error", message=str(error)), 400
    except Exception:
        current_app.logger.exception("Emotion feedback submission failed")
        return jsonify(
            status="error",
            message="The feedback could not be stored.",
        ), 503

    return jsonify(
        status="success",
        message=(
            "Feedback saved for human review. It will not train the model "
            "automatically."
        ),
        **result,
    ), 201
