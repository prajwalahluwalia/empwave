from datetime import datetime, timezone

from flask import Blueprint, current_app, jsonify, render_template, request

from empwave.services.feedback_store import get_feedback_store
from empwave.services.intent_classifier import (
    REGION_CONFIG,
    SUPPORTED_EMOTIONS,
    get_classifier,
)


web = Blueprint("web", __name__)
SUPPORTED_REGION_IDS = frozenset(REGION_CONFIG)


def _request_text():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return None, (
            jsonify(
                status="error",
                message="Request body must be a JSON object.",
            ),
            400,
        )

    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        return None, (
            jsonify(
                status="error",
                message="A non-empty text value is required.",
            ),
            400,
        )

    text = text.strip()
    if len(text) > 1000:
        return None, (
            jsonify(
                status="error",
                message="Text must be 1000 characters or fewer.",
            ),
            400,
        )
    return text, None


def _classifier():
    return current_app.extensions["empwave_classifier"]


def _simulation_regions(analysis):
    regions = []
    for region in analysis["regions"]:
        if region["id"] not in SUPPORTED_REGION_IDS:
            raise ValueError(
                f"Unsupported brain region returned: {region['id']}"
            )
        all_sources = region.get("sources", ())
        non_baseline_sources = [
            source
            for source in all_sources
            if source.get("type") != "baseline"
        ]
        sources = non_baseline_sources if non_baseline_sources else all_sources
        if not sources:
            continue
        trigger = next(
            (
                source.get("evidence")
                for source in sources
                if source.get("evidence")
            ),
            region.get("evidence"),
        )
        regions.append(
            {
                "id": region["id"],
                "strength": float(region["strength"]),
                "trigger": str(trigger or ""),
            }
        )
    return regions


def _spoken_summary(regions):
    if not regions:
        return "No brain region matched the sentence confidently."
    descriptions = [
        (
            f"{REGION_CONFIG[region['id']]['reason']} "
            f"Trigger: {region['trigger']}."
        )
        for region in regions
    ]
    return " ".join(descriptions)


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


@web.get("/health")
def health():
    classifier = _classifier()
    return jsonify(
        status="ok",
        model=classifier.model_name,
        model_loaded=True,
    )


@web.post("/simulate")
def simulate():
    text, error_response = _request_text()
    if error_response:
        return error_response

    try:
        analysis = _classifier().classify(text)
        regions = _simulation_regions(analysis)
    except Exception:
        current_app.logger.exception("Semantic simulation failed")
        return jsonify(
            message="The local NLP model could not analyze this text.",
        ), 503

    return jsonify(
        regions=regions,
        fallback=not regions,
        spoken_text=_spoken_summary(regions),
        analysis=analysis,
    )


@web.post("/api/process-speech")
def process_speech():
    text, error_response = _request_text()
    if error_response:
        return error_response

    try:
        analysis = _classifier().classify(text)
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
        analysis = _classifier().classify(text)
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
