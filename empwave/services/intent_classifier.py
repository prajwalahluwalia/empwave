"""Empwave semantic classification for listener brain responses."""

import logging
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from joblib import load
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINED_MODEL_PATH = (
    PROJECT_ROOT / "models" / "trained" / "empwave_classifier.joblib"
)
LOGGER = logging.getLogger(__name__)

REGION_CONFIG = {
    "prefrontal": {
        "threshold": 0.37,
        "reason": "The listener is evaluating meaning, choices, priorities, or consequences.",
        "prototypes": [
            "I need to solve a logical or mathematical problem.",
            "I am reasoning carefully and comparing possible choices.",
            "I need to decide what to do and plan the next steps.",
            "I struggle to balance my job with exercise and personal activities.",
            "I need better work life balance and time management.",
            "I am evaluating whether food, service, or an experience was good or terrible.",
            "I am judging the quality of something as positive or negative.",
            "I express an opinion that food tastes unpleasant or enjoyable.",
            "I need to reason about the consequences of being hit or injured.",
            "Someone describes a serious accident and I assess what it means.",
            "I am considering a why, how, or what-if question.",
            "I decide how to handle or explain a difficult situation.",
            "I manage work, exercise, and personal priorities.",
        ],
    },
    "broca": {
        "threshold": 0.46,
        "reason": "The message evokes forming, producing, or preparing spoken language.",
        "prototypes": [
            "I am preparing words to speak or answer aloud.",
            "I need to explain an idea verbally to another person.",
            "I am forming a sentence and choosing how to say it.",
            "I was asked to repeat, pronounce, or articulate some words.",
            "I am talking, speaking, or communicating a response.",
            "I explain something aloud.",
            "I say a sentence out loud.",
            "I verbally answer another person.",
        ],
    },
    "motor": {
        "threshold": 0.52,
        "reason": "The listener represents a deliberate physical action or voluntary movement.",
        "prototypes": [
            "I am running, walking, jumping, climbing, or swimming.",
            "I deliberately move my arms, hands, legs, or body.",
            "I dance or perform a voluntary physical action.",
            "I type, write, grab, throw, kick, or reach for something.",
            "I imagine carrying out a voluntary movement.",
            "I run.",
            "I jump.",
            "I dance.",
            "I move my hand.",
        ],
    },
    "parietal": {
        "threshold": 0.38,
        "reason": "The listener represents touch, bodily impact, or spatial sensation.",
        "prototypes": [
            "I feel touch, pain, pressure, or bodily impact.",
            "A vehicle struck my body and caused physical injury.",
            "I poked or injured my eye and felt bodily pain.",
            "My hand, eye, or another body part was hurt.",
            "The surface feels rough, smooth, soft, hard, warm, or cold.",
            "I judge distance, direction, location, or where my body is in space.",
            "I navigate through a place using spatial awareness.",
        ],
    },
    "occipital": {
        "threshold": 0.38,
        "reason": "The listener constructs or processes visual imagery from the message.",
        "prototypes": [
            "I see or look at a visual scene, picture, shape, or object.",
            "I notice colors, brightness, darkness, light, or shadows.",
            "I watch something happen with my eyes.",
            "I visualize or imagine how something looks.",
            "I read visible words or inspect an image.",
        ],
    },
    "temporal_l": {
        "threshold": 0.38,
        "reason": "The listener hears and decodes the incoming spoken message.",
        "prototypes": [
            "I hear a voice and decode spoken language.",
            "I listen to music, melody, rhythm, sound, or noise.",
            "I process an incoming spoken sentence.",
            "I recognize pitch, volume, or an auditory signal.",
            "Someone is speaking and I understand the words I hear.",
        ],
    },
    "temporal_r": {
        "threshold": 0.38,
        "reason": "The listener recalls or recognizes a stored memory or familiar experience.",
        "prototypes": [
            "I remember or recall something from my past.",
            "A childhood memory comes back to me.",
            "I recognize a familiar person, place, voice, or object.",
            "I feel nostalgic about something that happened before.",
            "I retrieve a stored memory or previous experience.",
        ],
    },
    "amygdala": {
        "threshold": 0.34,
        "reason": "The listener detects emotional valence, danger, or strong personal significance.",
        "prototypes": [
            "I feel fear, anger, sadness, happiness, love, or excitement.",
            "This is frightening, dangerous, threatening, or emotionally alarming.",
            "I feel disgust because something was terrible, awful, or gross.",
            "A serious accident or injury causes alarm and concern.",
            "The message has strong positive or negative emotional meaning.",
            "I react emotionally because something was amazing, delicious, or wonderful.",
            "I am happy.",
            "I am sad.",
            "I am afraid.",
            "I feel angry.",
            "I was hit by a car and the danger was alarming.",
        ],
    },
    "cerebellum": {
        "threshold": 0.52,
        "reason": "The listener represents literal physical balance, posture, or coordination.",
        "prototypes": [
            "I physically balance my body on one foot.",
            "I walk on a tightrope without falling.",
            "My posture and physical coordination keep me stable.",
            "I coordinate a skilled athletic body movement.",
            "I stumble, trip, or lose my physical balance.",
            "I stay upright while riding a skateboard or bicycle.",
            "I physically balance.",
            "I coordinate my body.",
        ],
    },
    "brainstem": {
        "threshold": 0.50,
        "reason": "The message concerns automatic functions such as breathing, heartbeat, or sleep.",
        "prototypes": [
            "I breathe in and out automatically.",
            "My heartbeat and pulse continue without conscious control.",
            "I fall asleep, wake up, yawn, or feel drowsy.",
            "An involuntary reflex controls a basic body function.",
            "My body regulates breathing, sleep, and heart rate.",
            "I breathe slowly.",
            "I am asleep.",
        ],
    },
}

ACTIVATION_STAGE = {
    "temporal_l": 0,
    "amygdala": 1,
    "occipital": 2,
    "parietal": 2,
    "temporal_r": 2,
    "brainstem": 2,
    "motor": 3,
    "cerebellum": 3,
    "prefrontal": 4,
    "broca": 5,
}

INTENT_LABELS = {
    "prefrontal": "Reasoning and evaluation",
    "broca": "Speech formulation",
    "motor": "Voluntary movement",
    "parietal": "Body or spatial sensation",
    "occipital": "Visual imagery",
    "temporal_l": "Auditory comprehension",
    "temporal_r": "Memory and recognition",
    "amygdala": "Emotional significance",
    "cerebellum": "Physical balance and coordination",
    "brainstem": "Autonomic body state",
}

SPECIAL_INTENTS = {
    "novelty": {
        "label": "Novelty and curiosity",
        "region_id": "prefrontal",
        "threshold": 0.48,
        "reason": (
            "The listener recognizes a first-time or newly started project, "
            "which invites curiosity and interest."
        ),
        "prototypes": [
            "I am beginning my first technical or creative project.",
            "Someone is proudly sharing a new project they started for the first time.",
            "This is my first machine learning or software project.",
            "I am trying a new kind of work for the first time.",
        ],
    },
}

EMOTION_REGION_WEIGHTS = {
    "admiration": {"amygdala": 0.55, "prefrontal": 0.35},
    "amusement": {"amygdala": 0.65, "prefrontal": 0.20},
    "anger": {"amygdala": 0.95, "prefrontal": 0.40, "brainstem": 0.25},
    "annoyance": {"amygdala": 0.72, "prefrontal": 0.35},
    "approval": {"amygdala": 0.45, "prefrontal": 0.55},
    "caring": {"amygdala": 0.60, "prefrontal": 0.40},
    "confusion": {"prefrontal": 0.85},
    "curiosity": {"prefrontal": 0.90},
    "desire": {"amygdala": 0.65, "prefrontal": 0.35},
    "disappointment": {"amygdala": 0.72, "prefrontal": 0.40},
    "disapproval": {"amygdala": 0.68, "prefrontal": 0.68},
    "disgust": {"amygdala": 0.95, "prefrontal": 0.45},
    "embarrassment": {"amygdala": 0.68, "prefrontal": 0.50},
    "excitement": {"amygdala": 0.82, "brainstem": 0.35, "prefrontal": 0.25},
    "fear": {"amygdala": 1.00, "brainstem": 0.58, "prefrontal": 0.35},
    "gratitude": {"amygdala": 0.60, "prefrontal": 0.35},
    "grief": {"amygdala": 0.92, "temporal_r": 0.35, "prefrontal": 0.30},
    "joy": {"amygdala": 0.78, "prefrontal": 0.25},
    "love": {"amygdala": 0.85, "prefrontal": 0.35},
    "nervousness": {"amygdala": 0.85, "brainstem": 0.62, "prefrontal": 0.35},
    "optimism": {"amygdala": 0.55, "prefrontal": 0.48},
    "pride": {"amygdala": 0.62, "prefrontal": 0.52},
    "realization": {"prefrontal": 0.90},
    "relief": {"amygdala": 0.58, "brainstem": 0.35},
    "remorse": {"amygdala": 0.78, "prefrontal": 0.68},
    "sadness": {"amygdala": 0.88, "prefrontal": 0.35},
    "surprise": {"amygdala": 0.82, "prefrontal": 0.35},
    "neutral": {},
}

INTENT_REGION_WEIGHTS = {
    "prefrontal": {"prefrontal": 1.0},
    "broca": {"broca": 1.0},
    "motor": {"motor": 1.0},
    "parietal": {"parietal": 1.0},
    "occipital": {"occipital": 1.0},
    "temporal_l": {"temporal_l": 1.0},
    "temporal_r": {"temporal_r": 1.0},
    "amygdala": {"amygdala": 1.0},
    "cerebellum": {"cerebellum": 1.0, "motor": 0.20},
    "brainstem": {"brainstem": 1.0},
}

REGION_ACTIVATION_THRESHOLDS = {
    "prefrontal": 0.30,
    "broca": 0.40,
    "motor": 0.42,
    "parietal": 0.34,
    "occipital": 0.34,
    "temporal_l": 0.38,
    "temporal_r": 0.34,
    "amygdala": 0.32,
    "cerebellum": 0.42,
    "brainstem": 0.32,
}
EMOTION_RUNTIME_MIN_CONFIDENCE = 0.93


def _split_clauses(text):
    clauses = [
        part.strip()
        for part in re.split(
            r"[.!?;,]+|\bwhile\b|\bthen\b|\bbut\b",
            text,
            flags=re.IGNORECASE,
        )
        if part.strip()
    ]
    if text not in clauses:
        clauses.insert(0, text)
    return clauses


class SemanticIntentClassifier:
    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.prototype_embeddings = {
            region_id: self.model.encode(
                config["prototypes"],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for region_id, config in REGION_CONFIG.items()
        }
        self.special_intent_embeddings = {
            intent_id: self.model.encode(
                config["prototypes"],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for intent_id, config in SPECIAL_INTENTS.items()
        }

    def classify(self, text):
        segments = _split_clauses(text)
        segment_embeddings = self.model.encode(
            segments,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        candidates = {}
        for region_id, config in REGION_CONFIG.items():
            similarities = (
                segment_embeddings @ self.prototype_embeddings[region_id].T
            )
            flat_index = int(np.argmax(similarities))
            segment_index, prototype_index = np.unravel_index(
                flat_index, similarities.shape
            )
            score = float(similarities[segment_index, prototype_index])
            candidates[region_id] = {
                "id": region_id,
                "intent": INTENT_LABELS[region_id],
                "score": score,
                "segment_index": int(segment_index),
                "evidence": segments[segment_index],
                "reason": config["reason"],
                "semantic_concept": config["prototypes"][prototype_index],
            }

        for intent_id, config in SPECIAL_INTENTS.items():
            similarities = (
                segment_embeddings @ self.special_intent_embeddings[intent_id].T
            )
            flat_index = int(np.argmax(similarities))
            segment_index, prototype_index = np.unravel_index(
                flat_index, similarities.shape
            )
            score = float(similarities[segment_index, prototype_index])
            if score < config["threshold"]:
                continue

            region_id = config["region_id"]
            if score <= candidates[region_id]["score"]:
                continue
            candidates[region_id] = {
                "id": region_id,
                "intent": config["label"],
                "score": score,
                "segment_index": int(segment_index),
                "evidence": segments[segment_index],
                "reason": config["reason"],
                "semantic_concept": config["prototypes"][prototype_index],
            }

        selected = {
            region_id: candidate
            for region_id, candidate in candidates.items()
            if candidate["score"] >= REGION_CONFIG[region_id]["threshold"]
        }

        auditory = candidates["temporal_l"]
        auditory["score"] = max(auditory["score"], 0.62)
        auditory["evidence"] = text
        selected["temporal_l"] = auditory

        self._resolve_context_conflicts(selected)

        regions = sorted(
            selected.values(),
            key=lambda region: (
                ACTIVATION_STAGE[region["id"]],
                -region["score"],
            ),
        )
        for region in regions:
            threshold = REGION_CONFIG[region["id"]]["threshold"]
            normalized = (region["score"] - threshold) / max(0.75 - threshold, 0.1)
            region["strength"] = round(
                max(0.55, min(1.0, 0.55 + normalized * 0.45)),
                3,
            )
            region["score"] = round(region["score"], 4)
            region.pop("segment_index", None)

        primary_candidates = [
            region for region in regions if region["id"] != "temporal_l"
        ]
        primary = max(
            primary_candidates or regions,
            key=lambda region: region["score"],
        )
        intents = [
            {
                "id": region["id"],
                "label": region["intent"],
                "score": region["score"],
                "evidence": region["evidence"],
                "reason": region["reason"],
                "semantic_concept": region["semantic_concept"],
            }
            for region in regions
        ]

        return {
            "model": self.model_name,
            "baseline_only": len(regions) == 1,
            "mapping_basis": (
                "Semantic intent mapping fallback; trained emotion artifact "
                "is unavailable."
            ),
            "emotions": [],
            "intents": intents,
            "primary_intent": {
                "label": primary["intent"],
                "region_id": primary["id"],
                "score": primary["score"],
                "evidence": primary["evidence"],
                "reason": primary["reason"],
                "semantic_concept": primary["semantic_concept"],
            },
            "regions": regions,
        }

    @staticmethod
    def _resolve_context_conflicts(selected):
        prefrontal = selected.get("prefrontal")
        cerebellum = selected.get("cerebellum")
        if not prefrontal or not cerebellum:
            return
        if prefrontal["segment_index"] != cerebellum["segment_index"]:
            return
        if prefrontal["score"] >= cerebellum["score"]:
            selected.pop("cerebellum")
        else:
            selected.pop("prefrontal")


class LayeredNlpClassifier:
    """Detect emotions and intents before deriving illustrative regions."""

    def __init__(self, artifact_path=TRAINED_MODEL_PATH):
        self.artifact_path = Path(artifact_path)
        artifact = load(self.artifact_path)
        metadata = artifact.get("metadata", {})
        classifiers = artifact.get("emotion_classifiers")
        labels = metadata.get("labels")
        thresholds = metadata.get("thresholds")
        encoder_name = metadata.get("encoder")
        if (
            metadata.get("artifact_version") != 2
            or metadata.get("task") != "multi_label_emotion_classification"
            or not isinstance(classifiers, dict)
            or not isinstance(labels, list)
            or not isinstance(thresholds, dict)
            or not isinstance(encoder_name, str)
        ):
            raise ValueError(
                "The Empwave model artifact is not a version 2 emotion "
                f"classifier: {self.artifact_path}"
            )

        unsupported = set(labels) - set(EMOTION_REGION_WEIGHTS)
        if unsupported:
            raise ValueError(
                "The trained model contains unsupported emotions: "
                f"{sorted(unsupported)}"
            )

        self.metadata = metadata
        self.emotion_classifiers = classifiers
        self.emotion_labels = labels
        self.emotion_thresholds = thresholds
        self.runtime_min_confidence = float(
            metadata.get("inference", {}).get(
                "runtime_min_confidence",
                EMOTION_RUNTIME_MIN_CONFIDENCE,
            )
        )
        inference_config = metadata.get("inference", {})
        self.neutral_competition_margin = float(
            inference_config.get("neutral_competition_margin", 0.15)
        )
        self.neutral_override_confidence = float(
            inference_config.get("neutral_override_confidence", 0.95)
        )
        self.max_emotions = int(
            inference_config.get("max_emotions", 5)
        )
        self.relative_fallback_min_confidence = float(
            inference_config.get(
                "relative_fallback_min_confidence",
                0.72,
            )
        )
        self.relative_fallback_band = float(
            inference_config.get("relative_fallback_band", 0.06)
        )
        self.max_relative_fallback_emotions = int(
            inference_config.get("max_relative_fallback_emotions", 3)
        )
        self.intent_classifier = SemanticIntentClassifier(encoder_name)
        self.model = self.intent_classifier.model
        self.model_name = "empwave-emotions-v2+semantic-intents"

    @staticmethod
    def _predict_probabilities(model, embeddings):
        if model.get("type") == "constant":
            return np.full(
                embeddings.shape[0],
                float(model["probability"]),
                dtype=np.float32,
            )
        return model["estimator"].predict_proba(embeddings)[:, 1]

    def _detect_emotions(self, text):
        segments = _split_clauses(text)
        embeddings = self.model.encode(
            segments,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        detected = []
        candidates = []
        neutral = None
        for emotion_id in self.emotion_labels:
            probabilities = self._predict_probabilities(
                self.emotion_classifiers[emotion_id],
                embeddings,
            )
            segment_index = (
                0
                if emotion_id == "neutral"
                else int(np.argmax(probabilities))
            )
            probability = float(probabilities[segment_index])
            threshold = float(self.emotion_thresholds[emotion_id])
            if emotion_id != "neutral":
                threshold = max(
                    threshold,
                    self.runtime_min_confidence,
                )
            result = {
                "id": emotion_id,
                "label": emotion_id.replace("_", " ").title(),
                "score": round(probability, 4),
                "threshold": round(threshold, 4),
                "salience": round(
                    max(
                        0.0,
                        min(
                            1.0,
                            (probability - threshold)
                            / max(1.0 - threshold, 0.01),
                        ),
                    ),
                    4,
                ),
                "evidence": segments[segment_index],
            }
            if emotion_id == "neutral":
                neutral = result
            else:
                candidates.append(result)
                if probability >= threshold:
                    detected.append(result)

        detected.sort(key=lambda emotion: -emotion["score"])
        if neutral and neutral["score"] >= neutral["threshold"]:
            detected = [
                emotion
                for emotion in detected
                if (
                    emotion["score"] >= self.neutral_override_confidence
                    or emotion["score"]
                    >= neutral["score"] + self.neutral_competition_margin
                )
            ]
        if detected:
            return detected[:self.max_emotions]
        if neutral and neutral["score"] >= neutral["threshold"]:
            return [neutral]
        relative_candidates = [
            emotion
            for emotion in candidates
            if emotion["score"] >= self.relative_fallback_min_confidence
        ]
        if relative_candidates:
            relative_candidates.sort(
                key=lambda emotion: -emotion["score"]
            )
            best_score = relative_candidates[0]["score"]
            return [
                {
                    **emotion,
                    "inference_mode": "relative_confidence",
                    "salience": round(
                        (
                            emotion["score"]
                            - self.relative_fallback_min_confidence
                        )
                        / max(
                            1.0 - self.relative_fallback_min_confidence,
                            0.01,
                        ),
                        4,
                    ),
                }
                for emotion in relative_candidates
                if emotion["score"]
                >= best_score - self.relative_fallback_band
            ][:self.max_relative_fallback_emotions]
        return []

    @staticmethod
    def _intent_results(semantic_analysis):
        return [
            {
                "id": region["id"],
                "label": region["intent"],
                "score": region["score"],
                "evidence": region["evidence"],
                "reason": region["reason"],
                "semantic_concept": region["semantic_concept"],
                "strength": region["strength"],
            }
            for region in semantic_analysis["regions"]
            if region["id"] != "temporal_l"
        ]

    @staticmethod
    def _add_contribution(contributions, region_id, source, weight):
        contribution = max(0.0, min(1.0, source["score"] * weight))
        if contribution <= 0:
            return
        contributions[region_id].append(
            {
                "type": source["type"],
                "id": source["id"],
                "label": source["label"],
                "score": source["score"],
                "weight": weight,
                "contribution": contribution,
            }
        )

    def _derive_regions(self, text, emotions, intents):
        contributions = {region_id: [] for region_id in REGION_CONFIG}
        self._add_contribution(
            contributions,
            "temporal_l",
            {
                "type": "baseline",
                "id": "spoken_input",
                "label": "Auditory comprehension",
                "score": 0.62,
            },
            1.0,
        )
        primary_emotion_id = (
            emotions[0]["id"]
            if emotions and emotions[0]["id"] != "neutral"
            else None
        )
        for emotion in emotions:
            source = {**emotion, "type": "emotion"}
            for region_id, weight in EMOTION_REGION_WEIGHTS[
                emotion["id"]
            ].items():
                if (
                    region_id == "brainstem"
                    and emotion["id"] != primary_emotion_id
                    and emotion["score"] < 0.99
                ):
                    continue
                self._add_contribution(
                    contributions,
                    region_id,
                    source,
                    weight,
                )

        for intent in intents:
            source = {**intent, "type": "intent"}
            for region_id, weight in INTENT_REGION_WEIGHTS[
                intent["id"]
            ].items():
                self._add_contribution(
                    contributions,
                    region_id,
                    source,
                    weight,
                )

        regions = []
        for region_id, sources in contributions.items():
            if not sources:
                continue
            remaining = 1.0
            for source in sources:
                remaining *= 1.0 - source["contribution"]
            score = 1.0 - remaining
            threshold = REGION_ACTIVATION_THRESHOLDS[region_id]
            if score < threshold:
                continue

            sources.sort(key=lambda source: -source["contribution"])
            source_summary = ", ".join(
                f"{source['label']} {round(source['score'] * 100)}%"
                for source in sources[:3]
            )
            normalized = (score - threshold) / max(1.0 - threshold, 0.01)
            regions.append(
                {
                    "id": region_id,
                    "intent": INTENT_LABELS[region_id],
                    "score": round(score, 4),
                    "evidence": text,
                    "reason": (
                        f"Derived from {source_summary}. "
                        f"{REGION_CONFIG[region_id]['reason']}"
                    ),
                    "semantic_concept": (
                        "Weighted emotion and intent evidence: "
                        + source_summary
                    ),
                    "strength": round(
                        max(0.55, min(1.0, 0.55 + normalized * 0.45)),
                        3,
                    ),
                    "sources": sources,
                }
            )

        regions.sort(
            key=lambda region: (
                ACTIVATION_STAGE[region["id"]],
                -region["score"],
            )
        )
        return regions

    @staticmethod
    def _primary_interpretation(text, emotions, intents, regions):
        region_by_id = {region["id"]: region for region in regions}
        cognitive_intents = [
            intent for intent in intents if intent["id"] != "amygdala"
        ]
        non_neutral_emotions = [
            emotion for emotion in emotions if emotion["id"] != "neutral"
        ]

        if cognitive_intents:
            strongest_intent = max(
                cognitive_intents,
                key=lambda intent: intent["strength"],
            )
        else:
            strongest_intent = None
        strongest_emotion = (
            max(
                non_neutral_emotions,
                key=lambda emotion: emotion["salience"],
            )
            if non_neutral_emotions
            else None
        )

        if (
            strongest_emotion
            and (
                not strongest_intent
                or strongest_emotion["salience"]
                >= strongest_intent["strength"]
            )
        ):
            primary_emotion = strongest_emotion
            primary_label = f"{primary_emotion['label']} emotional response"
            primary_region_id = max(
                EMOTION_REGION_WEIGHTS[primary_emotion["id"]],
                key=EMOTION_REGION_WEIGHTS[primary_emotion["id"]].get,
            )
            primary_region = region_by_id[primary_region_id]
        elif strongest_intent:
            primary_label = strongest_intent["label"]
            primary_region = region_by_id[strongest_intent["id"]]
        elif intents:
            primary_intent = max(intents, key=lambda intent: intent["score"])
            primary_label = primary_intent["label"]
            primary_region = region_by_id[primary_intent["id"]]
        else:
            primary_region = region_by_id["temporal_l"]
            primary_label = primary_region["intent"]

        return {
            "label": primary_label,
            "region_id": primary_region["id"],
            "score": primary_region["score"],
            "evidence": text,
            "reason": primary_region["reason"],
            "semantic_concept": primary_region["semantic_concept"],
        }

    def classify(self, text):
        semantic_analysis = self.intent_classifier.classify(text)
        emotions = self._detect_emotions(text)
        intents = self._intent_results(semantic_analysis)
        regions = self._derive_regions(text, emotions, intents)
        primary = self._primary_interpretation(
            text,
            emotions,
            intents,
            regions,
        )
        return {
            "model": self.model_name,
            "encoder": self.metadata["encoder"],
            "baseline_only": len(regions) == 1,
            "mapping_basis": (
                "Illustrative weighted mapping from detected emotions and "
                "semantic intents; not neuroimaging ground truth."
            ),
            "emotions": emotions,
            "intents": intents,
            "primary_intent": primary,
            "regions": regions,
        }


@lru_cache(maxsize=1)
def get_classifier():
    if TRAINED_MODEL_PATH.is_file():
        return LayeredNlpClassifier(TRAINED_MODEL_PATH)
    LOGGER.warning(
        "Trained Empwave artifact not found at %s; using semantic classifier.",
        TRAINED_MODEL_PATH,
    )
    return SemanticIntentClassifier()
