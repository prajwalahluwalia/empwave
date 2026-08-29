import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from app import app
from empwave import create_app
from empwave.services.feedback_store import FeedbackStore
from empwave.services.intent_classifier import get_classifier


class SemanticIntentClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classifier = get_classifier()

    def assert_regions(self, text, expected):
        result = self.classifier.classify(text)
        self.assertEqual(
            [region["id"] for region in result["regions"]],
            expected,
        )

    def test_conceptual_balance_is_reasoning(self):
        self.assert_regions(
            "I cannot balance my sports and work",
            ["temporal_l", "prefrontal"],
        )

    def test_literal_balance_is_cerebellar(self):
        self.assert_regions(
            "I balance on a tightrope",
            ["temporal_l", "cerebellum"],
        )

    def test_negative_evaluation_has_emotion_and_appraisal(self):
        self.assert_regions(
            "The food was terrible",
            ["temporal_l", "amygdala", "prefrontal"],
        )

    def test_physical_trauma_has_listener_response(self):
        self.assert_regions(
            "I got hit by a car",
            ["temporal_l", "amygdala", "parietal", "prefrontal"],
        )

    def test_neutral_statement_is_auditory_baseline(self):
        self.assert_regions(
            "The food is on the table",
            ["temporal_l"],
        )

    def test_unseen_food_opinion_has_evaluation(self):
        self.assert_regions(
            "That meal tasted revolting",
            ["temporal_l", "amygdala", "prefrontal"],
        )

    def test_unseen_physical_balance_paraphrase(self):
        self.assert_regions(
            "I am trying to stay upright on a skateboard",
            ["temporal_l", "cerebellum"],
        )

    def test_first_nlp_project_triggers_novelty_not_speech_production(self):
        result = self.classifier.classify("This is my first NLP project")
        self.assertEqual(
            [region["id"] for region in result["regions"]],
            ["temporal_l", "prefrontal"],
        )
        self.assertEqual(
            result["primary_intent"]["label"],
            "Novelty and curiosity",
        )

    def test_plain_project_statement_remains_neutral(self):
        self.assert_regions(
            "This is a project",
            ["temporal_l"],
        )

    def test_explicit_speech_request_triggers_broca(self):
        self.assert_regions(
            "Please explain this aloud",
            ["temporal_l", "broca"],
        )

    def test_hostile_phrase_maps_emotion_to_multiple_regions(self):
        result = self.classifier.classify("fuck you")
        self.assertEqual(
            [region["id"] for region in result["regions"]],
            ["temporal_l", "amygdala", "prefrontal"],
        )
        self.assertEqual(result["emotions"][0]["id"], "anger")
        self.assertEqual(
            result["primary_intent"]["region_id"],
            "amygdala",
        )

    def test_short_negative_judgment_uses_relative_emotion_evidence(self):
        result = self.classifier.classify("Ashlay sucks")
        self.assertEqual(
            [region["id"] for region in result["regions"]],
            ["temporal_l", "amygdala", "prefrontal"],
        )
        self.assertEqual(
            {emotion["id"] for emotion in result["emotions"]},
            {"disgust", "disappointment", "anger"},
        )

    def test_mixed_emotions_are_detected_independently(self):
        result = self.classifier.classify(
            "I feel happy but nervous about starting my new job"
        )
        emotions = {emotion["id"] for emotion in result["emotions"]}
        self.assertIn("joy", emotions)
        self.assertIn("nervousness", emotions)
        self.assertEqual(
            {region["id"] for region in result["regions"]},
            {"temporal_l", "amygdala", "prefrontal", "brainstem"},
        )

    def test_rich_sentence_can_activate_all_regions(self):
        text = (
            "I decide how to explain a childhood memory aloud while I watch "
            "a bright light, listen to music, touch a cold rough surface, "
            "run while balancing, breathe slowly, and feel excited."
        )
        result = self.classifier.classify(text)
        self.assertEqual(
            {region["id"] for region in result["regions"]},
            {
                "prefrontal",
                "broca",
                "motor",
                "parietal",
                "occipital",
                "temporal_l",
                "temporal_r",
                "amygdala",
                "cerebellum",
                "brainstem",
            },
        )


class ProcessSpeechApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        app.config["ENABLE_EMOTION_FEEDBACK"] = True
        app.config["FEEDBACK_DB_PATH"] = str(
            Path(self.temporary_directory.name) / "feedback.sqlite3"
        )
        self.client = app.test_client()

    def tearDown(self):
        app.config["ENABLE_EMOTION_FEEDBACK"] = False
        self.temporary_directory.cleanup()

    def test_process_speech_returns_structured_nlp_result(self):
        response = self.client.post(
            "/api/process-speech",
            json={"text": "The food was terrible"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "success")
        self.assertEqual(
            payload["model"],
            "empwave-emotions-v2+semantic-intents",
        )
        self.assertGreaterEqual(len(payload["emotions"]), 2)
        self.assertGreaterEqual(len(payload["intents"]), 1)
        self.assertIn("mapping_basis", payload)
        self.assertEqual(
            payload["primary_intent"]["label"],
            "Disgust emotional response",
        )
        self.assertEqual(
            payload["primary_intent"]["region_id"],
            "amygdala",
        )
        self.assertEqual(
            [region["id"] for region in payload["regions"]],
            ["temporal_l", "amygdala", "prefrontal"],
        )

    def test_process_speech_rejects_empty_text(self):
        response = self.client.post("/api/process-speech", json={"text": " "})
        self.assertEqual(response.status_code, 400)

    def test_simulate_returns_emotional_regions_with_triggers(self):
        text = "I am terrified of the dark"
        response = self.client.post("/simulate", json={"text": text})
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertFalse(payload["fallback"])
        self.assertTrue(payload["spoken_text"])
        self.assertIn("analysis", payload)
        self.assertIn(
            "amygdala",
            {region["id"] for region in payload["regions"]},
        )
        allowed_ids = {
            "prefrontal",
            "broca",
            "motor",
            "parietal",
            "occipital",
            "temporal_l",
            "temporal_r",
            "amygdala",
            "cerebellum",
            "brainstem",
        }
        for region in payload["regions"]:
            self.assertEqual(
                set(region),
                {"id", "strength", "trigger"},
            )
            self.assertIn(region["id"], allowed_ids)
            self.assertGreaterEqual(region["strength"], 0.0)
            self.assertLessEqual(region["strength"], 1.0)
            self.assertIn(region["trigger"], text)

    def test_simulate_returns_movement_region(self):
        response = self.client.post(
            "/simulate",
            json={"text": "I balance on a tightrope"},
        )
        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertFalse(payload["fallback"])
        self.assertEqual(
            [region["id"] for region in payload["regions"]],
            ["cerebellum"],
        )

    def test_simulate_returns_fallback_for_factual_text(self):
        response = self.client.post(
            "/simulate",
            json={"text": "The food is on the table"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["regions"], [])
        self.assertTrue(response.get_json()["fallback"])

    def test_simulate_returns_fallback_for_ambiguous_text(self):
        response = self.client.post(
            "/simulate",
            json={"text": "This is a project"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["regions"], [])
        self.assertTrue(response.get_json()["fallback"])

    def test_health_reports_preloaded_model(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json(),
            {
                "model": "empwave-emotions-v2+semantic-intents",
                "model_loaded": True,
                "status": "ok",
            },
        )
        self.assertIs(app.extensions["empwave_classifier"], get_classifier())

    def test_simulate_cors_allows_configured_origin(self):
        allowed_origin = "https://frontend.example"
        cors_app = create_app(
            {
                "TESTING": True,
                "ALLOWED_ORIGIN": allowed_origin,
            }
        )
        response = cors_app.test_client().post(
            "/simulate",
            json={"text": "I balance on a tightrope"},
            headers={"Origin": allowed_origin},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers.get("Access-Control-Allow-Origin"),
            allowed_origin,
        )

    def test_simulate_cors_rejects_other_origins(self):
        cors_app = create_app(
            {
                "TESTING": True,
                "ALLOWED_ORIGIN": "https://frontend.example",
            }
        )
        response = cors_app.test_client().post(
            "/simulate",
            json={"text": "I balance on a tightrope"},
            headers={"Origin": "https://attacker.example"},
        )
        self.assertEqual(response.status_code, 403)
        self.assertIsNone(
            response.headers.get("Access-Control-Allow-Origin")
        )

    def test_emotion_feedback_requires_consent(self):
        response = self.client.post(
            "/api/emotion-feedback",
            json={
                "text": "I feel nervous",
                "selected_emotions": ["nervousness"],
                "consent": False,
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("consent", response.get_json()["message"].lower())

    def test_emotion_feedback_can_be_disabled(self):
        app.config["ENABLE_EMOTION_FEEDBACK"] = False
        response = self.client.post(
            "/api/emotion-feedback",
            json={
                "text": "I feel nervous",
                "selected_emotions": ["nervousness"],
                "consent": True,
            },
        )
        self.assertEqual(response.status_code, 404)

    def test_emotion_feedback_is_stored_for_review(self):
        response = self.client.post(
            "/api/emotion-feedback",
            json={
                "text": "I feel happy but nervous",
                "selected_emotions": ["joy", "nervousness"],
                "consent": True,
            },
        )
        self.assertEqual(response.status_code, 201)
        payload = response.get_json()
        self.assertEqual(payload["status"], "success")

        with closing(
            sqlite3.connect(app.config["FEEDBACK_DB_PATH"])
        ) as connection:
            record = connection.execute(
                """
                SELECT perspective, selected_emotions, moderation_status
                FROM emotion_feedback
                WHERE id = ?
                """,
                (payload["feedback_id"],),
            ).fetchone()
        self.assertEqual(record[0], "speaker_self_report")
        self.assertEqual(
            json.loads(record[1]),
            ["joy", "nervousness"],
        )
        self.assertIn(
            record[2],
            {"pending_review", "priority_review"},
        )
        FeedbackStore(app.config["FEEDBACK_DB_PATH"]).review(
            payload["feedback_id"],
            "approved",
            "Verified self-report.",
        )
        with closing(
            sqlite3.connect(app.config["FEEDBACK_DB_PATH"])
        ) as connection:
            reviewed_status = connection.execute(
                """
                SELECT moderation_status FROM emotion_feedback
                WHERE id = ?
                """,
                (payload["feedback_id"],),
            ).fetchone()[0]
        self.assertEqual(reviewed_status, "approved")

    def test_duplicate_feedback_is_flagged_not_auto_accepted(self):
        submission = {
            "text": "This situation makes me angry",
            "selected_emotions": ["anger"],
            "consent": True,
        }
        first = self.client.post("/api/emotion-feedback", json=submission)
        second = self.client.post("/api/emotion-feedback", json=submission)
        self.assertEqual(first.status_code, 201)
        self.assertEqual(second.status_code, 201)
        payload = second.get_json()
        self.assertEqual(payload["status"], "success")
        self.assertIn(
            "duplicate_feedback",
            payload["moderation_flags"],
        )
        self.assertEqual(payload["moderation_status"], "priority_review")

    def test_neutral_cannot_be_combined_with_other_emotions(self):
        response = self.client.post(
            "/api/emotion-feedback",
            json={
                "text": "I am uncertain",
                "selected_emotions": ["neutral", "confusion"],
                "consent": True,
            },
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
