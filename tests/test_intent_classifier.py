import unittest

from app import app
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
        self.client = app.test_client()

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


if __name__ == "__main__":
    unittest.main()
