import json
import os
import unittest
from pathlib import Path

import responses
from dotenv import load_dotenv

from roboflow.models.keypoint_detection import KeypointDetectionModel
from roboflow.util.prediction import PredictionGroup

load_dotenv(Path("../../.env"))


with open(Path("tests/annotations/keypoint-detection-annotations/MM2A_46_R_T_predictions.json")) as f:
    MOCK_RESPONSE = json.load(f)


class TestKeypointDetection(unittest.TestCase):
    api_key = os.getenv("ROBOFLOW_API_KEY", "test-api-key")
    workspace = os.getenv("WORKSPACE_ID")
    dataset_id = os.getenv("PROJECT_NAME")
    version = "1"

    api_url = f"https://detect.roboflow.com/{dataset_id}/{version}"

    _default_params = {"api_key": api_key, "confidence": "40", "name": "YOUR_IMAGE.jpg"}

    def setUp(self):
        super().setUp()
        self.version_id = f"{self.workspace}/{self.dataset_id}/{self.version}"

    def test_init_sets_attributes(self):
        instance = KeypointDetectionModel(self.api_key, self.version_id, version=self.version)

        self.assertEqual(instance.id, self.version_id)
        self.assertEqual(instance.version, self.version)
        self.assertEqual(instance.base_url, "https://detect.roboflow.com/")

    @responses.activate
    def test_predict_local_image(self):
        instance = KeypointDetectionModel(self.api_key, self.version_id, version=self.version)

        responses.add(responses.POST, self.api_url, json=MOCK_RESPONSE, status=200)

        result = instance.predict("tests/images/MM2A_46_R_T.png")

        self.assertIsInstance(result, PredictionGroup)
        self.assertEqual(len(result.predictions), 1)

    @responses.activate
    def test_predict_with_confidence(self):
        instance = KeypointDetectionModel(self.api_key, self.version_id, version=self.version)

        responses.add(responses.POST, self.api_url, json=MOCK_RESPONSE, status=200)

        result = instance.predict("tests/images/MM2A_46_R_T.png", confidence=30)

        self.assertIsInstance(result, PredictionGroup)
        request = responses.calls[0].request
        self.assertEqual(request.params["confidence"], "30")

    @responses.activate
    def test_predict_error_response(self):
        instance = KeypointDetectionModel(self.api_key, self.version_id, version=self.version)

        responses.add(responses.POST, self.api_url, json={"error": "Invalid API key"}, status=401)

        with self.assertRaises(Exception):
            instance.predict("tests/images/MM2A_46_R_T.png")
