import unittest

import responses
from PIL import UnidentifiedImageError
from requests.exceptions import HTTPError

from roboflow.config import OBJECT_DETECTION_URL
from roboflow.models.object_detection import ObjectDetectionModel
from roboflow.util.prediction import PredictionGroup

MOCK_RESPONSE = {
    "predictions": [
        {
            "x": 189.5,
            "y": 100,
            "width": 163,
            "height": 186,
            "class": "helmet",
            "confidence": 0.544,
        }
    ],
    "image": {"width": 2048, "height": 1371},
}


class TestObjectDetection(unittest.TestCase):
    api_key = "my-api-key"
    workspace = "roboflow"
    dataset_id = "test-123"
    version = "23"

    api_url = f"{OBJECT_DETECTION_URL}/{dataset_id}/{version}"

    _default_params = {
        "api_key": api_key,
        "confidence": "40",
        "format": "json",
        "labels": "false",
        "name": "YOUR_IMAGE.jpg",
        "overlap": "30",
        "stroke": "1",
    }

    def setUp(self):
        super().setUp()
        self.version_id = f"{self.workspace}/{self.dataset_id}/{self.version}"

    def test_init_sets_attributes(self):
        instance = ObjectDetectionModel(self.api_key, self.version_id, version=self.version)

        self.assertEqual(instance.id, self.version_id)
        # self.assertEqual(instance.api_url,
        # f"{OBJECT_DETECTION_URL}/{self.dataset_id}/{self.version}")

    @responses.activate
    def test_predict_returns_prediction_group(self):
        print(self.api_url)
        image_path = "tests/images/rabbit.JPG"
        instance = ObjectDetectionModel(self.api_key, self.version_id, version=self.version)

        responses.add(responses.POST, self.api_url, json=MOCK_RESPONSE)

        group = instance.predict(image_path)

        self.assertIsInstance(group, PredictionGroup)

    @responses.activate
    def test_predict_with_local_image_request(self):
        image_path = "tests/images/rabbit.JPG"
        instance = ObjectDetectionModel(self.api_key, self.version_id, version=self.version)

        responses.add(responses.POST, self.api_url, json=MOCK_RESPONSE)

        instance.predict(image_path)

        request = responses.calls[0].request

        self.assertEqual(request.method, "POST")
        self.assertRegex(request.url, rf"^{self.api_url}")
        self.assertDictEqual(request.params, self._default_params)
        self.assertIsNotNone(request.body)

    @responses.activate
    def test_predict_with_a_numpy_array_request(self):
        import numpy as np

        np_array = np.ones((100, 100, 1), dtype=np.uint8)
        instance = ObjectDetectionModel(self.api_key, self.version_id, version=self.version)

        responses.add(responses.POST, self.api_url, json=MOCK_RESPONSE)

        instance.predict(np_array)

        request = responses.calls[0].request

        self.assertEqual(request.method, "POST")
        self.assertRegex(request.url, rf"^{self.api_url}")
        self.assertDictEqual(request.params, self._default_params)
        self.assertIsNotNone(request.body)

    def test_predict_with_local_wrong_image_request(self):
        image_path = "tests/images/not_an_image.txt"
        instance = ObjectDetectionModel(self.api_key, self.version_id, version=self.version)
        self.assertRaises(UnidentifiedImageError, instance.predict, image_path)

    @responses.activate
    def test_predict_with_hosted_image_request(self):
        image_path = "https://example.com/racoon.JPG"
        expected_params = {
            **self._default_params,
            "image": image_path,
        }
        instance = ObjectDetectionModel(self.api_key, self.version_id, version=self.version)

        # Mock the library validating that the URL is valid before sending to the API
        responses.add(responses.POST, self.api_url, json=MOCK_RESPONSE)

        instance.predict(image_path, hosted=True)

        request = responses.calls[0].request

        self.assertEqual(request.method, "POST")
        self.assertRegex(request.url, rf"^{self.api_url}")
        self.assertDictEqual(request.params, expected_params)
        self.assertIsNone(request.body)

    @responses.activate
    def test_predict_with_confidence_request(self):
        confidence = "100"
        image_path = "tests/images/rabbit.JPG"
        expected_params = {**self._default_params, "confidence": confidence}
        instance = ObjectDetectionModel(self.api_key, self.version_id, version=self.version)

        responses.add(responses.POST, self.api_url, json=MOCK_RESPONSE)

        instance.predict(image_path, confidence=confidence)

        request = responses.calls[0].request

        self.assertEqual(request.method, "POST")
        self.assertRegex(request.url, rf"^{self.api_url}")
        self.assertDictEqual(request.params, expected_params)
        self.assertIsNotNone(request.body)

    @responses.activate
    def test_predict_with_non_200_response_raises_http_error(self):
        image_path = "tests/images/rabbit.JPG"
        responses.add(responses.POST, self.api_url, status=403)

        instance = ObjectDetectionModel(self.api_key, self.version_id, version=self.version)

        with self.assertRaises(HTTPError):
            instance.predict(image_path)
