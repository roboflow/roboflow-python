import unittest

import responses
from requests.exceptions import HTTPError

from roboflow.config import INSTANCE_SEGMENTATION_URL
from roboflow.models.instance_segmentation import InstanceSegmentationModel
from roboflow.util.prediction import PredictionGroup

MOCK_RESPONSE = {
    "predictions": [
        {
            "x": 812.0,
            "y": 362.9,
            "width": 277,
            "height": 206,
            "class": "J",
            "confidence": 0.598,
            "points": [
                {"x": 831.0, "y": 527.0},
                {"x": 931.0, "y": 389.0},
                {"x": 831.0, "y": 527.0},
            ],
        },
        {
            "x": 363.8,
            "y": 665.5,
            "width": 707,
            "height": 669,
            "class": "K",
            "confidence": 0.52,
            "points": [
                {"x": 131.0, "y": 999.0},
                {"x": 269.0, "y": 666.0},
                {"x": 131.0, "y": 999.0},
            ],
        },
    ],
    "image": {"width": 1333, "height": 1000},
}


class TestInstanceSegmentation(unittest.TestCase):
    api_key = "my-api-key"
    workspace = "roboflow"
    dataset_id = "test-123"
    version = "23"

    api_url = f"https://outline.roboflow.com/{dataset_id}/{version}"

    _default_params = {
        "api_key": api_key,
        "confidence": "40",
    }

    def setUp(self):
        super().setUp()
        self.version_id = f"{self.workspace}/{self.dataset_id}/{self.version}"

    def test_init_sets_attributes(self):
        instance = InstanceSegmentationModel(self.api_key, self.version_id)

        self.assertEqual(instance.id, self.version_id)
        self.assertEqual(
            instance.api_url,
            f"{INSTANCE_SEGMENTATION_URL}/{self.dataset_id}/{self.version}",
        )

    @responses.activate
    def test_predict_returns_prediction_group(self):
        image_path = "tests/images/rabbit.JPG"
        instance = InstanceSegmentationModel(self.api_key, self.version_id)

        responses.add(responses.POST, self.api_url, json=MOCK_RESPONSE)

        group = instance.predict(image_path)

        self.assertIsInstance(group, PredictionGroup)

    @responses.activate
    def test_predict_with_local_image_request(self):
        image_path = "tests/images/rabbit.JPG"
        instance = InstanceSegmentationModel(self.api_key, self.version_id)

        responses.add(responses.POST, self.api_url, json=MOCK_RESPONSE)

        instance.predict(image_path)

        request = responses.calls[0].request

        self.assertEqual(request.method, "POST")
        self.assertRegex(request.url, rf"^{self.api_url}")
        self.assertDictEqual(request.params, self._default_params)
        self.assertIsNotNone(request.body)

    @responses.activate
    def test_predict_with_hosted_image_request(self):
        image_path = "https://example.com/raccoon.JPG"
        expected_params = {
            **self._default_params,
            "image": image_path,
        }
        instance = InstanceSegmentationModel(self.api_key, self.version_id)

        # Mock the library validating that the URL is valid before sending to the API
        responses.add(responses.HEAD, image_path)
        responses.add(responses.POST, self.api_url, json=MOCK_RESPONSE)

        instance.predict(image_path)

        request = responses.calls[1].request

        self.assertEqual(request.method, "POST")
        self.assertRegex(request.url, rf"^{self.api_url}")
        self.assertDictEqual(request.params, expected_params)
        self.assertIsNone(request.body)

    @responses.activate
    def test_predict_with_confidence_request(self):
        confidence = "100"
        image_path = "tests/images/rabbit.JPG"
        expected_params = {**self._default_params, "confidence": confidence}
        instance = InstanceSegmentationModel(self.api_key, self.version_id)

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

        instance = InstanceSegmentationModel(self.api_key, self.version_id)

        with self.assertRaises(HTTPError):
            instance.predict(image_path)
