import unittest

from requests.exceptions import HTTPError
import responses

from roboflow.config import SEMANTIC_SEGMENTATION_URL
from roboflow.models.semantic_segmentation import SemanticSegmentationModel
from roboflow.util.prediction import PredictionGroup


MOCK_RESPONSE = {

    "segmentation_mask": "iVBORw0KGgoAAAANSUhEUgAAAgAAAAIACAAAAADRE4smAAACjElEQVR4nO3bzXKbMBiGUanT+79ldVHXwSmmFmJGfcU5i8SZbDR8DzL4pxQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgKXUOnsFTGX+AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMuosxewklpKm72GXj9mLyBf/etBEgGMqqVGTv5BADcngEF151GSn7MXkO116HFXgMUOMCZ//gK4TuT8BTCkvfyKJICbE8CQ9vyRSgBDMm/9tgQwLHoDWCDhaWopj+nXkpuBHeBT9dtL/t9OndQzSQCf2j/Fa8mdfSlFAD3a3l+H20IAAXzszbN83fw3b/4COO9PEIFT3xDAeW2zJ6TeBHg7eEjLvgUsxQ4wrGXPXwDDoscvgHE1+ypQAGfU/U8CJpaQuObpvt4FeBy/9vIoigD6fR2z9nwZaPtyUBQB9Ds6ZnEFuAa4OQF0O9w043ZUAdxcXLGT/eN4xV0C2AH6LDd/AVwpcP4CuFDi/AVwocjrKQF0iTzJDwmgz3IFCKDTagUI4OYEcJ3IzUEAnSIv9Q/4VHCXd+NvsWGkrnum9xUE8hTQL3LQ7wjghJUKEMBlMrMQwBm7s868nMpc9X/iefB+fzc8cgtwGzjg8XWAyMFzOZspAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACwil+AQDJrnsYcnwAAAABJRU5ErkJggg==",
    "class_map": {
        "0": "background",
        "1": "object"
    },
    "image": {
        "width": 800,
        "height": 600
    }
}


class TestSemanticSegmentation(unittest.TestCase):
    api_key = "my-api-key"
    workspace = "roboflow"
    dataset_id = "test-123"
    version = "23"

    api_url = f"https://segment.roboflow.com/{dataset_id}/{version}"

    _default_params = { "api_key": api_key, "confidence": "50" }

    version_id = f"{workspace}/{dataset_id}/{version}"

    def test_init_sets_attributes(self):
        instance = SemanticSegmentationModel(self.api_key, self.version_id)

        self.assertEqual(instance.id, self.version_id)
        self.assertEqual(instance.api_url, f"{SEMANTIC_SEGMENTATION_URL}/{self.dataset_id}/{self.version}")

    @responses.activate
    def test_predict_returns_prediction_group(self):
        image_path = "tests/images/rabbit.JPG"
        instance = SemanticSegmentationModel(self.api_key, self.version_id)

        responses.add(responses.POST, self.api_url, json=MOCK_RESPONSE)

        group = instance.predict(image_path)

        self.assertIsInstance(group, PredictionGroup)

    @responses.activate
    def test_predict_with_local_image_request(self):
        image_path = "tests/images/rabbit.JPG"
        instance = SemanticSegmentationModel(self.api_key, self.version_id)

        responses.add(responses.POST, self.api_url, json=MOCK_RESPONSE)

        instance.predict(image_path)

        request = responses.calls[0].request

        self.assertEqual(request.method, 'POST')
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
        instance = SemanticSegmentationModel(self.api_key, self.version_id)

        # Mock the library validating that the URL is valid before sending to the API
        responses.add(responses.HEAD, image_path)
        responses.add(responses.POST, self.api_url, json=MOCK_RESPONSE)

        instance.predict(image_path)

        request = responses.calls[1].request

        self.assertEqual(request.method, 'POST')
        self.assertRegex(request.url, rf"^{self.api_url}")
        self.assertDictEqual(request.params, expected_params)
        self.assertIsNone(request.body)

    @responses.activate
    def test_predict_with_confidence_request(self):
        confidence = "100"
        image_path = "tests/images/rabbit.JPG"
        expected_params = {
            **self._default_params,
            "confidence": confidence
        }
        instance = SemanticSegmentationModel(self.api_key, self.version_id)

        responses.add(responses.POST, self.api_url, json=MOCK_RESPONSE)

        instance.predict(image_path, confidence=confidence)

        request = responses.calls[0].request

        self.assertEqual(request.method, 'POST')
        self.assertRegex(request.url, rf"^{self.api_url}")
        self.assertDictEqual(request.params, expected_params)
        self.assertIsNotNone(request.body)

    @responses.activate
    def test_predict_with_non_200_response_raises_http_error(self):
        image_path = "tests/images/rabbit.JPG"
        responses.add(responses.POST, self.api_url, status=403)

        instance = SemanticSegmentationModel(self.api_key, self.version_id)

        with self.assertRaises(HTTPError):
            instance.predict(image_path)
