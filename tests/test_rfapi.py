import json
import os
import unittest
import urllib
from unittest.mock import mock_open, patch

import responses

from roboflow.adapters.rfapi import (
    RoboflowError,
    create_training_v2,
    get_train_recipe,
    get_training,
    list_trainings_for_version,
    upload_image,
)
from roboflow.config import API_URL, DEFAULT_BATCH_NAME


class TestUploadImage(unittest.TestCase):
    API_KEY = "test_api_key"
    PROJECT_URL = "test_project"
    SEQUENCE_NUMBER = 1
    SEQUENCE_SIZE = 10
    TAG_NAMES_LOCAL = ["lonely-tag"]
    TAG_NAMES_HOSTED = ["tag1", "tag2"]
    IMAGE_PATH_LOCAL = "test_image.jpg"
    IMAGE_PATH_HOSTED = "http://example.com/test_image.jpg"
    IMAGE_NAME_HOSTED = os.path.basename(IMAGE_PATH_HOSTED)

    @responses.activate
    @patch("roboflow.adapters.rfapi.open", new_callable=mock_open, read_data=b"image_data")
    def test_upload_image_local(self, _mock_file):
        scenarios = [
            {
                "desc": "with batch_name",
                "batch_name": "My personal batch",
                "expected_url": (
                    f"{API_URL}/dataset/{self.PROJECT_URL}/upload?"
                    f"api_key={self.API_KEY}&batch=My%20personal%20batch"
                    f"&sequence_number=1&sequence_size=10&tag=lonely-tag"
                ),
            },
            {
                "desc": "without batch_name",
                "expected_url": (
                    f"{API_URL}/dataset/{self.PROJECT_URL}/upload?"
                    f"api_key={self.API_KEY}&batch={urllib.parse.quote_plus(DEFAULT_BATCH_NAME)}"
                    f"&sequence_number=1&sequence_size=10&tag=lonely-tag"
                ),
            },
            {
                "desc": "without batch_name",
                "batch_name": None,
                "expected_url": (
                    f"{API_URL}/dataset/{self.PROJECT_URL}/upload?"
                    f"api_key={self.API_KEY}&batch={urllib.parse.quote_plus(DEFAULT_BATCH_NAME)}"
                    f"&sequence_number=1&sequence_size=10&tag=lonely-tag"
                ),
            },
        ]

        for scenario in scenarios:
            with self.subTest(scenario=scenario["desc"]):
                self._reset_responses()
                responses.add(responses.POST, scenario["expected_url"], json={"success": True}, status=200)

                upload_image_payload = {
                    "sequence_number": self.SEQUENCE_NUMBER,
                    "sequence_size": self.SEQUENCE_SIZE,
                    "tag_names": self.TAG_NAMES_LOCAL,
                }

                if "batch_name" in scenario:
                    upload_image_payload["batch_name"] = scenario["batch_name"]

                result = upload_image(self.API_KEY, self.PROJECT_URL, self.IMAGE_PATH_LOCAL, **upload_image_payload)
                self.assertTrue(result["success"], msg=f"Failed in scenario: {scenario['desc']}")

    @responses.activate
    def test_upload_image_hosted(self):
        scenarios = [
            {
                "desc": "with batch_name",
                "batch_name": "My batch",
                "expected_url": (
                    f"{API_URL}/dataset/{self.PROJECT_URL}/upload?"
                    f"api_key={self.API_KEY}&name={self.IMAGE_NAME_HOSTED}"
                    f"&split=train&image={urllib.parse.quote_plus(self.IMAGE_PATH_HOSTED)}"
                    f"&batch=My%20batch&tag=tag1&tag=tag2"
                ),
            },
            {
                "desc": "without batch_name",
                "expected_url": (
                    f"{API_URL}/dataset/{self.PROJECT_URL}/upload?"
                    f"api_key={self.API_KEY}&batch={urllib.parse.quote_plus(DEFAULT_BATCH_NAME)}"
                    f"&name={self.IMAGE_NAME_HOSTED}&split=train"
                    f"&image={urllib.parse.quote_plus(self.IMAGE_PATH_HOSTED)}&tag=tag1&tag=tag2"
                ),
            },
            {
                "desc": "without batch_name",
                "batch_name": None,
                "expected_url": (
                    f"{API_URL}/dataset/{self.PROJECT_URL}/upload?"
                    f"api_key={self.API_KEY}&name={self.IMAGE_NAME_HOSTED}"
                    f"&split=train&image={urllib.parse.quote_plus(self.IMAGE_PATH_HOSTED)}"
                    f"&batch={urllib.parse.quote_plus(DEFAULT_BATCH_NAME)}&tag=tag1&tag=tag2"
                ),
            },
        ]

        for scenario in scenarios:
            with self.subTest(scenario=scenario["desc"]):
                self._reset_responses()
                responses.add(responses.POST, scenario["expected_url"], json={"success": True}, status=200)

                upload_image_payload = {
                    "hosted_image": True,
                    "tag_names": self.TAG_NAMES_HOSTED,
                }

                if "batch_name" in scenario:
                    upload_image_payload["batch_name"] = scenario["batch_name"]

                result = upload_image(self.API_KEY, self.PROJECT_URL, self.IMAGE_PATH_HOSTED, **upload_image_payload)
                self.assertTrue(result["success"], msg=f"Failed in scenario: {scenario['desc']}")

    @responses.activate
    @patch("roboflow.adapters.rfapi.open", new_callable=mock_open, read_data=b"image_data")
    def test_upload_image_local_with_metadata(self, _mock_file):
        metadata = {"camera_id": "cam001", "location": "warehouse"}
        expected_url = (
            f"{API_URL}/dataset/{self.PROJECT_URL}/upload?"
            f"api_key={self.API_KEY}&batch={urllib.parse.quote_plus(DEFAULT_BATCH_NAME)}"
            f"&tag=lonely-tag"
        )
        responses.add(responses.POST, expected_url, json={"success": True}, status=200)

        result = upload_image(
            self.API_KEY,
            self.PROJECT_URL,
            self.IMAGE_PATH_LOCAL,
            tag_names=self.TAG_NAMES_LOCAL,
            metadata=metadata,
        )
        self.assertTrue(result["success"])

        # Verify metadata was sent as a multipart field
        request_body = responses.calls[0].request.body
        self.assertIn(b'"camera_id"', request_body)
        self.assertIn(b'"warehouse"', request_body)

    @responses.activate
    def test_upload_image_hosted_with_metadata(self):
        metadata = {"camera_id": "cam001", "location": "warehouse"}
        metadata_encoded = urllib.parse.quote_plus(json.dumps(metadata))
        expected_url = (
            f"{API_URL}/dataset/{self.PROJECT_URL}/upload?"
            f"api_key={self.API_KEY}&name={self.IMAGE_NAME_HOSTED}"
            f"&split=train&image={urllib.parse.quote_plus(self.IMAGE_PATH_HOSTED)}"
            f"&batch={urllib.parse.quote_plus(DEFAULT_BATCH_NAME)}"
            f"&tag=tag1&tag=tag2&metadata={metadata_encoded}"
        )
        responses.add(responses.POST, expected_url, json={"success": True}, status=200)

        result = upload_image(
            self.API_KEY,
            self.PROJECT_URL,
            self.IMAGE_PATH_HOSTED,
            hosted_image=True,
            tag_names=self.TAG_NAMES_HOSTED,
            metadata=metadata,
        )
        self.assertTrue(result["success"])

    @responses.activate
    def test_upload_image_local_uploads_original_bytes(self):
        """Server-side dedup relies on the SDK uploading the file bytes exactly as-is."""
        import tempfile

        raw_bytes = b"\x89PNG\r\n\x1a\nfake-png-bytes-not-decodable-as-jpeg"
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            tf.write(raw_bytes)
            tmp_path = tf.name

        try:
            expected_url = (
                f"{API_URL}/dataset/{self.PROJECT_URL}/upload?"
                f"api_key={self.API_KEY}&batch={urllib.parse.quote_plus(DEFAULT_BATCH_NAME)}"
            )
            responses.add(responses.POST, expected_url, json={"success": True}, status=200)

            result = upload_image(self.API_KEY, self.PROJECT_URL, tmp_path)
            self.assertTrue(result["success"])

            request_body = responses.calls[0].request.body
            self.assertIn(raw_bytes, request_body)
            self.assertIn(b"image/png", request_body)
        finally:
            os.unlink(tmp_path)

    def _reset_responses(self):
        responses.reset()


class TestV2Trainings(unittest.TestCase):
    API_KEY = "test_api_key"
    WORKSPACE = "test-workspace"
    PROJECT = "test-project"
    VERSION = "3"
    BASE_URL = f"{API_URL}/{WORKSPACE}/{PROJECT}/{VERSION}/v2/trainings"

    RECIPE_RESPONSE = {
        "modelType": "rfdetr-medium",
        "family": "rf-detr",
        "taskType": "object-detection",
        "schema": {"hyperparameters": [{"key": "lr", "type": "float"}]},
        "template": {
            "schema_version": 1,
            "input": {},
            "online_preprocessing": [],
            "online_augmentation": {"splits": ["train"], "steps": []},
            "source_version": {},
            "hyperparameters": {},
        },
        "usage": "...",
    }

    def _request_query(self):
        return dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(responses.calls[0].request.url).query))

    def _request_body(self):
        return json.loads(responses.calls[0].request.body)

    @responses.activate
    def test_get_train_recipe(self):
        responses.add(responses.GET, f"{self.BASE_URL}/recipe", json=self.RECIPE_RESPONSE, status=200)

        result = get_train_recipe(self.API_KEY, self.WORKSPACE, self.PROJECT, self.VERSION, model_type="rfdetr-medium")

        self.assertEqual(result, self.RECIPE_RESPONSE)
        query = self._request_query()
        self.assertEqual(query["api_key"], self.API_KEY)
        self.assertEqual(query["modelType"], "rfdetr-medium")

    @responses.activate
    def test_get_train_recipe_raises_on_error(self):
        responses.add(responses.GET, f"{self.BASE_URL}/recipe", json={"error": "bad model type"}, status=400)

        with self.assertRaises(RoboflowError):
            get_train_recipe(self.API_KEY, self.WORKSPACE, self.PROJECT, self.VERSION, model_type="nope")

    @responses.activate
    def test_create_training_v2_sends_only_provided_keys_in_camel_case(self):
        responses.add(
            responses.POST,
            self.BASE_URL,
            json={"trainingId": "abc123", "status": "queued", "jobId": "job-1"},
            status=200,
        )

        recipe = {"schema_version": 1, "hyperparameters": {"lr": 0.0002}}
        result = create_training_v2(
            self.API_KEY,
            self.WORKSPACE,
            self.PROJECT,
            self.VERSION,
            model_type="rfdetr-medium",
            speed="fast",
            checkpoint="ckpt",
            epochs=10,
            train_recipe=recipe,
        )

        self.assertEqual(result["trainingId"], "abc123")
        self.assertEqual(self._request_query()["api_key"], self.API_KEY)
        body = self._request_body()
        self.assertEqual(
            body,
            {
                "modelType": "rfdetr-medium",
                "speed": "fast",
                "checkpoint": "ckpt",
                "epochs": 10,
                "trainRecipe": recipe,
            },
        )

    @responses.activate
    def test_create_training_v2_omits_none_keys(self):
        responses.add(responses.POST, self.BASE_URL, json={"trainingId": "abc123"}, status=200)

        create_training_v2(self.API_KEY, self.WORKSPACE, self.PROJECT, self.VERSION)

        self.assertEqual(self._request_body(), {})

    @responses.activate
    def test_create_training_v2_raises_on_error(self):
        responses.add(responses.POST, self.BASE_URL, json={"error": "nope"}, status=500)

        with self.assertRaises(RoboflowError):
            create_training_v2(self.API_KEY, self.WORKSPACE, self.PROJECT, self.VERSION, model_type="rfdetr-medium")

    @responses.activate
    def test_list_trainings_for_version_unwraps_trainings_key(self):
        payload = {"trainings": [{"trainingId": "t-1"}, {"trainingId": "t-2"}]}
        responses.add(responses.GET, self.BASE_URL, json=payload, status=200)

        result = list_trainings_for_version(self.API_KEY, self.WORKSPACE, self.PROJECT, self.VERSION)

        self.assertEqual(result, payload["trainings"])
        self.assertEqual(self._request_query(), {"api_key": self.API_KEY})

    @responses.activate
    def test_list_trainings_for_version_raises_on_error(self):
        responses.add(responses.GET, self.BASE_URL, json={"error": "nope"}, status=404)

        with self.assertRaises(RoboflowError):
            list_trainings_for_version(self.API_KEY, self.WORKSPACE, self.PROJECT, self.VERSION)

    @responses.activate
    def test_get_training_with_training_id(self):
        responses.add(responses.GET, f"{self.BASE_URL}/get", json={"trainingId": "t-1"}, status=200)

        result = get_training(self.API_KEY, self.WORKSPACE, self.PROJECT, self.VERSION, training_id="t-1")

        self.assertEqual(result, {"trainingId": "t-1"})
        query = self._request_query()
        self.assertEqual(query["api_key"], self.API_KEY)
        self.assertEqual(query["trainingId"], "t-1")

    @responses.activate
    def test_get_training_without_training_id(self):
        responses.add(responses.GET, f"{self.BASE_URL}/get", json={"trainingId": "latest"}, status=200)

        result = get_training(self.API_KEY, self.WORKSPACE, self.PROJECT, self.VERSION)

        self.assertEqual(result, {"trainingId": "latest"})
        self.assertNotIn("trainingId", self._request_query())

    @responses.activate
    def test_get_training_raises_on_error(self):
        responses.add(responses.GET, f"{self.BASE_URL}/get", json={"error": "nope"}, status=404)

        with self.assertRaises(RoboflowError):
            get_training(self.API_KEY, self.WORKSPACE, self.PROJECT, self.VERSION, training_id="missing")


if __name__ == "__main__":
    unittest.main()
