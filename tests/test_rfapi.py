import json
import os
import unittest
import urllib
from unittest.mock import mock_open, patch

import responses

from roboflow.adapters.rfapi import upload_image
from roboflow.config import API_URL, DEFAULT_BATCH_NAME


class TestUploadImage(unittest.TestCase):
    API_KEY = "test_api_key"
    PROJECT_URL = "test_project"
    SEQUENCE_NUMBER = 1
    SEQUENCE_SIZE = 10
    TAG_NAMES_LOCAL = ["lonely-tag"]
    TAG_NAMES_HOSTED = ["tag1", "tag2"]
    IMAGE_PATH_LOCAL = "test_image.jpg"
    IMAGE_PATH_LOCAL_PNG = "test_image.png"
    IMAGE_PATH_HOSTED = "http://example.com/test_image.jpg"
    IMAGE_NAME_HOSTED = os.path.basename(IMAGE_PATH_HOSTED)

    @responses.activate
    @patch("roboflow.adapters.rfapi.open", new_callable=mock_open, read_data=b"image_data", create=True)
    def test_upload_image_local(self, mock_image_file):
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
                self.assertIn(b"image_data", responses.calls[0].request.body)
                self.assertIn(b"Content-Type: image/jpeg", responses.calls[0].request.body)

        self.assertEqual(mock_image_file.call_count, len(scenarios))

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
    @patch("roboflow.adapters.rfapi.open", new_callable=mock_open, read_data=b"image_data", create=True)
    def test_upload_image_local_with_metadata(self, mock_image_file):
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
        mock_image_file.assert_called_once_with(self.IMAGE_PATH_LOCAL, "rb")

        # Verify metadata was sent as a multipart field
        request_body = responses.calls[0].request.body
        self.assertIn(b'"camera_id"', request_body)
        self.assertIn(b'"warehouse"', request_body)

    @responses.activate
    @patch("roboflow.adapters.rfapi.open", new_callable=mock_open, read_data=b"png_image_data", create=True)
    def test_upload_image_local_preserves_original_file_type(self, mock_image_file):
        expected_url = (
            f"{API_URL}/dataset/{self.PROJECT_URL}/upload?"
            f"api_key={self.API_KEY}&batch={urllib.parse.quote_plus(DEFAULT_BATCH_NAME)}"
        )
        responses.add(responses.POST, expected_url, json={"success": True}, status=200)

        result = upload_image(self.API_KEY, self.PROJECT_URL, self.IMAGE_PATH_LOCAL_PNG)

        self.assertTrue(result["success"])
        mock_image_file.assert_called_once_with(self.IMAGE_PATH_LOCAL_PNG, "rb")
        request_body = responses.calls[0].request.body
        self.assertIn(b"png_image_data", request_body)
        self.assertIn(b'filename="test_image.png"', request_body)
        self.assertIn(b"Content-Type: image/png", request_body)

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

    def _reset_responses(self):
        responses.reset()


if __name__ == "__main__":
    unittest.main()
