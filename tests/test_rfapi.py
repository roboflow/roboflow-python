import os
import unittest
import urllib
from unittest.mock import patch

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
    IMAGE_PATH_HOSTED = "http://example.com/test_image.jpg"
    IMAGE_NAME_HOSTED = os.path.basename(IMAGE_PATH_HOSTED)

    @responses.activate
    @patch("roboflow.util.image_utils.file2jpeg")
    def test_upload_image_local(self, mock_file2jpeg):
        mock_file2jpeg.return_value = b"image_data"

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

    def _reset_responses(self):
        responses.reset()


if __name__ == "__main__":
    unittest.main()
