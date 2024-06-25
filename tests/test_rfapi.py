import os
import unittest
from unittest.mock import patch

import responses
import urllib

from roboflow.config import API_URL, DEFAULT_BATCH_NAME
from roboflow.adapters.rfapi import upload_image


class TestUploadImage(unittest.TestCase):

    @responses.activate
    @patch("roboflow.util.image_utils.file2jpeg")
    def test_upload_image_local(self, mock_file2jpeg):
        api_key = "test_api_key"
        project_url = "test_project"
        sequence_number = 1
        sequence_size = 10
        tag_names = ["lonely-tag"]
        
        scenarios = [
            {
                "desc": "with batch_name",
                "batch_name": "My personal batch",
                "expected_url": (f"{API_URL}/dataset/{project_url}/upload?api_key={api_key}"
                                 f"&batch_name=My%20personal%20batch&sequence_number=1&sequence_size=10&tag=lonely-tag")
            },
            {
                "desc": "without batch_name",
                "batch_name": None,
                "expected_url": (f"{API_URL}/dataset/{project_url}/upload?api_key={api_key}"
                                 f"&batch_name={DEFAULT_BATCH_NAME}&sequence_number=1&sequence_size=10&tag=lonely-tag")
            }
        ]
        
        mock_file2jpeg.return_value = b"image_data"
        
        for scenario in scenarios:
            with self.subTest(scenario=scenario["desc"]):
                responses.reset()  # Clear previous responses
                responses.add(responses.POST, scenario["expected_url"], json={"success": True}, status=200)
                
                upload_image_payload = dict(
                    sequence_number=sequence_number,
                    sequence_size=sequence_size,
                    tag_names=tag_names
                )
                
                if scenario["batch_name"]:
                    upload_image_payload.update(batch_name=scenario["batch_name"])
                
                result = upload_image(api_key, project_url, "test_image.jpg", **upload_image_payload)
                self.assertTrue(result["success"])

    @responses.activate
    def test_upload_image_hosted(self):
        api_key = "test_api_key"
        project_url = "test_project"
        image_path = "http://example.com/test_image.jpg"
        tag_names = ["tag1", "tag2"]
        image_name = os.path.basename(image_path)
        
        scenarios = [
            {
                "desc": "with batch_name",
                "batch_name": "My batch",
                "expected_url": (f"{API_URL}/dataset/{project_url}/upload?api_key={api_key}"
                                 f"&name={image_name}&split=train&image={urllib.parse.quote_plus(image_path)}"
                                 f"&batch_name=My%20batch&tag=tag1&tag=tag2")
            },
            {
                "desc": "without batch_name",
                "batch_name": None,
                "expected_url": (f"{API_URL}/dataset/{project_url}/upload?api_key={api_key}"
                                 f"&name={image_name}&split=train&image={urllib.parse.quote_plus(image_path)}"
                                 f"&batch_name={DEFAULT_BATCH_NAME}&tag=tag1&tag=tag2")
            }
        ]
        
        for scenario in scenarios:
            with self.subTest(scenario=scenario["desc"]):
                responses.reset()  # Clear previous responses
                responses.add(responses.POST, scenario["expected_url"], json={"success": True}, status=200)
                
                upload_image_payload = dict(
                    hosted_image=True,
                    tag_names=tag_names
                )
                
                if scenario["batch_name"]:
                    upload_image_payload.update(batch_name=scenario["batch_name"])
                
                result = upload_image(api_key, project_url, image_path, **upload_image_payload)
                self.assertTrue(result["success"])


if __name__ == '__main__':
    unittest.main()
