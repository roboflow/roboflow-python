import os
from sys import intern

import requests
import responses
import unittest
from unittest.mock import patch

from .helpers import get_version


class TestVersion(unittest.TestCase):

    def setUp(self):
        super(TestVersion, self).setUp()
        self.version = get_version(project_name="Test Dataset", id="test-workspace/test-project/2", version_number="3")

    def test_get_download_location_with_env_variable(self):
        with patch.dict(os.environ, { "DATASET_DIRECTORY": "/my/exports"}, clear=True):

            # This is a weird python thing to get access to the private function for testing.
            __get_download_location = self.version._Version__get_download_location
            location = __get_download_location()
            self.assertEqual(location, "/my/exports/Test-Dataset-3")

    def test_get_download_location_without_env_variable(self):
        # This is a weird python thing to get access to the private function for testing.
        __get_download_location = self.version._Version__get_download_location
        location = __get_download_location()
        self.assertEqual(location, "Test-Dataset-3")

    def test_get_download_url(self):
        # This is a weird python thing to get access to the private function for testing.
        __get_download_url = self.version._Version__get_download_url
        url = __get_download_url("yolo1337")
        self.assertEqual(url, "https://api.roboflow.com/test-workspace/test-project/3/yolo1337")

    def test_download_with_location_overwrites_location(self):
        pass

    def test_download_without_format_raises_error(self):
        with self.assertRaises(RuntimeError):
            self.version.download()

    @responses.activate
    def test_export_returns_true_on_api_success(self):
        version = get_version(project_name="Test Dataset", id="test-workspace/test-project/2", version_number="4")
        api_url = f"https://api.roboflow.com/test-workspace/test-project/4/test-format"
        responses.add(responses.POST, api_url, status=204)
        
        export = version.export("test-format")
        request = responses.calls[0].request

        self.assertTrue(export)
        self.assertEqual(request.method, "POST")
        self.assertRegex(request.url, rf"^{api_url}")
        self.assertDictEqual(request.params, { "api_key": "test-api-key" })

    @responses.activate
    def test_export_raises_error_on_bad_request(self):
        version = get_version(project_name="Test Dataset", id="test-workspace/test-project/2", version_number="4")
        api_url = f"https://api.roboflow.com/test-workspace/test-project/4/test-format"
        responses.add(responses.POST, api_url, status=400, json={ "error": "BROKEN!!"})
        
        with self.assertRaises(RuntimeError):
            version.export("test-format")

    @responses.activate
    def test_export_raises_error_on_api_failure(self):
        version = get_version(project_name="Test Dataset", id="test-workspace/test-project/2", version_number="4")
        api_url = f"https://api.roboflow.com/test-workspace/test-project/4/test-format"
        responses.add(responses.POST, api_url, status=500)
        
        with self.assertRaises(requests.exceptions.HTTPError):
            version.export("test-format")

class TestGetFormatIdentifier(unittest.TestCase):
    def setUp(self):
        super(TestGetFormatIdentifier, self).setUp()
        self.version = get_version(project_name="Test Dataset", id="test-workspace/test-project/2", version_number="3")

        # This is a weird python thing to get access to the private function for testing.
        self.get_format_identifier = self.version._Version__get_format_identifier

    def test_returns_simple_format(self):
        self.assertEqual(self.get_format_identifier("coco"), "coco")

    def test_returns_friendly_names_for_supported_formats(self):
        formats = [("yolov5", "yolov5pytorch"), ("yolov7", "yolov7pytorch")]
        for external_format, internal_format in formats:
            self.assertEqual(self.get_format_identifier(external_format), internal_format)

    def test_falls_back_to_instance_variable_if_model_format_is_none(self):
        self.version.model_format = "fallback"
        self.assertEqual(self.get_format_identifier(None), "fallback")

    def test_falls_back_to_instance_variable_if_model_format_is_none_and_converts_human_readable_format_to_identifier(self):
        self.version.model_format = "yolov5"
        self.assertEqual(self.get_format_identifier(None), "yolov5pytorch")

    def test_raises_runtime_error_if_model_format_is_none(self):
        self.version.model_format = None
        with self.assertRaises(RuntimeError):
            self.get_format_identifier(None)
