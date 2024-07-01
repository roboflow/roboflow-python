import os
import unittest
from unittest.mock import patch

import requests
import responses

from roboflow.core.version import Version, unwrap_version_id
from tests.helpers import get_version


class TestDownload(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.api_url = "https://api.roboflow.com/test-workspace/test-project/4/coco"
        self.version = get_version(
            project_name="Test Dataset",
            id="test-workspace/test-project/2",
            version_number="4",
        )

        self.generating_url = "https://api.roboflow.com/Test Workspace Name/Test Dataset/4"

    @responses.activate
    def test_download_raises_exception_on_bad_request(self):
        responses.add(responses.GET, self.api_url, status=404, json={"error": "Broken"})
        responses.add(
            responses.GET,
            self.generating_url,
            json={"version": {"generating": False, "progress": 1.0}},
        )

        with self.assertRaises(RuntimeError):
            self.version.download("coco")

    @responses.activate
    def test_download_raises_exception_on_api_failure(self):
        responses.add(responses.GET, self.api_url, status=500)
        responses.add(
            responses.GET,
            self.generating_url,
            json={"version": {"generating": False, "progress": 1.0}},
        )
        with self.assertRaises(requests.exceptions.HTTPError):
            self.version.download("coco")

    @responses.activate
    @patch.object(Version, "_Version__download_zip")
    @patch.object(Version, "_Version__extract_zip")
    @patch.object(Version, "_Version__reformat_yaml")
    def test_download_returns_dataset(self, *_):
        responses.add(responses.GET, self.api_url, json={"export": {"link": None}})
        responses.add(
            responses.GET,
            self.generating_url,
            json={"version": {"generating": False, "progress": 1.0}},
        )
        dataset = self.version.download("coco", location="/my-spot")
        self.assertEqual(dataset.name, self.version.name)
        self.assertEqual(dataset.version, self.version.version)
        self.assertEqual(dataset.model_format, "coco")
        self.assertEqual(dataset.location, "/my-spot")


class TestExport(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.api_url = "https://api.roboflow.com/test-workspace/test-project/4/test-format"
        self.version = get_version(
            project_name="Test Dataset",
            id="test-workspace/test-project/2",
            version_number="4",
        )

        self.generating_url = "https://api.roboflow.com/Test Workspace Name/Test Dataset/4"

    @responses.activate
    def test_export_returns_true_on_api_success(self):
        responses.add(responses.GET, self.api_url, status=200)
        responses.add(
            responses.GET,
            self.generating_url,
            json={"version": {"generating": False, "progress": 1.0}},
        )
        export = self.version.export("test-format")
        request = responses.calls[0].request

        self.assertTrue(export)
        self.assertEqual(request.method, "GET")
        self.assertDictEqual(request.params, {"nocache": "true", "api_key": "test-api-key"})

    @responses.activate
    def test_export_raises_error_on_bad_request(self):
        responses.add(responses.GET, self.api_url, status=400, json={"error": "BROKEN!!"})
        responses.add(
            responses.GET,
            self.generating_url,
            json={"version": {"generating": False, "progress": 1.0}},
        )
        with self.assertRaises(RuntimeError):
            self.version.export("test-format")

    @responses.activate
    def test_export_raises_error_on_api_failure(self):
        responses.add(responses.GET, self.api_url, status=500)
        responses.add(
            responses.GET,
            self.generating_url,
            json={"version": {"generating": False, "progress": 1.0}},
        )
        with self.assertRaises(requests.exceptions.HTTPError):
            self.version.export("test-format")


@patch.object(os, "makedirs")
class TestGetDownloadLocation(unittest.TestCase):
    def setUp(self, *_):
        super().setUp()
        self.version = get_version(
            project_name="Test Dataset",
            id="test-workspace/test-project/2",
            version_number="3",
        )

        # This is a weird python thing to get access to the private function for testing
        self.get_download_location = self.version._Version__get_download_location
        self.generating_url = "https://api.roboflow.com/Test Workspace Name/Test Dataset/4"

    @responses.activate
    def test_get_download_location_with_env_variable(self, *_):
        responses.add(
            responses.GET,
            self.generating_url,
            json={"version": {"generating": False, "progress": 1.0}},
        )
        with patch.dict(os.environ, {"DATASET_DIRECTORY": "/my/exports"}, clear=True):
            self.assertEqual(self.get_download_location(), "/my/exports/Test-Dataset-3")

    @responses.activate
    def test_get_download_location_without_env_variable(self, *_):
        responses.add(
            responses.GET,
            self.generating_url,
            json={"version": {"generating": False, "progress": 1.0}},
        )
        self.assertEqual(self.get_download_location(), "Test-Dataset-3")


class TestGetDownloadURL(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.version = get_version(
            project_name="Test Dataset",
            id="test-workspace/test-project/2",
            version_number="3",
        )

        # This is a weird python thing to get access to the private function for testing
        self.get_download_url = self.version._Version__get_download_url
        self.generating_url = "https://api.roboflow.com/Test Workspace Name/Test Dataset/4"

    @responses.activate
    def test_get_download_url(self):
        responses.add(
            responses.GET,
            self.generating_url,
            json={"version": {"generating": False, "progress": 1.0}},
        )
        url = self.get_download_url("yolo1337")
        self.assertEqual(url, "https://api.roboflow.com/test-workspace/test-project/3/yolo1337")


class TestGetFormatIdentifier(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.version = get_version(
            project_name="Test Dataset",
            id="test-workspace/test-project/2",
            version_number="3",
        )

        # This is a weird python thing to get access to the private function for testing
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

    def test_falls_back_to_instance_variable_if_model_format_is_none_and_converts_human_readable_format_to_identifier(  # noqa: E501
        self,
    ):
        self.version.model_format = "yolov5"
        self.assertEqual(self.get_format_identifier(None), "yolov5pytorch")

    def test_raises_runtime_error_if_model_format_is_none(self):
        self.version.model_format = None
        with self.assertRaises(RuntimeError):
            self.get_format_identifier(None)


def test_unwrap_version_id_when_full_identifier_is_given() -> None:
    # when
    result = unwrap_version_id(version_id="some-workspace/some-project/3")

    # then
    assert result == "3"


def test_unwrap_version_id_when_only_version_id_is_given() -> None:
    # when
    result = unwrap_version_id(version_id="3")

    # then
    assert result == "3"
