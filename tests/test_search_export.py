import io
import os
import shutil
import unittest
import zipfile
from unittest.mock import MagicMock, patch

import responses
import requests

from roboflow.adapters.rfapi import RoboflowError, get_search_export, start_search_export
from roboflow.config import API_URL


class TestStartSearchExport(unittest.TestCase):
    API_KEY = "test_key"
    WORKSPACE = "my-workspace"

    @responses.activate
    def test_success(self):
        url = f"{API_URL}/{self.WORKSPACE}/search/export?api_key={self.API_KEY}"
        responses.add(responses.POST, url, json={"success": True, "link": "export_123"}, status=202)

        export_id = start_search_export(self.API_KEY, self.WORKSPACE, query="*", format="coco")
        self.assertEqual(export_id, "export_123")

        body = responses.calls[0].request.body
        self.assertIn(b'"query"', body)
        self.assertIn(b'"format"', body)

    @responses.activate
    def test_with_dataset(self):
        url = f"{API_URL}/{self.WORKSPACE}/search/export?api_key={self.API_KEY}"
        responses.add(responses.POST, url, json={"success": True, "link": "export_456"}, status=202)

        export_id = start_search_export(
            self.API_KEY, self.WORKSPACE, query="tag:train", format="yolov8", dataset="my-dataset"
        )
        self.assertEqual(export_id, "export_456")

        body = responses.calls[0].request.body
        self.assertIn(b'"dataset"', body)

    @responses.activate
    def test_error_response(self):
        url = f"{API_URL}/{self.WORKSPACE}/search/export?api_key={self.API_KEY}"
        responses.add(responses.POST, url, body="Bad Request", status=400)

        with self.assertRaises(RoboflowError):
            start_search_export(self.API_KEY, self.WORKSPACE, query="*", format="coco")


class TestGetSearchExport(unittest.TestCase):
    API_KEY = "test_key"
    WORKSPACE = "my-workspace"

    @responses.activate
    def test_not_ready(self):
        url = f"{API_URL}/{self.WORKSPACE}/search/export/exp1?api_key={self.API_KEY}"
        responses.add(responses.GET, url, json={"ready": False}, status=200)

        result = get_search_export(self.API_KEY, self.WORKSPACE, "exp1")
        self.assertFalse(result["ready"])

    @responses.activate
    def test_ready(self):
        url = f"{API_URL}/{self.WORKSPACE}/search/export/exp1?api_key={self.API_KEY}"
        responses.add(responses.GET, url, json={"ready": True, "link": "https://download.url/file.zip"}, status=200)

        result = get_search_export(self.API_KEY, self.WORKSPACE, "exp1")
        self.assertTrue(result["ready"])
        self.assertEqual(result["link"], "https://download.url/file.zip")

    @responses.activate
    def test_error_response(self):
        url = f"{API_URL}/{self.WORKSPACE}/search/export/exp1?api_key={self.API_KEY}"
        responses.add(responses.GET, url, body="Not Found", status=404)

        with self.assertRaises(RoboflowError):
            get_search_export(self.API_KEY, self.WORKSPACE, "exp1")


class TestWorkspaceSearchExportValidation(unittest.TestCase):
    def _make_workspace(self):
        from roboflow.core.workspace import Workspace

        info = {
            "workspace": {
                "name": "Test",
                "url": "test-ws",
                "projects": [],
                "members": [],
            }
        }
        return Workspace(info, api_key="test_key", default_workspace="test-ws", model_format="yolov8")

    def test_mutual_exclusion(self):
        ws = self._make_workspace()
        with self.assertRaises(ValueError) as ctx:
            ws.search_export(query="*", dataset="ds", annotation_group="ag")
        self.assertIn("mutually exclusive", str(ctx.exception))


class TestWorkspaceSearchExportFlow(unittest.TestCase):
    @staticmethod
    def _build_zip_bytes(files):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for filename, content in files.items():
                zip_file.writestr(filename, content)
        return buffer.getvalue()

    def _make_workspace(self):
        from roboflow.core.workspace import Workspace

        info = {
            "workspace": {
                "name": "Test",
                "url": "test-ws",
                "projects": [],
                "members": [],
            }
        }
        return Workspace(info, api_key="test_key", default_workspace="test-ws", model_format="yolov8")

    @patch("roboflow.core.workspace.rfapi")
    @patch("roboflow.core.workspace.requests")
    def test_full_flow(self, mock_requests, mock_rfapi):
        ws = self._make_workspace()

        mock_rfapi.start_search_export.return_value = "exp_abc"
        mock_rfapi.get_search_export.return_value = {"ready": True, "link": "https://example.com/export.zip"}

        fake_zip = self._build_zip_bytes({"images/sample.jpg": "fake-image-data"})
        mock_response = MagicMock()
        mock_response.headers = {"content-length": str(len(fake_zip))}
        mock_response.raise_for_status.return_value = None
        mock_response.iter_content.return_value = [fake_zip[:1024], fake_zip[1024:]]
        mock_requests.get.return_value = mock_response

        location = "./test_search_export_output"
        try:
            result = ws.search_export(query="*", format="coco", location=location)

            expected_location = os.path.abspath(location)
            self.assertEqual(result, expected_location)
            self.assertTrue(os.path.exists(os.path.join(expected_location, "images", "sample.jpg")))
            self.assertFalse(os.path.exists(os.path.join(expected_location, "roboflow.zip")))

            mock_rfapi.start_search_export.assert_called_once_with(
                api_key="test_key",
                workspace_url="test-ws",
                query="*",
                format="coco",
                dataset=None,
                annotation_group=None,
                name=None,
            )
            mock_rfapi.get_search_export.assert_called_once_with(
                api_key="test_key",
                workspace_url="test-ws",
                export_id="exp_abc",
            )
            mock_response.raise_for_status.assert_called_once()
            mock_response.iter_content.assert_called_once_with(chunk_size=1024)
        finally:
            if os.path.exists(location):
                shutil.rmtree(location)

    @patch("roboflow.core.workspace.rfapi")
    @patch("roboflow.core.workspace.requests")
    def test_full_flow_without_content_length_still_streams(self, mock_requests, mock_rfapi):
        ws = self._make_workspace()

        mock_rfapi.start_search_export.return_value = "exp_abc"
        mock_rfapi.get_search_export.return_value = {"ready": True, "link": "https://example.com/export.zip"}

        fake_zip = self._build_zip_bytes({"annotations/instances.json": "{}"})
        mock_response = MagicMock()
        mock_response.headers = {}
        mock_response.raise_for_status.return_value = None
        mock_response.iter_content.return_value = [fake_zip]
        mock_requests.get.return_value = mock_response

        location = "./test_search_export_no_content_length"
        try:
            result = ws.search_export(query="*", format="coco", location=location)
            expected_location = os.path.abspath(location)
            self.assertEqual(result, expected_location)
            self.assertTrue(os.path.exists(os.path.join(expected_location, "annotations", "instances.json")))
            mock_response.iter_content.assert_called_once_with(chunk_size=1024)
        finally:
            if os.path.exists(location):
                shutil.rmtree(location)

    @patch("roboflow.core.workspace.rfapi")
    @patch("roboflow.core.workspace.requests")
    def test_download_http_error_raises_roboflow_error(self, mock_requests, mock_rfapi):
        ws = self._make_workspace()

        mock_rfapi.start_search_export.return_value = "exp_abc"
        mock_rfapi.get_search_export.return_value = {"ready": True, "link": "https://example.com/export.zip"}

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("403 Client Error")
        mock_requests.get.return_value = mock_response

        with self.assertRaises(RoboflowError) as context:
            ws.search_export(query="*", format="coco", location="./test_search_export_http_error")

        self.assertIn("Failed to download search export", str(context.exception))

    @patch("roboflow.core.workspace.rfapi")
    @patch("roboflow.core.workspace.requests")
    def test_no_extract(self, mock_requests, mock_rfapi):
        ws = self._make_workspace()

        mock_rfapi.start_search_export.return_value = "exp_abc"
        mock_rfapi.get_search_export.return_value = {"ready": True, "link": "https://example.com/export.zip"}

        fake_zip = self._build_zip_bytes({"images/sample.jpg": "fake-image-data"})
        mock_response = MagicMock()
        mock_response.headers = {"content-length": str(len(fake_zip))}
        mock_response.raise_for_status.return_value = None
        mock_response.iter_content.return_value = [fake_zip]
        mock_requests.get.return_value = mock_response

        location = "./test_search_export_no_extract"
        try:
            result = ws.search_export(query="*", format="coco", location=location, extract_zip=False)

            expected_zip = os.path.join(os.path.abspath(location), "roboflow.zip")
            self.assertEqual(result, expected_zip)
            self.assertTrue(os.path.exists(expected_zip))
        finally:
            if os.path.exists(location):
                shutil.rmtree(location)


if __name__ == "__main__":
    unittest.main()
