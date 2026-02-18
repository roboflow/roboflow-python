import io
import os
import shutil
import unittest
import zipfile

import responses

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


class TestWorkspaceSearchExport(unittest.TestCase):
    API_KEY = "test_key"
    WORKSPACE = "test-ws"
    DOWNLOAD_URL = "https://example.com/export.zip"

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
                "url": self.WORKSPACE,
                "projects": [],
                "members": [],
            }
        }
        return Workspace(info, api_key=self.API_KEY, default_workspace=self.WORKSPACE, model_format="yolov8")

    def _register_responses(self, zip_bytes=b"", download_status=200):
        export_url = f"{API_URL}/{self.WORKSPACE}/search/export?api_key={self.API_KEY}"
        responses.add(responses.POST, export_url, json={"success": True, "link": "exp_abc"}, status=202)

        poll_url = f"{API_URL}/{self.WORKSPACE}/search/export/exp_abc?api_key={self.API_KEY}"
        responses.add(responses.GET, poll_url, json={"ready": True, "link": self.DOWNLOAD_URL}, status=200)

        responses.add(responses.GET, self.DOWNLOAD_URL, body=zip_bytes, status=download_status)

    def test_mutual_exclusion(self):
        ws = self._make_workspace()
        with self.assertRaises(ValueError) as ctx:
            ws.search_export(query="*", dataset="ds", annotation_group="ag")
        self.assertIn("mutually exclusive", str(ctx.exception))

    @responses.activate
    def test_full_flow(self):
        ws = self._make_workspace()
        fake_zip = self._build_zip_bytes({"images/sample.jpg": "fake-image-data"})
        self._register_responses(fake_zip)

        location = "./test_search_export_output"
        try:
            result = ws.search_export(query="*", format="coco", location=location)

            expected_location = os.path.abspath(location)
            self.assertEqual(result, expected_location)
            self.assertTrue(os.path.exists(os.path.join(expected_location, "images", "sample.jpg")))
            self.assertFalse(os.path.exists(os.path.join(expected_location, "roboflow.zip")))
        finally:
            if os.path.exists(location):
                shutil.rmtree(location)

    @responses.activate
    def test_download_http_error(self):
        ws = self._make_workspace()
        self._register_responses(download_status=403)

        with self.assertRaises(RoboflowError) as ctx:
            ws.search_export(query="*", format="coco", location="./test_search_export_http_error")

        self.assertIn("Failed to download search export", str(ctx.exception))

    @responses.activate
    def test_no_extract(self):
        ws = self._make_workspace()
        fake_zip = self._build_zip_bytes({"images/sample.jpg": "fake-image-data"})
        self._register_responses(fake_zip)

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
