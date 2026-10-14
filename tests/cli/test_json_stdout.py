"""Under --json, stdout carries only the JSON document; SDK prints go to stderr (#546)."""

import io
import json
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import responses
from typer.testing import CliRunner

from roboflow.cli import app
from roboflow.config import API_URL
from tests import PROJECT_NAME, PROJECT_VERSION, ROBOFLOW_API_KEY, WORKSPACE_NAME, RoboflowTest

runner = CliRunner()

DOWNLOAD_URL = "https://example.com/export.zip"


def _zip_bytes(files: dict) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buffer.getvalue()


class TestJsonStdout(RoboflowTest):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp()
        env = patch.dict(os.environ, {"ROBOFLOW_API_KEY": ROBOFLOW_API_KEY})
        env.start()
        self.addCleanup(env.stop)
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _invoke(self, *argv: str):
        return runner.invoke(app, ["--api-key", ROBOFLOW_API_KEY, "--workspace", WORKSPACE_NAME, *argv])

    def _assert_json_stdout(self, result, expected_stderr: str) -> dict:
        self.assertEqual(result.exit_code, 0, msg=result.output)
        data = json.loads(result.stdout)
        self.assertIn(expected_stderr, result.stderr)
        return data

    def _mock_search_export(self):
        responses.add(
            responses.POST,
            f"{API_URL}/{WORKSPACE_NAME}/search/export?api_key={ROBOFLOW_API_KEY}",
            json={"success": True, "link": "exp1"},
            status=202,
        )
        responses.add(
            responses.GET,
            f"{API_URL}/{WORKSPACE_NAME}/search/export/exp1?api_key={ROBOFLOW_API_KEY}",
            json={"ready": True, "link": DOWNLOAD_URL},
        )
        responses.add(responses.GET, DOWNLOAD_URL, body=_zip_bytes({"images/a.jpg": "x"}))

    def test_search_export(self):
        self._mock_search_export()
        out = os.path.join(self.tmp, "out")

        result = self._invoke("--json", "search", f"project:{PROJECT_NAME}", "--export", "-l", out)

        data = self._assert_json_stdout(result, "Export started (id=exp1)")
        self.assertEqual(data, {"status": "completed", "path": out})

    def test_search_export_text_mode_keeps_sdk_output_on_stdout(self):
        self._mock_search_export()
        out = os.path.join(self.tmp, "out")

        result = self._invoke("search", f"project:{PROJECT_NAME}", "--export", "-l", out)

        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIn("Export started (id=exp1)", result.stdout)
        self.assertIn(f"Export completed: {out}", result.stdout)

    def test_image_upload_directory(self):
        Path(self.tmp, "a.jpg").write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")
        responses.add(
            responses.POST, re.compile(rf"{re.escape(API_URL)}/.*upload.*"), json={"success": True, "id": "i1"}
        )

        result = self._invoke("--json", "image", "upload", self.tmp, "-p", PROJECT_NAME)

        data = self._assert_json_stdout(result, "[UPLOADED]")
        self.assertEqual(data, {"status": "imported", "path": self.tmp, "count": 1})
        self.assertIn(f"Uploading to existing project {WORKSPACE_NAME}/{PROJECT_NAME}", result.stderr)

    def test_model_upload(self):
        Path(self.tmp, "roboflow_deploy.zip").write_bytes(b"weights")
        bundle = SimpleNamespace(model_type="yolov8", archive_path=Path(self.tmp, "roboflow_deploy.zip"))
        responses.add(
            responses.POST,
            re.compile(rf"{re.escape(API_URL)}/{WORKSPACE_NAME}/models/prepareUpload\?.*"),
            json={"url": "https://storage.example.com/signed"},
        )
        responses.add(responses.PUT, "https://storage.example.com/signed")

        with (
            patch("roboflow.util.model_processor.package_custom_weights_interactive", return_value=bundle),
            patch("roboflow.util.model_processor.validate_model_type_for_project"),
        ):
            result = self._invoke("--json", "model", "upload", "-m", self.tmp, "-t", "yolov8", "-p", PROJECT_NAME)

        data = self._assert_json_stdout(result, "View the status of your deployment")
        self.assertEqual(data, {"status": "uploaded"})
        self.assertNotIn("loading Roboflow", result.stdout + result.stderr)

    def test_version_download_with_on_demand_export(self):
        version_url = f"{API_URL}/{WORKSPACE_NAME}/{PROJECT_NAME}/{PROJECT_VERSION}"
        responses.add(
            responses.GET,
            f"{version_url}?api_key={ROBOFLOW_API_KEY}&nocache=true",
            json={"version": {"generating": False, "progress": 1.0, "images": 10}},
        )
        export_url = f"{version_url}/coco?api_key={ROBOFLOW_API_KEY}&nocache=true"
        responses.add(responses.GET, export_url, json={"ready": False, "progress": 0.4})
        responses.add(responses.GET, export_url, json={"export": {"link": DOWNLOAD_URL}})
        zip_body = _zip_bytes({"README.txt": "x"})
        responses.add(responses.GET, DOWNLOAD_URL, body=zip_body, headers={"content-length": str(len(zip_body))})
        out = os.path.join(self.tmp, "ds")

        with patch("roboflow.core.version.time.sleep"):
            result = self._invoke(
                "--json",
                "version",
                "download",
                f"{WORKSPACE_NAME}/{PROJECT_NAME}/{PROJECT_VERSION}",
                "-f",
                "coco",
                "-l",
                out,
            )

        data = self._assert_json_stdout(result, "Version export complete for coco format")
        self.assertEqual(data["format"], "coco")
        self.assertEqual(data["location"], out)
        self.assertIn("Exporting format coco in progress : 40.0%", result.stderr)
