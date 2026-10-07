"""Failed model uploads raise instead of printing an error and returning normally."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import responses

from roboflow.adapters.rfapi import RoboflowError
from roboflow.config import API_URL
from roboflow.core.workspace import Workspace
from tests.helpers import get_version

SIGNED_URL = "https://storage.example.com/upload"


class _WeightsDirMixin(unittest.TestCase):
    def setUp(self):
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        self.model_path = tmpdir.name
        with open(os.path.join(self.model_path, "roboflow_deploy.zip"), "wb") as f:
            f.write(b"weights")


class TestVersionDeploy(_WeightsDirMixin):
    UPLOAD_MODEL_URL = f"{API_URL}/Test Workspace Name/Test Project Name/1/uploadModel"

    def deploy(self):
        bundle = SimpleNamespace(model_type="yolov8n", archive_path=SimpleNamespace(name="roboflow_deploy.zip"))
        with patch("roboflow.core.version.package_custom_weights_interactive", return_value=bundle):
            get_version().deploy("yolov8n", self.model_path, "weights/best.pt")

    @responses.activate
    def test_upload_url_error_raises(self):
        responses.add(responses.GET, self.UPLOAD_MODEL_URL, json={"error": "nope"}, status=500)

        with self.assertRaises(RoboflowError) as ctx:
            self.deploy()

        self.assertEqual(ctx.exception.status_code, 500)
        self.assertIn("model upload URL", str(ctx.exception))

    @responses.activate
    def test_429_raises(self):
        responses.add(responses.GET, self.UPLOAD_MODEL_URL, status=429)

        with self.assertRaises(RoboflowError) as ctx:
            self.deploy()

        self.assertEqual(ctx.exception.status_code, 429)
        self.assertIn("already has a trained model", str(ctx.exception))

    @responses.activate
    def test_weights_upload_error_raises(self):
        responses.add(responses.GET, self.UPLOAD_MODEL_URL, json={"url": SIGNED_URL})
        responses.add(responses.PUT, SIGNED_URL, status=403)

        with self.assertRaises(RoboflowError) as ctx:
            self.deploy()

        self.assertEqual(ctx.exception.status_code, 403)

    @responses.activate
    def test_success_prints_status_url(self):
        responses.add(responses.GET, self.UPLOAD_MODEL_URL, json={"url": SIGNED_URL})
        responses.add(responses.PUT, SIGNED_URL, status=200)

        with patch("builtins.print") as mock_print:
            self.deploy()

        printed = " ".join(str(call.args[0]) for call in mock_print.call_args_list)
        self.assertIn("View the status of your deployment at:", printed)


class TestWorkspaceUploadZip(_WeightsDirMixin):
    PREPARE_URL = f"{API_URL}/test-ws/models/prepareUpload"

    def upload(self):
        workspace = Workspace(
            {"workspace": {"name": "Test", "projects": [], "url": "test-ws"}},
            api_key="test-key",
            default_workspace="test-ws",
            model_format="yolov8",
        )
        workspace._upload_zip("yolov8n", self.model_path, ["proj"], "my-model", "roboflow_deploy.zip")

    @responses.activate
    def test_deployment_url_error_raises(self):
        responses.add(responses.POST, self.PREPARE_URL, json={"error": "nope"}, status=400)

        with self.assertRaises(RoboflowError) as ctx:
            self.upload()

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("nope", str(ctx.exception))

    @responses.activate
    def test_weights_upload_error_raises(self):
        responses.add(responses.POST, self.PREPARE_URL, json={"url": SIGNED_URL})
        responses.add(responses.PUT, SIGNED_URL, status=500)

        with self.assertRaises(RoboflowError) as ctx:
            self.upload()

        self.assertEqual(ctx.exception.status_code, 500)

    @responses.activate
    def test_success_prints_status_url(self):
        responses.add(responses.POST, self.PREPARE_URL, json={"url": SIGNED_URL})
        responses.add(responses.PUT, SIGNED_URL, status=200)

        with patch("builtins.print") as mock_print:
            self.upload()

        printed = " ".join(str(call.args[0]) for call in mock_print.call_args_list)
        self.assertIn("View the status of your deployment for project proj", printed)


class TestCliModelUpload(_WeightsDirMixin):
    @patch("roboflow.Roboflow")
    def test_failed_upload_exits_non_zero(self, mock_rf_cls):
        from roboflow.cli.handlers.model import _upload_model

        version = mock_rf_cls.return_value.workspace.return_value.project.return_value.version.return_value
        version.deploy.side_effect = RoboflowError("An error occurred when uploading the model: 500", status_code=500)
        args = SimpleNamespace(
            json=True,
            api_key="test-key",
            workspace="test-ws",
            project=["proj"],
            version_number=1,
            model_type="yolov8n",
            model_path=self.model_path,
            filename="weights/best.pt",
            model_name=None,
        )
        stdout, stderr = io.StringIO(), io.StringIO()

        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as ctx:
                _upload_model(args)

        self.assertEqual(ctx.exception.code, 1)
        self.assertNotIn("uploaded", stdout.getvalue())
        self.assertIn("uploading the model", json.loads(stderr.getvalue())["error"]["message"])


if __name__ == "__main__":
    unittest.main()
