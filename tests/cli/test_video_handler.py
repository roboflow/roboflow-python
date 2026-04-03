"""Tests for the video CLI handler."""

import json
import unittest
from unittest.mock import patch

from typer.testing import CliRunner

from roboflow.cli import app

runner = CliRunner()


class TestVideoRegistration(unittest.TestCase):
    """Verify video handler registers expected subcommands."""

    def test_video_app_exists(self) -> None:
        from roboflow.cli.handlers.video import video_app

        self.assertIsNotNone(video_app)

    def test_video_infer_exists(self) -> None:
        result = runner.invoke(app, ["video", "infer", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_video_status_exists(self) -> None:
        result = runner.invoke(app, ["video", "status", "--help"])
        self.assertEqual(result.exit_code, 0)


class TestVideoStatus(unittest.TestCase):
    """Test video status handler."""

    @patch("roboflow.config.load_roboflow_api_key", return_value=None)
    def test_status_no_api_key(self, _mock_key) -> None:
        result = runner.invoke(app, ["--json", "video", "status", "job-123"])
        self.assertNotEqual(result.exit_code, 0)

    @patch("roboflow.adapters.rfapi.get_video_job_status")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_status_success(self, _mock_key, mock_api) -> None:
        mock_api.return_value = {"status": "completed", "progress": "100%"}
        result = runner.invoke(app, ["video", "status", "job-abc"])
        self.assertIn("job-abc", result.output)
        self.assertIn("completed", result.output)

    @patch("roboflow.adapters.rfapi.get_video_job_status")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_status_json_output(self, _mock_key, mock_api) -> None:
        mock_api.return_value = {"status": "processing", "progress": "50%"}
        result = runner.invoke(app, ["--json", "video", "status", "job-abc"])
        data = json.loads(result.output)
        self.assertEqual(data["status"], "processing")

    @patch("roboflow.adapters.rfapi.get_video_job_status")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_status_passes_job_id_to_api(self, _mock_key, mock_api) -> None:
        mock_api.return_value = {"status": "completed"}
        runner.invoke(app, ["video", "status", "my-unique-job-777"])
        mock_api.assert_called_once_with("fake-key", "my-unique-job-777")


if __name__ == "__main__":
    unittest.main()
