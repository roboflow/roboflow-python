"""Unit tests for roboflow.cli.handlers.train."""

import io
import json
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from roboflow.cli import app

runner = CliRunner()


class TestTrainRegister(unittest.TestCase):
    """Verify train handler registers expected subcommands."""

    def test_train_help(self) -> None:
        result = runner.invoke(app, ["train", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_train_start_help(self) -> None:
        result = runner.invoke(app, ["train", "start", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("project", result.output.lower())
        self.assertIn("version", result.output.lower())


class TestTrainStart(unittest.TestCase):
    """Test _start handler function."""

    def _make_args(self, **kwargs: object) -> types.SimpleNamespace:
        defaults = {
            "json": False,
            "api_key": "test-key",
            "workspace": "test-ws",
            "project": "my-project",
            "version_number": 1,
            "model_type": None,
            "checkpoint": None,
            "speed": None,
            "epochs": None,
        }
        defaults.update(kwargs)
        return types.SimpleNamespace(**defaults)

    @patch("roboflow.adapters.rfapi.start_version_training")
    def test_start_success(self, mock_train: MagicMock) -> None:
        from roboflow.cli.handlers.train import _start

        mock_train.return_value = True

        args = self._make_args(json=True)
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            _start(args)
        finally:
            sys.stdout = old_stdout

        result = json.loads(buf.getvalue())
        self.assertEqual(result["status"], "training_started")
        self.assertEqual(result["project"], "my-project")
        self.assertEqual(result["version"], 1)

    @patch("roboflow.adapters.rfapi.start_version_training")
    def test_start_with_all_options(self, mock_train: MagicMock) -> None:
        from roboflow.cli.handlers.train import _start

        mock_train.return_value = True

        args = self._make_args(
            json=True,
            model_type="yolov8n",
            checkpoint="abc",
            speed="fast",
            epochs=50,
        )
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            _start(args)
        finally:
            sys.stdout = old_stdout

        mock_train.assert_called_once_with(
            "test-key",
            "test-ws",
            "my-project",
            "1",
            speed="fast",
            checkpoint="abc",
            model_type="yolov8n",
            epochs=50,
        )

    @patch("roboflow.adapters.rfapi.start_version_training")
    def test_start_api_error(self, mock_train: MagicMock) -> None:
        from roboflow.adapters.rfapi import RoboflowError
        from roboflow.cli.handlers.train import _start

        mock_train.side_effect = RoboflowError("training failed")

        args = self._make_args()
        with self.assertRaises(SystemExit) as ctx:
            _start(args)
        self.assertEqual(ctx.exception.code, 1)

    @patch("roboflow.config.load_roboflow_api_key", return_value=None)
    def test_start_no_api_key(self, _mock_key: MagicMock) -> None:
        from roboflow.cli.handlers.train import _start

        args = self._make_args(api_key=None)
        with self.assertRaises(SystemExit) as ctx:
            _start(args)
        self.assertEqual(ctx.exception.code, 2)

    @patch("roboflow.adapters.rfapi.start_version_training")
    def test_start_json_error_not_double_encoded(self, mock_train: MagicMock) -> None:
        from roboflow.adapters.rfapi import RoboflowError
        from roboflow.cli.handlers.train import _start

        # Simulate API returning a JSON error string
        mock_train.side_effect = RoboflowError('{"error": {"message": "Unsupported request"}}')

        args = self._make_args(json=True)
        buf = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = buf
        try:
            with self.assertRaises(SystemExit):
                _start(args)
        finally:
            sys.stderr = old_stderr

        result = json.loads(buf.getvalue())
        # Should be a parsed object, not a double-encoded JSON string
        self.assertIsInstance(result["error"], dict)
        self.assertEqual(result["error"]["message"], "Unsupported request")


if __name__ == "__main__":
    unittest.main()
