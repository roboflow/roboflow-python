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


class TestTrainSubcommandsRegister(unittest.TestCase):
    """train cancel/stop/results subcommands register correctly."""

    def test_cancel_help(self) -> None:
        result = runner.invoke(app, ["train", "cancel", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("cancel", result.output.lower())

    def test_stop_help(self) -> None:
        result = runner.invoke(app, ["train", "stop", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_results_help(self) -> None:
        result = runner.invoke(app, ["train", "results", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_delete_help(self) -> None:
        result = runner.invoke(app, ["train", "delete", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("trash", result.output.lower())

    def test_restore_help(self) -> None:
        result = runner.invoke(app, ["train", "restore", "--help"])
        self.assertEqual(result.exit_code, 0)


class TestTrainCancelStopResults(unittest.TestCase):
    """_cancel / _stop / _results business logic."""

    def _args(self, **kwargs: object) -> types.SimpleNamespace:
        defaults = {
            "json": True,
            "api_key": "test-key",
            "workspace": "test-ws",
            "target": "my-project/3",
            "continue_if_no_refund": False,
            "quiet": True,
        }
        defaults.update(kwargs)
        return types.SimpleNamespace(**defaults)

    def _capture_stdout(self, fn, args):
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            fn(args)
        finally:
            sys.stdout = old
        return buf.getvalue()

    @patch("roboflow.adapters.rfapi.cancel_version_training")
    def test_cancel_success(self, mock_cancel: MagicMock) -> None:
        from roboflow.cli.handlers.train import _cancel

        mock_cancel.return_value = {"refund": True}
        out = self._capture_stdout(_cancel, self._args())

        mock_cancel.assert_called_once_with("test-key", "test-ws", "my-project", "3", continue_if_no_refund=False)
        result = json.loads(out)
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(result["project"], "my-project")
        self.assertEqual(result["version"], "3")
        self.assertTrue(result.get("refund"))

    @patch("roboflow.adapters.rfapi.cancel_version_training")
    def test_cancel_409_surfaces_hint(self, mock_cancel: MagicMock) -> None:
        from roboflow.adapters import rfapi
        from roboflow.cli.handlers.train import _cancel

        mock_cancel.side_effect = rfapi.RoboflowError("Cannot cancel non-running train job.")
        buf = io.StringIO()
        old = sys.stderr
        sys.stderr = buf
        try:
            with self.assertRaises(SystemExit) as cm:
                _cancel(self._args())
        finally:
            sys.stderr = old
        self.assertEqual(cm.exception.code, 3)
        err = json.loads(buf.getvalue())
        self.assertIn("Cannot cancel", err["error"]["message"])
        self.assertIn("in-flight", err["error"].get("hint", ""))

    @patch("roboflow.adapters.rfapi.stop_version_training")
    def test_stop_success(self, mock_stop: MagicMock) -> None:
        from roboflow.cli.handlers.train import _stop

        mock_stop.return_value = {"success": True}
        out = self._capture_stdout(_stop, self._args())

        mock_stop.assert_called_once_with("test-key", "test-ws", "my-project", "3")
        result = json.loads(out)
        self.assertEqual(result["status"], "stop_requested")

    @patch("roboflow.adapters.rfapi.delete_version_training")
    def test_delete_success(self, mock_delete: MagicMock) -> None:
        from roboflow.cli.handlers.train import _delete

        mock_delete.return_value = {"trainingId": "t-1", "inTrash": True, "alreadyInTrash": False}
        out = self._capture_stdout(_delete, self._args(training_id="t-1"))

        mock_delete.assert_called_once_with("test-key", "test-ws", "my-project", "3", training_id="t-1")
        result = json.loads(out)
        self.assertEqual(result["status"], "in_trash")
        self.assertTrue(result["inTrash"])

    @patch("roboflow.adapters.rfapi.delete_version_training")
    def test_delete_in_progress_surfaces_hint(self, mock_delete: MagicMock) -> None:
        from roboflow.adapters import rfapi
        from roboflow.cli.handlers.train import _delete

        mock_delete.side_effect = rfapi.RoboflowError(
            "This training is still in progress. Stop or cancel it before deleting it."
        )
        buf = io.StringIO()
        old = sys.stderr
        sys.stderr = buf
        try:
            with self.assertRaises(SystemExit) as cm:
                _delete(self._args(training_id=None))
        finally:
            sys.stderr = old
        self.assertEqual(cm.exception.code, 3)
        err = json.loads(buf.getvalue())
        self.assertIn("in progress", err["error"]["message"])
        self.assertIn("train stop", err["error"].get("hint", ""))

    @patch("roboflow.adapters.rfapi.restore_version_training")
    def test_restore_success(self, mock_restore: MagicMock) -> None:
        from roboflow.cli.handlers.train import _restore

        mock_restore.return_value = {"trainingId": "t-1", "restored": True}
        out = self._capture_stdout(_restore, self._args(training_id="t-1"))

        mock_restore.assert_called_once_with("test-key", "test-ws", "my-project", "3", training_id="t-1")
        result = json.loads(out)
        self.assertEqual(result["status"], "restored")

    @patch("roboflow.adapters.rfapi.get_training_results")
    def test_results_nas_run(self, mock_get: MagicMock) -> None:
        from roboflow.cli.handlers.train import _results

        mock_get.return_value = {
            "trainingId": "test-ws/my-project/3",
            "status": "finished",
            "jobType": "nas",
            "modelGroup": "rfdetrNasGroup-3",
            "modelCount": 5,
            "recommendedByHardware": {"gpu": "my-project-3-nas-gpu-a"},
            "models": [{"modelId": "my-project-3-nas-gpu-a"}],
        }
        out = self._capture_stdout(_results, self._args())
        result = json.loads(out)
        self.assertEqual(result["jobType"], "nas")
        self.assertEqual(result["modelCount"], 5)


if __name__ == "__main__":
    unittest.main()
