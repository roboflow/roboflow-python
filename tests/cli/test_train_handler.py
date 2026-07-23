"""Unit tests for roboflow.cli.handlers.train."""

import io
import json
import os
import re
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from roboflow.cli import app

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


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


RECIPE_RESPONSE = {
    "modelType": "rfdetr-medium",
    "family": "rf-detr",
    "taskType": "object-detection",
    "schema": {"hyperparameters": [{"key": "lr", "type": "float"}]},
    "template": {
        "schema_version": 1,
        "input": {},
        "online_preprocessing": [],
        "online_augmentation": {"splits": ["train"], "steps": []},
        "source_version": {},
        "hyperparameters": {},
    },
    "usage": "...",
}


class TestTrainRecipe(unittest.TestCase):
    """`train recipe` describe command."""

    def _make_args(self, **kwargs: object) -> types.SimpleNamespace:
        defaults = {
            "json": True,
            "api_key": "test-key",
            "workspace": "test-ws",
            "project": "my-project",
            "version_number": 3,
            "model_type": "rfdetr-medium",
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

    def test_recipe_help(self) -> None:
        result = runner.invoke(app, ["train", "recipe", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("model", result.output.lower())

    @patch("roboflow.adapters.rfapi.get_train_recipe")
    def test_recipe_prints_response_as_json(self, mock_recipe: MagicMock) -> None:
        from roboflow.cli.handlers.train import _recipe

        mock_recipe.return_value = RECIPE_RESPONSE
        out = self._capture_stdout(_recipe, self._make_args())

        mock_recipe.assert_called_once_with("test-key", "test-ws", "my-project", "3", model_type="rfdetr-medium")
        self.assertEqual(json.loads(out), RECIPE_RESPONSE)

    @patch("roboflow.adapters.rfapi.get_train_recipe")
    def test_recipe_via_cli_runner(self, mock_recipe: MagicMock) -> None:
        mock_recipe.return_value = RECIPE_RESPONSE
        result = runner.invoke(
            app,
            [
                "--api-key",
                "test-key",
                "--workspace",
                "test-ws",
                "train",
                "recipe",
                "-p",
                "my-project",
                "-v",
                "3",
                "-m",
                "rfdetr-medium",
            ],
        )
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertEqual(json.loads(result.output), RECIPE_RESPONSE)

    @patch("roboflow.adapters.rfapi.get_train_recipe")
    def test_recipe_api_error(self, mock_recipe: MagicMock) -> None:
        from roboflow.adapters.rfapi import RoboflowError
        from roboflow.cli.handlers.train import _recipe

        mock_recipe.side_effect = RoboflowError("no recipe for model type")
        with self.assertRaises(SystemExit) as ctx:
            _recipe(self._make_args())
        self.assertEqual(ctx.exception.code, 1)

    @patch("roboflow.config.load_roboflow_api_key", return_value=None)
    def test_recipe_no_api_key(self, _mock_key: MagicMock) -> None:
        from roboflow.cli.handlers.train import _recipe

        with self.assertRaises(SystemExit) as ctx:
            _recipe(self._make_args(api_key=None))
        self.assertEqual(ctx.exception.code, 2)


class TestTrainStartV2(unittest.TestCase):
    """`train start` with --train-recipe goes through v2 create_training_v2."""

    def _make_args(self, **kwargs: object) -> types.SimpleNamespace:
        defaults = {
            "json": True,
            "api_key": "test-key",
            "workspace": "test-ws",
            "project": "my-project",
            "version_number": 3,
            "model_type": "rfdetr-medium",
            "checkpoint": None,
            "speed": None,
            "epochs": None,
            "train_recipe": None,
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

    @patch("roboflow.adapters.rfapi.get_version")
    @patch("roboflow.adapters.rfapi.create_training_v2")
    @patch("roboflow.adapters.rfapi.get_train_recipe")
    def test_start_with_train_recipe_submits_as_is(
        self, mock_recipe: MagicMock, mock_create: MagicMock, mock_get_version: MagicMock
    ) -> None:
        from roboflow.adapters.rfapi import RoboflowError
        from roboflow.cli.handlers.train import _start

        mock_create.return_value = {"trainingId": "t-2", "status": "queued"}
        mock_get_version.side_effect = RoboflowError("offline")

        recipe = {"schema_version": 1, "hyperparameters": {"lr": 0.5}}
        args = self._make_args(train_recipe=json.dumps(recipe))
        out = self._capture_stdout(_start, args)

        mock_recipe.assert_not_called()
        self.assertEqual(mock_create.call_args.kwargs["train_recipe"], recipe)
        self.assertEqual(json.loads(out)["trainingId"], "t-2")

    def test_start_with_invalid_train_recipe_json(self) -> None:
        from roboflow.cli.handlers.train import _start

        args = self._make_args(train_recipe="[unterminated")
        buf = io.StringIO()
        old = sys.stderr
        sys.stderr = buf
        try:
            with self.assertRaises(SystemExit) as ctx:
                _start(args)
        finally:
            sys.stderr = old
        self.assertEqual(ctx.exception.code, 1)
        err = json.loads(buf.getvalue())
        self.assertIn("Invalid JSON", err["error"]["message"])

    @patch("roboflow.adapters.rfapi.create_training_v2")
    def test_start_with_non_object_train_recipe_json(self, mock_create: MagicMock) -> None:
        from roboflow.cli.handlers.train import _start

        args = self._make_args(train_recipe="[1]")
        buf = io.StringIO()
        old = sys.stderr
        sys.stderr = buf
        try:
            with self.assertRaises(SystemExit) as ctx:
                _start(args)
        finally:
            sys.stderr = old
        self.assertEqual(ctx.exception.code, 1)
        err = json.loads(buf.getvalue())
        self.assertIn("must be a JSON object", err["error"]["message"])
        self.assertIn("list", err["error"]["message"])
        mock_create.assert_not_called()  # rejected before any network call

    @patch("roboflow.adapters.rfapi.create_training_v2")
    def test_start_train_recipe_requires_model_type(self, mock_create: MagicMock) -> None:
        from roboflow.cli.handlers.train import _start

        args = self._make_args(train_recipe=json.dumps({"schema_version": 1}), model_type=None)
        buf = io.StringIO()
        old = sys.stderr
        sys.stderr = buf
        try:
            with self.assertRaises(SystemExit) as ctx:
                _start(args)
        finally:
            sys.stderr = old
        self.assertEqual(ctx.exception.code, 1)
        err = json.loads(buf.getvalue())
        self.assertIn("requires a model type", err["error"]["message"])
        mock_create.assert_not_called()  # rejected before any network call

    @patch("roboflow.adapters.rfapi.get_version")
    @patch("roboflow.adapters.rfapi.create_training_v2")
    def test_start_with_train_recipe_from_file(self, mock_create: MagicMock, mock_get_version: MagicMock) -> None:
        import tempfile

        from roboflow.adapters.rfapi import RoboflowError
        from roboflow.cli.handlers.train import _start

        mock_create.return_value = {"trainingId": "t-5", "status": "queued"}
        mock_get_version.side_effect = RoboflowError("offline")

        recipe = {"schema_version": 1, "hyperparameters": {"lr": 0.0003}}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "train_recipe.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(recipe, f)
            args = self._make_args(train_recipe=f"@{path}")
            self._capture_stdout(_start, args)

        self.assertEqual(mock_create.call_args.kwargs["train_recipe"], recipe)

    @patch("roboflow.adapters.rfapi.create_training_v2")
    def test_start_with_missing_train_recipe_file(self, mock_create: MagicMock) -> None:
        from roboflow.cli.handlers.train import _start

        args = self._make_args(train_recipe="@/nonexistent/train_recipe.json")
        buf = io.StringIO()
        old = sys.stderr
        sys.stderr = buf
        try:
            with self.assertRaises(SystemExit) as ctx:
                _start(args)
        finally:
            sys.stderr = old
        self.assertEqual(ctx.exception.code, 1)
        err = json.loads(buf.getvalue())
        self.assertIn("Cannot read --train-recipe file", err["error"]["message"])
        mock_create.assert_not_called()  # rejected before any network call

    @patch("roboflow.adapters.rfapi.create_training_v2")
    def test_start_with_invalid_json_in_train_recipe_file(self, mock_create: MagicMock) -> None:
        import tempfile

        from roboflow.cli.handlers.train import _start

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "train_recipe.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write("{not json")
            args = self._make_args(train_recipe=f"@{path}")
            buf = io.StringIO()
            old = sys.stderr
            sys.stderr = buf
            try:
                with self.assertRaises(SystemExit) as ctx:
                    _start(args)
            finally:
                sys.stderr = old
        self.assertEqual(ctx.exception.code, 1)
        err = json.loads(buf.getvalue())
        self.assertIn("Invalid JSON in --train-recipe file", err["error"]["message"])
        mock_create.assert_not_called()

    @patch("roboflow.adapters.rfapi.get_version")
    @patch("roboflow.adapters.rfapi.create_training_v2")
    def test_start_with_train_recipe_and_epochs_folds_epochs(
        self, mock_create: MagicMock, mock_get_version: MagicMock
    ) -> None:
        from roboflow.adapters.rfapi import RoboflowError
        from roboflow.cli.handlers.train import _start

        mock_create.return_value = {"trainingId": "t-4", "status": "queued"}
        mock_get_version.side_effect = RoboflowError("offline")

        recipe = {"schema_version": 1, "hyperparameters": {"lr": 0.5}}
        args = self._make_args(train_recipe=json.dumps(recipe), epochs=50)
        self._capture_stdout(_start, args)

        create_kwargs = mock_create.call_args.kwargs
        self.assertEqual(create_kwargs["train_recipe"]["hyperparameters"], {"lr": 0.5, "epochs": 50})
        self.assertEqual(create_kwargs["epochs"], 50)

    def test_start_help_shows_train_recipe_flag_only(self) -> None:
        result = runner.invoke(app, ["train", "start", "--help"])
        self.assertEqual(result.exit_code, 0)
        output = _strip_ansi(result.output)
        self.assertIn("--train-recipe", output)
        self.assertNotIn("--hyperparameters", output)


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
