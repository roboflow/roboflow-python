"""Unit tests for roboflow.cli.handlers.model."""

import io
import json
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from roboflow.cli import app

runner = CliRunner()


class TestModelRegister(unittest.TestCase):
    """Verify model handler registers expected subcommands."""

    def test_model_help(self) -> None:
        result = runner.invoke(app, ["model", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("list", result.output)
        self.assertIn("get", result.output)
        self.assertIn("upload", result.output)

    def test_model_list_help(self) -> None:
        result = runner.invoke(app, ["model", "list", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_model_get_help(self) -> None:
        result = runner.invoke(app, ["model", "get", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_model_upload_help(self) -> None:
        result = runner.invoke(app, ["model", "upload", "--help"])
        self.assertEqual(result.exit_code, 0)


class TestModelGet(unittest.TestCase):
    """Test _get_model handler."""

    def _make_args(self, **kwargs: object) -> types.SimpleNamespace:
        defaults = {
            "json": False,
            "api_key": "test-key",
            "workspace": "test-ws",
            "model_url": "test-ws/test-project",
        }
        defaults.update(kwargs)
        return types.SimpleNamespace(**defaults)

    @patch("roboflow.adapters.rfapi.get_project")
    def test_get_model_success(self, mock_get_project: MagicMock) -> None:
        from roboflow.cli.handlers.model import _get_model

        mock_get_project.return_value = {"project": {"name": "test"}}

        args = self._make_args(json=True)
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            _get_model(args)
        finally:
            sys.stdout = old_stdout

        result = json.loads(buf.getvalue())
        self.assertEqual(result["project"]["name"], "test")

    @patch("roboflow.adapters.rfapi.get_version")
    def test_get_model_with_version(self, mock_get_version: MagicMock) -> None:
        from roboflow.cli.handlers.model import _get_model

        mock_get_version.return_value = {"version": {"id": "test/1"}}

        args = self._make_args(model_url="test-ws/test-project/1", json=True)
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            _get_model(args)
        finally:
            sys.stdout = old_stdout

        result = json.loads(buf.getvalue())
        self.assertIn("version", result)
        mock_get_version.assert_called_once()

    @patch("roboflow.config.load_roboflow_api_key", return_value=None)
    def test_get_model_no_api_key(self, _mock_key: MagicMock) -> None:
        from roboflow.cli.handlers.model import _get_model

        args = self._make_args(api_key=None)
        with self.assertRaises(SystemExit) as ctx:
            _get_model(args)
        self.assertEqual(ctx.exception.code, 2)


class TestModelUpload(unittest.TestCase):
    """Test _upload_model handler."""

    def _make_args(self, **kwargs: object) -> types.SimpleNamespace:
        defaults = {
            "json": False,
            "api_key": "test-key",
            "workspace": "test-ws",
            "project": ["proj1"],
            "version_number": 1,
            "model_type": "yolov8",
            "model_path": "/path/to/model",
            "filename": "weights/best.pt",
            "model_name": None,
        }
        defaults.update(kwargs)
        return types.SimpleNamespace(**defaults)

    @patch("roboflow.Roboflow")
    def test_upload_single_version(self, mock_rf_cls: MagicMock) -> None:
        from roboflow.cli.handlers.model import _upload_model

        mock_version = MagicMock()
        mock_project = MagicMock()
        mock_project.version.return_value = mock_version
        mock_workspace = MagicMock()
        mock_workspace.project.return_value = mock_project
        mock_rf = MagicMock()
        mock_rf.workspace.return_value = mock_workspace
        mock_rf_cls.return_value = mock_rf

        args = self._make_args(json=True)
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            _upload_model(args)
        finally:
            sys.stdout = old_stdout

        result = json.loads(buf.getvalue())
        self.assertEqual(result["status"], "uploaded")
        mock_version.deploy.assert_called_once_with("yolov8", "/path/to/model", "weights/best.pt")

    @patch("roboflow.Roboflow")
    def test_upload_multi_project(self, mock_rf_cls: MagicMock) -> None:
        from roboflow.cli.handlers.model import _upload_model

        mock_workspace = MagicMock()
        mock_rf = MagicMock()
        mock_rf.workspace.return_value = mock_workspace
        mock_rf_cls.return_value = mock_rf

        args = self._make_args(project=["proj1", "proj2"], version_number=None, model_name="my-model", json=True)
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            _upload_model(args)
        finally:
            sys.stdout = old_stdout

        result = json.loads(buf.getvalue())
        self.assertEqual(result["status"], "uploaded")
        mock_workspace.deploy_model.assert_called_once()

    @patch("roboflow.Roboflow")
    def test_upload_no_project_errors(self, mock_rf_cls: MagicMock) -> None:
        from roboflow.cli.handlers.model import _upload_model

        mock_workspace = MagicMock()
        mock_rf = MagicMock()
        mock_rf.workspace.return_value = mock_workspace
        mock_rf_cls.return_value = mock_rf

        args = self._make_args(project=None, version_number=None)
        with self.assertRaises(SystemExit):
            _upload_model(args)


class TestParseErrorMessage(unittest.TestCase):
    """Test _parse_error_message helper (centralized in _output.py)."""

    def test_plain_string(self) -> None:
        from roboflow.cli._output import _parse_error_message

        parsed, human = _parse_error_message("something broke")
        self.assertIsNone(parsed)
        self.assertEqual(human, "something broke")

    def test_json_with_nested_error(self) -> None:
        from roboflow.cli._output import _parse_error_message

        raw = '{"error": {"message": "Unsupported request"}}'
        parsed, human = _parse_error_message(raw)
        self.assertIsNotNone(parsed)
        self.assertEqual(human, "Unsupported request")

    def test_json_with_string_error(self) -> None:
        from roboflow.cli._output import _parse_error_message

        raw = '{"error": "Not found"}'
        parsed, human = _parse_error_message(raw)
        self.assertIsNotNone(parsed)
        self.assertEqual(human, "Not found")


class TestModelListError(unittest.TestCase):
    """Test _list_models handles API errors cleanly."""

    def _make_args(self, **kwargs: object) -> types.SimpleNamespace:
        defaults = {
            "json": True,
            "api_key": "test-key",
            "workspace": "test-ws",
            "project": "nonexistent-project",
        }
        defaults.update(kwargs)
        return types.SimpleNamespace(**defaults)

    @patch("roboflow.Roboflow")
    def test_list_models_project_not_found(self, mock_rf_cls: MagicMock) -> None:
        from roboflow.cli.handlers.model import _list_models

        mock_workspace = MagicMock()
        mock_workspace.project.side_effect = RuntimeError("Project not found")
        mock_rf = MagicMock()
        mock_rf.workspace.return_value = mock_workspace
        mock_rf_cls.return_value = mock_rf

        args = self._make_args()
        buf = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = buf
        try:
            with self.assertRaises(SystemExit) as ctx:
                _list_models(args)
            self.assertEqual(ctx.exception.code, 3)
        finally:
            sys.stderr = old_stderr

        result = json.loads(buf.getvalue())
        self.assertIn("error", result)


class TestModelStarRegister(unittest.TestCase):
    """model star subcommand registers."""

    def test_star_help(self) -> None:
        result = runner.invoke(app, ["model", "star", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("nas", result.output.lower())

    def test_list_help_mentions_group(self) -> None:
        result = runner.invoke(app, ["model", "list", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("group", result.output.lower())


class TestModelStar(unittest.TestCase):
    """_star_model business logic."""

    def _args(self, **kwargs: object) -> types.SimpleNamespace:
        defaults = {
            "json": True,
            "api_key": "test-key",
            "workspace": "test-ws",
            "model_url": "my-proj-3-nas-gpu-b",
            "starred": True,
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

    @patch("roboflow.adapters.rfapi.favorite_nas_model")
    def test_star_success(self, mock_fav: MagicMock) -> None:
        from roboflow.cli.handlers.model import _star_model

        mock_fav.return_value = {"success": True, "model": {"url": "my-proj-3-nas-gpu-b"}}
        out = self._capture_stdout(_star_model, self._args())

        # Bare slug + -w, so workspace comes from args.workspace.
        mock_fav.assert_called_once_with("test-key", "test-ws", "my-proj-3-nas-gpu-b", starred=True)
        result = json.loads(out)
        self.assertTrue(result.get("success"))

    @patch("roboflow.adapters.rfapi.favorite_nas_model")
    def test_star_workspace_prefixed_url(self, mock_fav: MagicMock) -> None:
        """When the URL is `<ws>/<slug>`, the workspace flag overrides anyway."""
        from roboflow.cli.handlers.model import _star_model

        mock_fav.return_value = {"success": True, "model": {"url": "my-proj-3-nas-gpu-b"}}
        self._capture_stdout(_star_model, self._args(model_url="some-ws/my-proj-3-nas-gpu-b"))

        # -w wins over the prefix, slug is stripped of the workspace segment.
        mock_fav.assert_called_once_with("test-key", "test-ws", "my-proj-3-nas-gpu-b", starred=True)

    @patch("roboflow.adapters.rfapi.favorite_nas_model")
    def test_star_workspace_inferred_from_prefix(self, mock_fav: MagicMock) -> None:
        """No -w but `<ws>/<slug>` argument: workspace comes from the prefix."""
        from roboflow.cli.handlers.model import _star_model

        mock_fav.return_value = {"success": True, "model": {"url": "my-proj-3-nas-gpu-b"}}
        self._capture_stdout(
            _star_model,
            self._args(workspace=None, model_url="some-ws/my-proj-3-nas-gpu-b"),
        )

        mock_fav.assert_called_once_with("test-key", "some-ws", "my-proj-3-nas-gpu-b", starred=True)

    @patch("roboflow.adapters.rfapi.favorite_nas_model")
    def test_star_unstar_path(self, mock_fav: MagicMock) -> None:
        from roboflow.cli.handlers.model import _star_model

        mock_fav.return_value = {"success": True, "model": {"url": "my-proj-3-nas-gpu-b"}}
        self._capture_stdout(_star_model, self._args(starred=False))

        mock_fav.assert_called_once_with("test-key", "test-ws", "my-proj-3-nas-gpu-b", starred=False)

    @patch("roboflow.adapters.rfapi.favorite_nas_model")
    def test_star_non_nas_surfaces_hint(self, mock_fav: MagicMock) -> None:
        from roboflow.adapters import rfapi
        from roboflow.cli.handlers.model import _star_model

        mock_fav.side_effect = rfapi.RoboflowError(
            '{"code":"MODEL_NOT_NAS","message":"Starring is only supported for NAS-trained models."}'
        )
        buf = io.StringIO()
        old = sys.stderr
        sys.stderr = buf
        try:
            with self.assertRaises(SystemExit) as cm:
                _star_model(self._args())
        finally:
            sys.stderr = old
        self.assertEqual(cm.exception.code, 3)
        err = json.loads(buf.getvalue())
        # output_error parses the JSON body; the code surfaces alongside the message.
        self.assertEqual(err["error"].get("code"), "MODEL_NOT_NAS")
        self.assertIn("NAS-only", err["error"].get("hint", ""))


class TestModelListGroupFilter(unittest.TestCase):
    """_list_models with --group hits the public /models endpoint."""

    def _args(self, **kwargs: object) -> types.SimpleNamespace:
        defaults = {
            "json": True,
            "api_key": "test-key",
            "workspace": "test-ws",
            "project": "my-project",
            "group": None,
            "quiet": True,
        }
        defaults.update(kwargs)
        return types.SimpleNamespace(**defaults)

    @patch("roboflow.adapters.rfapi.list_project_models")
    def test_list_with_group_uses_public_endpoint(self, mock_list: MagicMock) -> None:
        from roboflow.cli.handlers.model import _list_models

        mock_list.return_value = [
            {
                "url": "my-ws/my-proj-3-nas-gpu-abc",
                "modelType": "rfdetr-nas",
                "metrics": {
                    "map50": 87.3,
                    "map5095": 57.6,
                    "hardware": "gpu",
                    "latency": 8.7,
                },
                "recommended": True,
            }
        ]

        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            _list_models(self._args(group="rfdetrNasGroup-3"))
        finally:
            sys.stdout = old

        mock_list.assert_called_once_with("test-key", "test-ws", "my-project", group="rfdetrNasGroup-3")
        rows = json.loads(buf.getvalue())
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["metrics"]["hardware"], "gpu")


if __name__ == "__main__":
    unittest.main()
