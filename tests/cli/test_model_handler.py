"""Unit tests for roboflow.cli.handlers.model."""

import io
import json
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


class TestModelRegister(unittest.TestCase):
    """Verify model handler registers expected subcommands."""

    def test_register_adds_model_parser(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["model"])
        self.assertEqual(args.command, "model")

    def test_model_list_parser(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["model", "list", "-p", "my-project"])
        self.assertEqual(args.project, "my-project")
        self.assertTrue(callable(args.func))

    def test_model_get_parser(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["model", "get", "my-ws/my-model"])
        self.assertEqual(args.model_url, "my-ws/my-model")
        self.assertTrue(callable(args.func))

    def test_model_upload_parser(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "model", "upload",
            "-p", "proj1",
            "-t", "yolov8",
            "-m", "/path/to/model",
        ])
        self.assertEqual(args.project, ["proj1"])
        self.assertEqual(args.model_type, "yolov8")
        self.assertEqual(args.model_path, "/path/to/model")
        self.assertEqual(args.filename, "weights/best.pt")
        self.assertTrue(callable(args.func))

    def test_model_upload_multiple_projects(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "model", "upload",
            "-p", "proj1", "-p", "proj2",
            "-t", "yolov8",
            "-m", "/path/to/model",
        ])
        self.assertEqual(args.project, ["proj1", "proj2"])


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


if __name__ == "__main__":
    unittest.main()
