"""Unit tests for roboflow.cli.handlers.infer."""

import io
import json
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from roboflow.cli import app

runner = CliRunner()


class TestInferRegister(unittest.TestCase):
    """Verify infer handler registers as a top-level command."""

    def test_register_adds_infer_parser(self) -> None:
        result = runner.invoke(app, ["infer", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("model", result.output.lower())

    def test_infer_help_shows_options(self) -> None:
        result = runner.invoke(app, ["infer", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("confidence", result.output.lower())
        self.assertIn("overlap", result.output.lower())


class TestInferHandler(unittest.TestCase):
    """Test _infer handler function."""

    def _make_args(self, **kwargs: object) -> types.SimpleNamespace:
        defaults = {
            "json": False,
            "api_key": "test-key",
            "workspace": "test-ws",
            "model": "test-project/1",
            "file": "test.jpg",
            "confidence": 0.5,
            "overlap": 0.5,
            "type": "object-detection",
        }
        defaults.update(kwargs)
        return types.SimpleNamespace(**defaults)

    @patch("roboflow.models.object_detection.ObjectDetectionModel")
    def test_infer_text_output(self, mock_model_cls: MagicMock) -> None:
        from roboflow.cli.handlers.infer import _infer

        mock_group = MagicMock()
        mock_group.__str__ = lambda self: "detection results"
        mock_group.__iter__ = lambda self: iter([])
        mock_model = MagicMock()
        mock_model.predict.return_value = mock_group
        mock_model_cls.return_value = mock_model

        args = self._make_args()
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            _infer(args)
        finally:
            sys.stdout = old_stdout

        self.assertIn("detection results", buf.getvalue())

    @patch("roboflow.models.object_detection.ObjectDetectionModel")
    def test_infer_json_output(self, mock_model_cls: MagicMock) -> None:
        from roboflow.cli.handlers.infer import _infer

        mock_pred = MagicMock()
        mock_pred.json.return_value = {"class": "dog", "confidence": 0.9}
        mock_group = MagicMock()
        mock_group.__iter__ = lambda self: iter([mock_pred])
        mock_model = MagicMock()
        mock_model.predict.return_value = mock_group
        mock_model_cls.return_value = mock_model

        args = self._make_args(json=True)
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            _infer(args)
        finally:
            sys.stdout = old_stdout

        result = json.loads(buf.getvalue())
        self.assertIsInstance(result, list)
        self.assertEqual(result[0]["class"], "dog")

    @patch("roboflow.models.object_detection.ObjectDetectionModel")
    @patch("roboflow.adapters.rfapi.get_project")
    def test_infer_auto_detects_type(self, mock_get_project: MagicMock, mock_model_cls: MagicMock) -> None:
        from roboflow.cli.handlers.infer import _infer

        mock_get_project.return_value = {"project": {"type": "object-detection"}}
        mock_group = MagicMock()
        mock_group.__str__ = lambda self: "results"
        mock_group.__iter__ = lambda self: iter([])
        mock_model = MagicMock()
        mock_model.predict.return_value = mock_group
        mock_model_cls.return_value = mock_model

        args = self._make_args(type=None)
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            _infer(args)
        finally:
            sys.stdout = old_stdout

        mock_get_project.assert_called_once()

    @patch("roboflow.config.load_roboflow_api_key", return_value=None)
    def test_infer_no_api_key(self, _mock_key: MagicMock) -> None:
        from roboflow.cli.handlers.infer import _infer

        args = self._make_args(api_key=None)
        with self.assertRaises(SystemExit) as ctx:
            _infer(args)
        self.assertEqual(ctx.exception.code, 2)

    @patch("roboflow.models.object_detection.ObjectDetectionModel")
    def test_infer_confidence_converted_to_percentage(self, mock_model_cls: MagicMock) -> None:
        from roboflow.cli.handlers.infer import _infer

        mock_group = MagicMock()
        mock_group.__str__ = lambda self: "results"
        mock_group.__iter__ = lambda self: iter([])
        mock_model = MagicMock()
        mock_model.predict.return_value = mock_group
        mock_model_cls.return_value = mock_model

        args = self._make_args(confidence=0.7, overlap=0.3)
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            _infer(args)
        finally:
            sys.stdout = old_stdout

        mock_model.predict.assert_called_once_with("test.jpg", confidence=70, overlap=30)


class TestInferVLM(unittest.TestCase):
    """VLM (text-image-pairs) path returns raw dict passthrough."""

    def _make_args(self, **kwargs: object) -> types.SimpleNamespace:
        defaults = {
            "json": False,
            "api_key": "test-key",
            "workspace": "test-ws",
            "model": "test-project/1",
            "file": "https://example.com/img.jpg",
            "confidence": 0.5,
            "overlap": 0.5,
            "type": "text-image-pairs",
        }
        defaults.update(kwargs)
        return types.SimpleNamespace(**defaults)

    @patch("roboflow.models.vlm.VLMModel")
    def test_infer_vlm_json_passthrough(self, mock_model_cls: MagicMock) -> None:
        from roboflow.cli.handlers.infer import _infer

        raw = {"inference_id": "abc", "response": {">": "caption text"}}
        mock_model = MagicMock()
        mock_model.predict.return_value = raw
        mock_model_cls.return_value = mock_model

        args = self._make_args(json=True)
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            _infer(args)
        finally:
            sys.stdout = old_stdout

        result = json.loads(buf.getvalue())
        self.assertEqual(result, raw)

    @patch("roboflow.models.vlm.VLMModel")
    def test_infer_vlm_skips_confidence_overlap(self, mock_model_cls: MagicMock) -> None:
        from roboflow.cli.handlers.infer import _infer

        mock_model = MagicMock()
        mock_model.predict.return_value = {"ok": True}
        mock_model_cls.return_value = mock_model

        args = self._make_args(confidence=0.7, overlap=0.3)
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            _infer(args)
        finally:
            sys.stdout = old_stdout

        mock_model.predict.assert_called_once_with("https://example.com/img.jpg")


if __name__ == "__main__":
    unittest.main()
