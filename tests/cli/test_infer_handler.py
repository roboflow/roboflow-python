"""Unit tests for roboflow.cli.handlers.infer."""

import io
import json
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


class TestInferRegister(unittest.TestCase):
    """Verify infer handler registers as a top-level command."""

    def test_register_adds_infer_parser(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["infer", "image.jpg", "-m", "proj/1"])
        self.assertEqual(args.command, "infer")
        self.assertEqual(args.file, "image.jpg")
        self.assertEqual(args.model, "proj/1")
        self.assertTrue(callable(args.func))

    def test_infer_default_values(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["infer", "img.png", "-m", "proj/1"])
        self.assertEqual(args.confidence, 0.5)
        self.assertEqual(args.overlap, 0.5)
        self.assertIsNone(args.type)

    def test_infer_all_flags(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "infer", "img.png",
            "-m", "proj/1",
            "-c", "0.7",
            "-o", "0.3",
            "-t", "object-detection",
        ])
        self.assertAlmostEqual(args.confidence, 0.7)
        self.assertAlmostEqual(args.overlap, 0.3)
        self.assertEqual(args.type, "object-detection")

    def test_infer_type_choices(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["infer", "img.png", "-m", "proj/1", "-t", "invalid-type"])


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


if __name__ == "__main__":
    unittest.main()
