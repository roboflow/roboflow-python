"""Unit tests for roboflow.cli.handlers.annotation."""

import argparse
import io
import json
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


def _build_annotation_parser():
    """Build a minimal parser with just the annotation handler registered."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", "-j", action="store_true", default=False)
    parser.add_argument("--api-key", "-k", dest="api_key", default=None)
    parser.add_argument("--workspace", "-w", dest="workspace", default=None)
    parser.add_argument("--quiet", "-q", action="store_true", default=False)
    sub = parser.add_subparsers(title="commands", dest="command")

    from roboflow.cli.handlers.annotation import register

    register(sub)
    return parser


class TestAnnotationParserRegistration(unittest.TestCase):
    """Verify the annotation handler registers its subcommands."""

    def test_annotation_subcommand_exists(self):
        parser = _build_annotation_parser()
        args = parser.parse_args(["annotation", "batch", "list", "-p", "proj"])
        self.assertEqual(args.project, "proj")
        self.assertTrue(callable(args.func))

    def test_annotation_batch_get(self):
        parser = _build_annotation_parser()
        args = parser.parse_args(["annotation", "batch", "get", "batch-1", "-p", "proj"])
        self.assertEqual(args.batch_id, "batch-1")
        self.assertEqual(args.project, "proj")

    def test_annotation_job_list(self):
        parser = _build_annotation_parser()
        args = parser.parse_args(["annotation", "job", "list", "-p", "proj"])
        self.assertEqual(args.project, "proj")

    def test_annotation_job_get(self):
        parser = _build_annotation_parser()
        args = parser.parse_args(["annotation", "job", "get", "job-1", "-p", "proj"])
        self.assertEqual(args.job_id, "job-1")

    def test_annotation_job_create(self):
        parser = _build_annotation_parser()
        args = parser.parse_args(
            [
                "annotation",
                "job",
                "create",
                "-p",
                "proj",
                "--name",
                "my-job",
                "--batch",
                "batch-1",
                "--num-images",
                "10",
                "--labeler",
                "a@b.com",
                "--reviewer",
                "c@d.com",
            ]
        )
        self.assertEqual(args.name, "my-job")
        self.assertEqual(args.batch, "batch-1")
        self.assertEqual(args.num_images, 10)
        self.assertEqual(args.labeler, "a@b.com")
        self.assertEqual(args.reviewer, "c@d.com")


class TestAnnotationStub(unittest.TestCase):
    """Verify stub handlers print not-yet-implemented."""

    def test_stub_prints_message(self):
        from roboflow.cli._output import stub as _stub

        args = types.SimpleNamespace(json=False)

        buf = io.StringIO()
        old = sys.stderr
        sys.stderr = buf
        try:
            with self.assertRaises(SystemExit) as ctx:
                _stub(args)
            self.assertEqual(ctx.exception.code, 1)
        finally:
            sys.stderr = old

        self.assertIn("not yet implemented", buf.getvalue())

    def test_stub_json_mode(self):
        from roboflow.cli._output import stub as _stub

        args = types.SimpleNamespace(json=True)

        buf = io.StringIO()
        old = sys.stderr
        sys.stderr = buf
        try:
            with self.assertRaises(SystemExit) as ctx:
                _stub(args)
            self.assertEqual(ctx.exception.code, 1)
        finally:
            sys.stderr = old

        result = json.loads(buf.getvalue())
        self.assertIn("not yet implemented", result["error"]["message"])


# ---------------------------------------------------------------------------
# Behavior tests (mocked API)
# ---------------------------------------------------------------------------

_RESOLVE = "roboflow.cli.handlers.annotation._resolve_project_context"


class TestBatchList(unittest.TestCase):
    """annotation batch list"""

    @patch("roboflow.adapters.rfapi.list_batches")
    @patch(_RESOLVE, return_value=("key", "ws", "proj"))
    def test_text_output(self, _resolve, mock_api):
        mock_api.return_value = {"batches": [{"name": "b1", "id": "1", "status": "annotating", "images": 5}]}
        parser = _build_annotation_parser()
        args = parser.parse_args(["annotation", "batch", "list", "-p", "ws/proj"])

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            args.func(args)
        self.assertIn("b1", buf.getvalue())

    @patch("roboflow.adapters.rfapi.list_batches")
    @patch(_RESOLVE, return_value=("key", "ws", "proj"))
    def test_json_output(self, _resolve, mock_api):
        mock_api.return_value = {"batches": [{"name": "b1", "id": "1"}]}
        parser = _build_annotation_parser()
        args = parser.parse_args(["--json", "annotation", "batch", "list", "-p", "ws/proj"])

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            args.func(args)
        data = json.loads(buf.getvalue())
        self.assertIsInstance(data, list)
        self.assertEqual(data[0]["name"], "b1")

    @patch(_RESOLVE, return_value=None)
    def test_resolve_failure(self, _resolve):
        parser = _build_annotation_parser()
        args = parser.parse_args(["annotation", "batch", "list", "-p", "bad"])
        # Should return without crashing when resolve returns None
        args.func(args)


class TestBatchGet(unittest.TestCase):
    """annotation batch get"""

    @patch("roboflow.adapters.rfapi.get_batch")
    @patch(_RESOLVE, return_value=("key", "ws", "proj"))
    def test_text_output(self, _resolve, mock_api):
        mock_api.return_value = {"batch": {"name": "b1", "id": "1", "status": "annotating"}}
        parser = _build_annotation_parser()
        args = parser.parse_args(["annotation", "batch", "get", "1", "-p", "ws/proj"])

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            args.func(args)
        self.assertIn("b1", buf.getvalue())

    @patch("roboflow.adapters.rfapi.get_batch")
    @patch(_RESOLVE, return_value=("key", "ws", "proj"))
    def test_json_output(self, _resolve, mock_api):
        mock_api.return_value = {"batch": {"name": "b1", "id": "1"}}
        parser = _build_annotation_parser()
        args = parser.parse_args(["--json", "annotation", "batch", "get", "1", "-p", "ws/proj"])

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            args.func(args)
        data = json.loads(buf.getvalue())
        self.assertIn("batch", data)


class TestJobList(unittest.TestCase):
    """annotation job list"""

    @patch("roboflow.adapters.rfapi.list_annotation_jobs")
    @patch(_RESOLVE, return_value=("key", "ws", "proj"))
    def test_text_output(self, _resolve, mock_api):
        mock_api.return_value = {"jobs": [{"name": "j1", "id": "10", "status": "active", "assigned_to": "a@b.com"}]}
        parser = _build_annotation_parser()
        args = parser.parse_args(["annotation", "job", "list", "-p", "ws/proj"])

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            args.func(args)
        self.assertIn("j1", buf.getvalue())

    @patch("roboflow.adapters.rfapi.list_annotation_jobs")
    @patch(_RESOLVE, return_value=("key", "ws", "proj"))
    def test_json_output(self, _resolve, mock_api):
        mock_api.return_value = {"jobs": [{"name": "j1", "id": "10"}]}
        parser = _build_annotation_parser()
        args = parser.parse_args(["--json", "annotation", "job", "list", "-p", "ws/proj"])

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            args.func(args)
        data = json.loads(buf.getvalue())
        self.assertIsInstance(data, list)


class TestJobGet(unittest.TestCase):
    """annotation job get"""

    @patch("roboflow.adapters.rfapi.get_annotation_job")
    @patch(_RESOLVE, return_value=("key", "ws", "proj"))
    def test_text_output(self, _resolve, mock_api):
        mock_api.return_value = {"job": {"name": "j1", "id": "10", "status": "active"}}
        parser = _build_annotation_parser()
        args = parser.parse_args(["annotation", "job", "get", "10", "-p", "ws/proj"])

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            args.func(args)
        self.assertIn("j1", buf.getvalue())


class TestJobCreate(unittest.TestCase):
    """annotation job create"""

    @patch("roboflow.Roboflow")
    @patch(_RESOLVE, return_value=("key", "ws", "proj"))
    def test_text_output(self, _resolve, mock_rf_cls):
        mock_project = MagicMock()
        mock_project.create_annotation_job.return_value = {"id": "42", "name": "new-job"}
        mock_rf_cls.return_value.workspace.return_value.project.return_value = mock_project

        parser = _build_annotation_parser()
        args = parser.parse_args(
            [
                "annotation",
                "job",
                "create",
                "-p",
                "ws/proj",
                "--name",
                "new-job",
                "--batch",
                "b1",
                "--num-images",
                "5",
                "--labeler",
                "a@b.com",
                "--reviewer",
                "c@d.com",
            ]
        )

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            args.func(args)
        self.assertIn("new-job", buf.getvalue())
        mock_project.create_annotation_job.assert_called_once_with(
            name="new-job",
            batch_id="b1",
            num_images=5,
            labeler_email="a@b.com",
            reviewer_email="c@d.com",
        )

    @patch("roboflow.Roboflow")
    @patch(_RESOLVE, return_value=("key", "ws", "proj"))
    def test_json_output(self, _resolve, mock_rf_cls):
        mock_project = MagicMock()
        mock_project.create_annotation_job.return_value = {"id": "42", "name": "new-job"}
        mock_rf_cls.return_value.workspace.return_value.project.return_value = mock_project

        parser = _build_annotation_parser()
        args = parser.parse_args(
            [
                "--json",
                "annotation",
                "job",
                "create",
                "-p",
                "ws/proj",
                "--name",
                "new-job",
                "--batch",
                "b1",
                "--num-images",
                "5",
                "--labeler",
                "a@b.com",
                "--reviewer",
                "c@d.com",
            ]
        )

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            args.func(args)
        data = json.loads(buf.getvalue())
        self.assertEqual(data["id"], "42")

    def test_create_requires_all_flags(self):
        parser = _build_annotation_parser()
        # Missing --reviewer should fail
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "annotation",
                    "job",
                    "create",
                    "-p",
                    "proj",
                    "--name",
                    "j",
                    "--batch",
                    "b",
                    "--num-images",
                    "1",
                    "--labeler",
                    "a@b.com",
                ]
            )


if __name__ == "__main__":
    unittest.main()
