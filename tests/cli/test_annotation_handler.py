"""Unit tests for roboflow.cli.handlers.annotation."""

import io
import json
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from roboflow.cli import app

runner = CliRunner()


class TestAnnotationParserRegistration(unittest.TestCase):
    """Verify the annotation handler registers its subcommands."""

    def test_annotation_subcommand_exists(self):
        result = runner.invoke(app, ["annotation", "batch", "list", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_annotation_batch_get(self):
        result = runner.invoke(app, ["annotation", "batch", "get", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_annotation_job_list(self):
        result = runner.invoke(app, ["annotation", "job", "list", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_annotation_job_get(self):
        result = runner.invoke(app, ["annotation", "job", "get", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_annotation_job_create(self):
        result = runner.invoke(app, ["annotation", "job", "create", "--help"])
        self.assertEqual(result.exit_code, 0)


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
        result = runner.invoke(app, ["annotation", "batch", "list", "-p", "ws/proj"])
        self.assertIn("b1", result.output)

    @patch("roboflow.adapters.rfapi.list_batches")
    @patch(_RESOLVE, return_value=("key", "ws", "proj"))
    def test_json_output(self, _resolve, mock_api):
        mock_api.return_value = {"batches": [{"name": "b1", "id": "1"}]}
        result = runner.invoke(app, ["--json", "annotation", "batch", "list", "-p", "ws/proj"])
        data = json.loads(result.output)
        self.assertIsInstance(data, list)
        self.assertEqual(data[0]["name"], "b1")

    @patch(_RESOLVE, return_value=None)
    def test_resolve_failure(self, _resolve):
        runner.invoke(app, ["annotation", "batch", "list", "-p", "bad"])
        # Should not crash when resolve returns None


class TestBatchGet(unittest.TestCase):
    """annotation batch get"""

    @patch("roboflow.adapters.rfapi.get_batch")
    @patch(_RESOLVE, return_value=("key", "ws", "proj"))
    def test_text_output(self, _resolve, mock_api):
        mock_api.return_value = {"batch": {"name": "b1", "id": "1", "status": "annotating"}}
        result = runner.invoke(app, ["annotation", "batch", "get", "1", "-p", "ws/proj"])
        self.assertIn("b1", result.output)

    @patch("roboflow.adapters.rfapi.get_batch")
    @patch(_RESOLVE, return_value=("key", "ws", "proj"))
    def test_json_output(self, _resolve, mock_api):
        mock_api.return_value = {"batch": {"name": "b1", "id": "1"}}
        result = runner.invoke(app, ["--json", "annotation", "batch", "get", "1", "-p", "ws/proj"])
        data = json.loads(result.output)
        self.assertIn("batch", data)


class TestJobList(unittest.TestCase):
    """annotation job list"""

    @patch("roboflow.adapters.rfapi.list_annotation_jobs")
    @patch(_RESOLVE, return_value=("key", "ws", "proj"))
    def test_text_output(self, _resolve, mock_api):
        mock_api.return_value = {"jobs": [{"name": "j1", "id": "10", "status": "active", "assigned_to": "a@b.com"}]}
        result = runner.invoke(app, ["annotation", "job", "list", "-p", "ws/proj"])
        self.assertIn("j1", result.output)

    @patch("roboflow.adapters.rfapi.list_annotation_jobs")
    @patch(_RESOLVE, return_value=("key", "ws", "proj"))
    def test_json_output(self, _resolve, mock_api):
        mock_api.return_value = {"jobs": [{"name": "j1", "id": "10"}]}
        result = runner.invoke(app, ["--json", "annotation", "job", "list", "-p", "ws/proj"])
        data = json.loads(result.output)
        self.assertIsInstance(data, list)


class TestJobGet(unittest.TestCase):
    """annotation job get"""

    @patch("roboflow.adapters.rfapi.get_annotation_job")
    @patch(_RESOLVE, return_value=("key", "ws", "proj"))
    def test_text_output(self, _resolve, mock_api):
        mock_api.return_value = {"job": {"name": "j1", "id": "10", "status": "active"}}
        result = runner.invoke(app, ["annotation", "job", "get", "10", "-p", "ws/proj"])
        self.assertIn("j1", result.output)


class TestJobCreate(unittest.TestCase):
    """annotation job create"""

    @patch("roboflow.Roboflow")
    @patch(_RESOLVE, return_value=("key", "ws", "proj"))
    def test_text_output(self, _resolve, mock_rf_cls):
        mock_project = MagicMock()
        mock_project.create_annotation_job.return_value = {"id": "42", "name": "new-job"}
        mock_rf_cls.return_value.workspace.return_value.project.return_value = mock_project

        result = runner.invoke(
            app,
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
            ],
        )
        self.assertIn("new-job", result.output)
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

        result = runner.invoke(
            app,
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
            ],
        )
        data = json.loads(result.output)
        self.assertEqual(data["id"], "42")

    def test_create_requires_all_flags(self):
        # Missing --reviewer should fail
        result = runner.invoke(
            app,
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
            ],
        )
        self.assertNotEqual(result.exit_code, 0)


if __name__ == "__main__":
    unittest.main()
