"""Unit tests for roboflow.cli.handlers.annotation."""

import io
import json
import sys
import types
import unittest
from unittest.mock import patch

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

    def test_full_annotation_administration_surface(self):
        commands = {
            "batch": ["admin-list", "admin-get", "images", "create", "merge", "delete"],
            "job": [
                "admin-list",
                "admin-get",
                "admin-create",
                "images",
                "reassign-images",
                "add-images",
                "update",
                "submit-review",
                "return-edits",
                "review-image",
                "review-images",
                "accept",
                "move-to-unassigned",
                "delete-annotations",
            ],
        }
        for group, names in commands.items():
            for name in names:
                with self.subTest(command=f"{group} {name}"):
                    result = runner.invoke(app, ["annotation", group, name, "--help"])
                    self.assertEqual(result.exit_code, 0, result.output)


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

    @patch("roboflow.adapters.rfapi.create_annotation_job_from_batch")
    @patch(_RESOLVE, return_value=("key", "ws", "proj"))
    def test_text_output(self, _resolve, mock_api):
        mock_api.return_value = {"id": "42", "name": "new-job"}

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
        mock_api.assert_called_once_with(
            "key",
            "ws",
            "proj",
            batch_id="b1",
            labeler_email="a@b.com",
            reviewer_email="c@d.com",
            name="new-job",
            num_images=5,
            instructions=None,
        )

    @patch("roboflow.adapters.rfapi.create_annotation_job_from_batch")
    @patch(_RESOLVE, return_value=("key", "ws", "proj"))
    def test_json_output(self, _resolve, mock_api):
        mock_api.return_value = {"id": "42", "name": "new-job"}

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


class TestAnnotationAdministrationCommands(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.list_annotation_jobs_admin")
    @patch(_RESOLVE, return_value=("key", "ws", "proj"))
    def test_admin_list_preserves_pagination_response(self, _resolve, mock_api):
        mock_api.return_value = {"jobs": [{"id": "job-1"}], "continuationToken": "next"}
        result = runner.invoke(
            app,
            [
                "--json",
                "annotation",
                "job",
                "admin-list",
                "-p",
                "ws/proj",
                "--limit",
                "10",
                "--after",
                "cursor",
                "--show-empty",
            ],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(json.loads(result.output)["continuationToken"], "next")
        mock_api.assert_called_once_with("key", "ws", "proj", limit=10, after="cursor", show_empty=True)

    @patch("roboflow.adapters.rfapi.get_annotation_job_admin")
    @patch(_RESOLVE, return_value=("key", "ws", "proj"))
    def test_admin_get_uses_administration_adapter(self, _resolve, mock_api):
        mock_api.return_value = {"id": "job-1"}
        result = runner.invoke(app, ["--json", "annotation", "job", "admin-get", "job-1", "-p", "ws/proj"])
        self.assertEqual(result.exit_code, 0, result.output)
        mock_api.assert_called_once_with("key", "ws", "proj", "job-1")

    @patch("roboflow.adapters.rfapi.create_annotation_job_admin")
    @patch(_RESOLVE, return_value=("key", "ws", "proj"))
    def test_admin_create_uses_administration_adapter(self, _resolve, mock_api):
        mock_api.return_value = {"id": "job-1"}
        result = runner.invoke(
            app,
            [
                "--json",
                "annotation",
                "job",
                "admin-create",
                "-p",
                "ws/proj",
                "--batch",
                "batch-1",
                "--labeler",
                "labeler@example.com",
                "--reviewer",
                "reviewer@example.com",
            ],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        mock_api.assert_called_once_with(
            "key",
            "ws",
            "proj",
            batch_id="batch-1",
            labeler_email="labeler@example.com",
            reviewer_email="reviewer@example.com",
            name=None,
            num_images=None,
            instructions=None,
        )

    @patch("roboflow.adapters.rfapi.create_annotation_batch")
    @patch(_RESOLVE, return_value=("key", "ws", "proj"))
    def test_batch_create_accepts_repeated_image_ids(self, _resolve, mock_api):
        mock_api.return_value = {"id": "batch-2"}
        result = runner.invoke(
            app,
            [
                "--json",
                "annotation",
                "batch",
                "create",
                "-p",
                "ws/proj",
                "--source-batch-id",
                "batch-1",
                "--image-id",
                "image-1",
                "--image-id",
                "image-2",
                "--name",
                "Round two",
            ],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        mock_api.assert_called_once_with(
            "key",
            "ws",
            "proj",
            source_batch_id="batch-1",
            image_ids=["image-1", "image-2"],
            name="Round two",
        )

    @patch("roboflow.adapters.rfapi.merge_annotation_batches")
    @patch(_RESOLVE, return_value=("key", "ws", "proj"))
    def test_batch_merge_requires_yes_non_interactively(self, _resolve, mock_api):
        command = [
            "annotation",
            "batch",
            "merge",
            "-p",
            "ws/proj",
            "--source-batch-id",
            "source",
            "--target-batch-id",
            "target",
        ]
        result = runner.invoke(app, command)
        self.assertEqual(result.exit_code, 1)
        mock_api.assert_not_called()

        mock_api.return_value = {"success": True}
        result = runner.invoke(app, [*command, "--yes"])
        self.assertEqual(result.exit_code, 0, result.output)
        mock_api.assert_called_once_with("key", "ws", "proj", source_batch_ids=["source"], target_batch_id="target")

    @patch("roboflow.adapters.rfapi.add_images_to_annotation_job")
    @patch(_RESOLVE, return_value=("key", "ws", "proj"))
    def test_job_add_images_requires_yes_non_interactively(self, _resolve, mock_api):
        command = [
            "annotation",
            "job",
            "add-images",
            "job-1",
            "-p",
            "ws/proj",
            "--image-id",
            "image-1",
        ]
        result = runner.invoke(app, command)
        self.assertEqual(result.exit_code, 1)
        mock_api.assert_not_called()

        mock_api.return_value = {"movedImageCount": 1}
        result = runner.invoke(app, [*command, "--yes"])
        self.assertEqual(result.exit_code, 0, result.output)
        mock_api.assert_called_once_with("key", "ws", "proj", "job-1", image_ids=["image-1"])

    @patch("roboflow.adapters.rfapi.reassign_annotation_job_images")
    @patch(_RESOLVE, return_value=("key", "ws", "proj"))
    def test_job_reassign_images_requires_yes_non_interactively(self, _resolve, mock_api):
        command = [
            "annotation",
            "job",
            "reassign-images",
            "-p",
            "ws/proj",
            "--image-id",
            "image-1",
            "--labeler",
            "labeler@example.com",
        ]
        result = runner.invoke(app, command)
        self.assertEqual(result.exit_code, 1)
        mock_api.assert_not_called()

        mock_api.return_value = {"jobId": "job-1"}
        result = runner.invoke(app, [*command, "--yes"])
        self.assertEqual(result.exit_code, 0, result.output)
        mock_api.assert_called_once_with(
            "key",
            "ws",
            "proj",
            image_ids=["image-1"],
            labeler_email="labeler@example.com",
            reviewer_email=None,
            instructions=None,
            name=None,
        )

    @patch("roboflow.adapters.rfapi.accept_annotation_job_images")
    @patch(_RESOLVE, return_value=("key", "ws", "proj"))
    def test_job_accept_maps_lists_and_split_counts(self, _resolve, mock_api):
        mock_api.return_value = {"success": True, "numImagesAdded": 1}
        result = runner.invoke(
            app,
            [
                "--json",
                "annotation",
                "job",
                "accept",
                "job-1",
                "-p",
                "ws/proj",
                "--split-method",
                "split",
                "--status",
                "approved",
                "--status",
                "annotated",
                "--train-count",
                "1",
                "--valid-count",
                "0",
                "--test-count",
                "0",
                "--image-id",
                "image-1",
                "--yes",
            ],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        mock_api.assert_called_once_with(
            "key",
            "ws",
            "proj",
            "job-1",
            split_method="split",
            statuses_to_include=["approved", "annotated"],
            train_count=1,
            valid_count=0,
            test_count=0,
            image_ids=["image-1"],
        )

    @patch("roboflow.adapters.rfapi.update_annotation_job", side_effect=ValueError("Provide exactly one field"))
    @patch(_RESOLVE, return_value=("key", "ws", "proj"))
    def test_job_update_surfaces_validation_error(self, _resolve, _mock_api):
        result = runner.invoke(app, ["annotation", "job", "update", "job-1", "-p", "ws/proj"])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("exactly one", result.output)

    @patch("roboflow.adapters.rfapi.delete_annotation_job_annotations")
    @patch(_RESOLVE, return_value=("key", "ws", "proj"))
    def test_delete_annotations_executes_with_yes(self, _resolve, mock_api):
        mock_api.return_value = {"success": True}
        result = runner.invoke(
            app,
            ["annotation", "job", "delete-annotations", "job-1", "-p", "ws/proj", "--yes"],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        mock_api.assert_called_once_with("key", "ws", "proj", "job-1")


if __name__ == "__main__":
    unittest.main()
