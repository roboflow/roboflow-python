"""Unit tests for roboflow.cli.handlers.image."""

import io
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import responses
from typer.testing import CliRunner

from roboflow.adapters.rfapi import ImageUploadError
from roboflow.cli import app
from roboflow.config import API_URL
from tests import PROJECT_NAME, ROBOFLOW_API_KEY, WORKSPACE_NAME, RoboflowTest

runner = CliRunner()


def _make_args(**overrides):
    defaults = {
        "json": False,
        "api_key": "test-key",
        "workspace": "test-ws",
        "quiet": False,
    }
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


class TestImageParserRegistration(unittest.TestCase):
    """Verify the image handler registers its subcommands."""

    def test_image_subcommand_exists(self):
        result = runner.invoke(app, ["image", "upload", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_image_upload_help(self):
        result = runner.invoke(app, ["image", "upload", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("project", result.output.lower())

    def test_image_get_help(self):
        result = runner.invoke(app, ["image", "get", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_image_search_help(self):
        result = runner.invoke(app, ["image", "search", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_image_tag_help(self):
        result = runner.invoke(app, ["image", "tag", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_image_delete_help(self):
        result = runner.invoke(app, ["image", "delete", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_image_annotate_help(self):
        result = runner.invoke(app, ["image", "annotate", "--help"])
        self.assertEqual(result.exit_code, 0)


class TestImageUploadSingle(unittest.TestCase):
    """Test the single-file upload path."""

    @patch("roboflow.Roboflow")
    def test_upload_single_file(self, mock_rf_cls):
        from roboflow.cli.handlers.image import _handle_upload

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"fake-image")
            tmp = f.name
        try:
            mock_project = MagicMock()
            mock_rf_cls.return_value.workspace.return_value.project.return_value = mock_project

            args = _make_args(
                path=tmp,
                project="proj",
                annotation=None,
                split="train",
                batch=None,
                tag=None,
                metadata=None,
                concurrency=10,
                retries=0,
                labelmap=None,
                is_prediction=False,
            )

            buf = io.StringIO()
            old = sys.stdout
            sys.stdout = buf
            try:
                _handle_upload(args)
            finally:
                sys.stdout = old

            mock_project.single_upload.assert_called_once()
            self.assertIn("Uploaded", buf.getvalue())
        finally:
            os.unlink(tmp)

    @patch("roboflow.Roboflow")
    def test_upload_single_json_mode(self, mock_rf_cls):
        from roboflow.cli.handlers.image import _handle_upload

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"fake-image")
            tmp = f.name
        try:
            mock_project = MagicMock()
            mock_rf_cls.return_value.workspace.return_value.project.return_value = mock_project

            args = _make_args(
                json=True,
                path=tmp,
                project="proj",
                annotation=None,
                split="train",
                batch=None,
                tag=None,
                metadata=None,
                concurrency=10,
                retries=0,
                labelmap=None,
                is_prediction=False,
            )

            buf = io.StringIO()
            old = sys.stdout
            sys.stdout = buf
            try:
                _handle_upload(args)
            finally:
                sys.stdout = old

            result = json.loads(buf.getvalue())
            self.assertEqual(result["status"], "uploaded")
        finally:
            os.unlink(tmp)


class TestImageUploadDirectory(unittest.TestCase):
    """Test the directory import path."""

    @patch("roboflow.Roboflow")
    def test_upload_directory(self, mock_rf_cls):
        from roboflow.cli.handlers.image import _handle_upload

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some fake images
            for name in ["a.jpg", "b.png", "c.txt"]:
                with open(os.path.join(tmpdir, name), "w") as f:
                    f.write("x")

            mock_ws = MagicMock()
            mock_ws.upload_dataset.return_value = {"uploaded": 1, "duplicates": 1, "failed": []}
            mock_rf_cls.return_value.workspace.return_value = mock_ws

            args = _make_args(
                json=True,
                path=tmpdir,
                project="proj",
                annotation=None,
                split="train",
                batch=None,
                tag=None,
                metadata=None,
                concurrency=5,
                retries=1,
                labelmap=None,
                is_prediction=False,
            )

            buf = io.StringIO()
            old = sys.stdout
            sys.stdout = buf
            try:
                _handle_upload(args)
            finally:
                sys.stdout = old

            mock_ws.upload_dataset.assert_called_once()
            result = json.loads(buf.getvalue())
            self.assertEqual(result["status"], "imported")
            self.assertEqual(result["count"], 2)  # .jpg and .png only

    @patch("roboflow.Roboflow")
    def test_upload_zip_file_routes_to_directory_handler(self, mock_rf_cls):
        from roboflow.cli.handlers.image import _handle_upload

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
            f.write(b"fake zip")
            zip_path = f.name

        try:
            mock_ws = MagicMock()
            mock_ws.upload_dataset.return_value = {"status": "completed", "task_id": "t1"}
            mock_project = MagicMock()
            mock_rf_cls.return_value.workspace.return_value = mock_ws
            mock_ws.project.return_value = mock_project

            args = _make_args(
                json=True,
                path=zip_path,
                project="proj",
                annotation=None,
                split="train",
                batch=None,
                tag=None,
                metadata=None,
                concurrency=10,
                retries=0,
                labelmap=None,
                is_prediction=False,
                no_wait=False,
            )

            buf = io.StringIO()
            old = sys.stdout
            sys.stdout = buf
            try:
                _handle_upload(args)
            finally:
                sys.stdout = old

            mock_ws.upload_dataset.assert_called_once()
            mock_project.single_upload.assert_not_called()
        finally:
            os.unlink(zip_path)

    @patch("roboflow.Roboflow")
    def test_no_wait_forwarded(self, mock_rf_cls):
        from roboflow.cli.handlers.image import _handle_upload

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_ws = MagicMock()
            mock_ws.upload_dataset.return_value = {"status": "pending", "task_id": "t9"}
            mock_rf_cls.return_value.workspace.return_value = mock_ws

            args = _make_args(
                json=True,
                path=tmpdir,
                project="proj",
                annotation=None,
                split="train",
                batch=None,
                tag=None,
                metadata=None,
                concurrency=10,
                retries=0,
                labelmap=None,
                is_prediction=False,
                zip_upload=True,
                no_wait=True,
            )

            buf = io.StringIO()
            old = sys.stdout
            sys.stdout = buf
            try:
                _handle_upload(args)
            finally:
                sys.stdout = old

            _, kwargs = mock_ws.upload_dataset.call_args
            self.assertEqual(kwargs.get("wait"), False)
            self.assertEqual(kwargs.get("use_zip_upload"), True)

    @patch("roboflow.Roboflow")
    def test_zip_flow_uses_server_result_in_output(self, mock_rf_cls):
        from roboflow.cli.handlers.image import _handle_upload

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_ws = MagicMock()
            mock_ws.upload_dataset.return_value = {"status": "completed", "task_id": "t1"}
            mock_rf_cls.return_value.workspace.return_value = mock_ws

            args = _make_args(
                json=True,
                path=tmpdir,
                project="proj",
                annotation=None,
                split="train",
                batch=None,
                tag="foo,bar",
                metadata=None,
                concurrency=10,
                retries=0,
                labelmap=None,
                is_prediction=False,
                zip_upload=True,
                no_wait=False,
            )

            buf = io.StringIO()
            old = sys.stdout
            sys.stdout = buf
            try:
                _handle_upload(args)
            finally:
                sys.stdout = old

            result = json.loads(buf.getvalue())
            self.assertEqual(result["task_id"], "t1")
            self.assertEqual(result["status"], "completed")

            _, kwargs = mock_ws.upload_dataset.call_args
            self.assertEqual(kwargs.get("tags"), ["foo", "bar"])
            self.assertEqual(kwargs.get("use_zip_upload"), True)

    @patch("roboflow.Roboflow")
    def test_zip_upload_flag_defaults_false(self, mock_rf_cls):
        from roboflow.cli.handlers.image import _handle_upload

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_ws = MagicMock()
            mock_ws.upload_dataset.return_value = {"uploaded": 0, "duplicates": 0, "failed": []}
            mock_rf_cls.return_value.workspace.return_value = mock_ws

            args = _make_args(
                json=True,
                path=tmpdir,
                project="proj",
                annotation=None,
                split="train",
                batch=None,
                tag=None,
                metadata=None,
                concurrency=10,
                retries=0,
                labelmap=None,
                is_prediction=False,
            )

            buf = io.StringIO()
            old = sys.stdout
            sys.stdout = buf
            try:
                _handle_upload(args)
            finally:
                sys.stdout = old

            _, kwargs = mock_ws.upload_dataset.call_args
            self.assertEqual(kwargs.get("use_zip_upload"), False)

    @patch("roboflow.cli.handlers.image._handle_upload")
    def test_annotation_overwrite_flag_three_states(self, mock_handle_upload):
        with tempfile.TemporaryDirectory() as tmpdir:
            for extra_argv, expected in [
                ([], None),
                (["--annotation-overwrite"], True),
                (["--no-annotation-overwrite"], False),
            ]:
                mock_handle_upload.reset_mock()
                result = runner.invoke(
                    app,
                    ["--workspace", "ws", "--api-key", "k", "image", "upload", tmpdir, "-p", "proj"] + extra_argv,
                )
                self.assertEqual(result.exit_code, 0)
                args = mock_handle_upload.call_args.args[0]
                self.assertEqual(args.annotation_overwrite, expected)

    @patch("roboflow.Roboflow")
    def test_annotation_overwrite_forwarded_to_upload_dataset(self, mock_rf_cls):
        from roboflow.cli.handlers.image import _handle_upload

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_ws = MagicMock()
            mock_ws.upload_dataset.return_value = {"status": "completed", "task_id": "t1"}
            mock_rf_cls.return_value.workspace.return_value = mock_ws

            args = _make_args(
                json=True,
                path=tmpdir,
                project="proj",
                annotation=None,
                split=None,
                batch=None,
                tag=None,
                metadata=None,
                concurrency=10,
                retries=0,
                labelmap=None,
                is_prediction=False,
                zip_upload=True,
                annotation_overwrite=True,
                no_wait=False,
            )

            buf = io.StringIO()
            old = sys.stdout
            sys.stdout = buf
            try:
                _handle_upload(args)
            finally:
                sys.stdout = old

            _, kwargs = mock_ws.upload_dataset.call_args
            self.assertEqual(kwargs.get("annotation_overwrite"), True)

    @patch("roboflow.Roboflow")
    def test_upload_directory_omits_default_split_when_not_explicit(self, mock_rf_cls):
        from roboflow.cli.handlers.image import _handle_upload

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_ws = MagicMock()
            mock_rf_cls.return_value.workspace.return_value = mock_ws

            args = _make_args(
                json=True,
                path=tmpdir,
                project="proj",
                annotation=None,
                split=None,
                batch=None,
                tag=None,
                metadata=None,
                concurrency=10,
                retries=0,
                labelmap=None,
                is_prediction=False,
            )

            buf = io.StringIO()
            old = sys.stdout
            sys.stdout = buf
            try:
                _handle_upload(args)
            finally:
                sys.stdout = old

            _, kwargs = mock_ws.upload_dataset.call_args
            self.assertIsNone(kwargs.get("split"))

    @patch("roboflow.Roboflow")
    def test_upload_directory_forwards_explicit_split(self, mock_rf_cls):
        from roboflow.cli.handlers.image import _handle_upload

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_ws = MagicMock()
            mock_rf_cls.return_value.workspace.return_value = mock_ws

            args = _make_args(
                json=True,
                path=tmpdir,
                project="proj",
                annotation=None,
                split="valid",
                batch=None,
                tag=None,
                metadata=None,
                concurrency=10,
                retries=0,
                labelmap=None,
                is_prediction=False,
            )

            buf = io.StringIO()
            old = sys.stdout
            sys.stdout = buf
            try:
                _handle_upload(args)
            finally:
                sys.stdout = old

            _, kwargs = mock_ws.upload_dataset.call_args
            self.assertEqual(kwargs.get("split"), "valid")


class TestImageDelete(unittest.TestCase):
    """Test the delete handler."""

    @patch("roboflow.adapters.rfapi.workspace_delete_images")
    def test_delete_images(self, mock_delete_images):
        from roboflow.cli.handlers.image import _handle_delete

        mock_delete_images.return_value = {"deleted": 2, "skipped": 0}

        args = _make_args(json=True, image_ids="id1,id2", project="proj")

        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            _handle_delete(args)
        finally:
            sys.stdout = old

        mock_delete_images.assert_called_once_with(
            api_key="test-key",
            workspace_url="test-ws",
            image_ids=["id1", "id2"],
        )
        result = json.loads(buf.getvalue())
        self.assertEqual(result["deleted"], 2)


class TestImageSearch(unittest.TestCase):
    """Test the search handler."""

    @patch("roboflow.adapters.rfapi.workspace_search")
    def test_search(self, mock_workspace_search):
        from roboflow.cli.handlers.image import _handle_search

        mock_workspace_search.return_value = {"results": [], "total": 0}

        args = _make_args(json=True, query="tag:test", project="proj", limit=10, cursor=None)

        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            _handle_search(args)
        finally:
            sys.stdout = old

        mock_workspace_search.assert_called_once()
        # -p must scope via a `project:<slug>` filter prepended to the query.
        called_query = mock_workspace_search.call_args.kwargs["query"]
        self.assertEqual(called_query, "project:proj tag:test")
        result = json.loads(buf.getvalue())
        self.assertEqual(result["total"], 0)

    @patch("roboflow.adapters.rfapi.workspace_search")
    def test_search_without_project_is_unscoped(self, mock_workspace_search):
        from roboflow.cli.handlers.image import _handle_search

        mock_workspace_search.return_value = {"results": [], "total": 0}
        args = _make_args(json=True, query="tag:test", project=None, limit=10, cursor=None)

        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            _handle_search(args)
        finally:
            sys.stdout = old

        called_query = mock_workspace_search.call_args.kwargs["query"]
        self.assertEqual(called_query, "tag:test")

    @patch("roboflow.cli.handlers.search._search")
    @patch("roboflow.cli.handlers.image._handle_search")
    def test_search_with_project_and_export_scopes_the_export(self, mock_handle_search, mock_search):
        # `-p ... --export` must export the project, not silently drop --export.
        result = runner.invoke(
            app,
            ["--workspace", "ws", "--api-key", "k", "image", "search", "tag:test", "-p", "proj", "--export"],
        )

        self.assertEqual(result.exit_code, 0)
        mock_handle_search.assert_not_called()
        mock_search.assert_called_once()
        export_args = mock_search.call_args.args[0]
        self.assertTrue(export_args.export)
        # Export scopes by the `dataset` (project slug) body param.
        self.assertEqual(export_args.dataset, "proj")

    @patch("roboflow.Roboflow")
    def test_search_export_forwards_cli_api_key_to_sdk(self, mock_roboflow):
        # The export path must honor an explicitly supplied --api-key, not only
        # saved/env credentials (CI/agent workflows pass the key directly).
        mock_roboflow.return_value = MagicMock()
        result = runner.invoke(
            app,
            ["--workspace", "ws", "--api-key", "MY_KEY", "image", "search", "tag:test", "-p", "proj", "--export"],
        )

        self.assertEqual(result.exit_code, 0)
        mock_roboflow.assert_called_once()
        self.assertEqual(mock_roboflow.call_args.kwargs.get("api_key"), "MY_KEY")

    @patch("roboflow.cli.handlers.search._search")
    @patch("roboflow.cli.handlers.image._handle_search")
    def test_search_with_project_no_export_uses_roboql_filter(self, mock_handle_search, mock_search):
        result = runner.invoke(
            app,
            ["--workspace", "ws", "--api-key", "k", "image", "search", "tag:test", "-p", "proj"],
        )

        self.assertEqual(result.exit_code, 0)
        mock_search.assert_not_called()
        mock_handle_search.assert_called_once()


class TestImageAnnotate(unittest.TestCase):
    """Test the annotate handler."""

    @patch("roboflow.adapters.rfapi.save_annotation")
    def test_annotate(self, mock_save_annotation):
        from roboflow.cli.handlers.image import _handle_annotate

        mock_save_annotation.return_value = {"success": True}

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            f.write("annotation data")
            ann_path = f.name

        try:
            args = _make_args(
                json=True,
                image_id="img-1",
                project="proj",
                annotation_file=ann_path,
                annotation_format=None,
                labelmap=None,
            )

            buf = io.StringIO()
            old = sys.stdout
            sys.stdout = buf
            try:
                _handle_annotate(args)
            finally:
                sys.stdout = old

            mock_save_annotation.assert_called_once()
            result = json.loads(buf.getvalue())
            self.assertEqual(result["status"], "saved")
        finally:
            os.unlink(ann_path)


class TestUploadPathNotFound(unittest.TestCase):
    """Test error when path doesn't exist."""

    def test_nonexistent_path(self):
        from roboflow.cli.handlers.image import _handle_upload

        args = _make_args(
            path="/nonexistent/path.jpg",
            project="proj",
            annotation=None,
            split="train",
            batch=None,
            tag=None,
            metadata=None,
            concurrency=10,
            retries=0,
            labelmap=None,
            is_prediction=False,
        )

        with self.assertRaises(SystemExit):
            _handle_upload(args)


class TestImageMetadataRegistration(unittest.TestCase):
    """Verify the metadata command and tag alias register correctly."""

    def test_image_metadata_help(self):
        result = runner.invoke(app, ["image", "metadata", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("tags", result.output.lower())
        self.assertIn("metadata", result.output.lower())

    def test_tag_is_alias(self):
        result = runner.invoke(app, ["image", "tag", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("tags", result.output.lower())
        self.assertNotIn("project", result.output.lower())


class TestImageMetadataSingle(unittest.TestCase):
    """Test the single-image metadata path."""

    @patch("roboflow.cli._resolver.resolve_ws_and_key", return_value=("test-ws", "test-key"))
    @patch("roboflow.adapters.rfapi.update_image_metadata", return_value={"success": True})
    def test_metadata_single(self, mock_update, mock_resolve):
        from roboflow.cli.handlers.image import _handle_metadata

        args = _make_args(
            image_ids="img-1",
            metadata='{"camera": "cam1"}',
            remove_metadata=None,
            add_tags="review",
            remove_tags=None,
            poll=False,
            timeout=1800,
        )
        _handle_metadata(args)
        mock_update.assert_called_once_with(
            api_key="test-key",
            workspace_url="test-ws",
            image_id="img-1",
            metadata={"camera": "cam1"},
            remove_metadata=None,
            add_tags=["review"],
            remove_tags=None,
        )

    def test_metadata_invalid_json(self):
        from roboflow.cli.handlers.image import _handle_metadata

        args = _make_args(
            image_ids="img-1",
            metadata="not-json",
            remove_metadata=None,
            add_tags=None,
            remove_tags=None,
            poll=False,
            timeout=1800,
        )
        buf = io.StringIO()
        old = sys.stderr
        sys.stderr = buf
        try:
            with self.assertRaises(SystemExit):
                _handle_metadata(args)
        finally:
            sys.stderr = old
        self.assertIn("Invalid metadata JSON", buf.getvalue())

    def test_metadata_nothing_to_do(self):
        from roboflow.cli.handlers.image import _handle_metadata

        args = _make_args(
            image_ids="img-1",
            metadata=None,
            remove_metadata=None,
            add_tags=None,
            remove_tags=None,
            poll=False,
            timeout=1800,
        )
        buf = io.StringIO()
        old = sys.stderr
        sys.stderr = buf
        try:
            with self.assertRaises(SystemExit):
                _handle_metadata(args)
        finally:
            sys.stderr = old
        self.assertIn("Nothing to update", buf.getvalue())


class TestImageMetadataBatch(unittest.TestCase):
    """Test the batch (multi-image) metadata path."""

    @patch("roboflow.cli._resolver.resolve_ws_and_key", return_value=("test-ws", "test-key"))
    @patch(
        "roboflow.adapters.rfapi.batch_update_image_metadata",
        return_value={"taskId": "t1", "url": "https://api.roboflow.com/test-ws/asynctasks/t1"},
    )
    def test_metadata_batch_no_poll(self, mock_batch, mock_resolve):
        from roboflow.cli.handlers.image import _handle_metadata

        args = _make_args(
            image_ids="img-1,img-2,img-3",
            metadata=None,
            remove_metadata=None,
            add_tags="review",
            remove_tags=None,
            poll=False,
            timeout=1800,
            json=True,
        )
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            _handle_metadata(args)
        finally:
            sys.stdout = old
        data = json.loads(buf.getvalue())
        self.assertEqual(data["taskId"], "t1")
        self.assertEqual(data["imageCount"], 3)
        mock_batch.assert_called_once()
        updates = mock_batch.call_args[1]["updates"]
        self.assertEqual(len(updates), 3)
        self.assertEqual(updates[0]["imageId"], "img-1")
        self.assertEqual(updates[0]["addTags"], ["review"])

    def test_metadata_batch_over_limit(self):
        from roboflow.cli.handlers.image import _handle_metadata

        ids = ",".join([f"img-{i}" for i in range(1001)])
        args = _make_args(
            image_ids=ids,
            metadata=None,
            remove_metadata=None,
            add_tags="review",
            remove_tags=None,
            poll=False,
            timeout=1800,
        )
        with patch("roboflow.cli._resolver.resolve_ws_and_key", return_value=("test-ws", "test-key")):
            buf = io.StringIO()
            old = sys.stderr
            sys.stderr = buf
            try:
                with self.assertRaises(SystemExit):
                    _handle_metadata(args)
            finally:
                sys.stderr = old
            self.assertIn("Too many images", buf.getvalue())


if __name__ == "__main__":
    unittest.main()


class TestImageUploadDirectoryFailures(RoboflowTest):
    """Failed directory uploads exit non-zero with a failure count (#545)."""

    def setUp(self):
        super().setUp()
        rf_patch = patch("roboflow.Roboflow")
        rf_patch.start().return_value.workspace.return_value = self.workspace
        self.addCleanup(rf_patch.stop)
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir)
        for name in ["a.jpg", "b.jpg", "c.jpg"]:
            with open(os.path.join(self.tmpdir, name), "wb") as f:
                f.write(b"x")

    def _invoke(self):
        return runner.invoke(
            app,
            ["--json", "--api-key", ROBOFLOW_API_KEY, "image", "upload", self.tmpdir, "-p", PROJECT_NAME, "-c", "1"],
        )

    def test_every_image_fails(self):
        with patch("roboflow.core.project.Project.upload_image", side_effect=ImageUploadError("Invalid image")):
            result = self._invoke()

        self.assertEqual(result.exit_code, 1)
        error = json.loads(result.stderr.strip().splitlines()[-1])["error"]
        self.assertTrue(error["message"].startswith("3 of 3 images failed to upload"))
        self.assertEqual(error["failed_count"], 3)
        self.assertEqual(error["uploaded"], 0)
        self.assertEqual({f["error"] for f in error["failed"]}, {"Invalid image"})
        self.assertIn("hint", error)
        self.assertNotIn('"status"', result.stdout)

    def test_partial_failure_reports_count(self):
        def upload_image(image_path, **kwargs):
            if image_path.endswith("b.jpg"):
                raise ImageUploadError("Invalid image")
            return {"id": image_path, "success": True}, 0.1, 0

        with patch("roboflow.core.project.Project.upload_image", side_effect=upload_image):
            result = self._invoke()

        self.assertEqual(result.exit_code, 1)
        error = json.loads(result.stderr.strip().splitlines()[-1])["error"]
        self.assertEqual(error["failed_count"], 1)
        self.assertEqual(error["uploaded"], 2)
        self.assertTrue(error["failed"][0]["path"].endswith("b.jpg"))

    def test_all_uploaded_exits_zero(self):
        image = ({"id": "img", "success": True}, 0.1, 0)
        with patch("roboflow.core.project.Project.upload_image", return_value=image):
            result = self._invoke()

        self.assertEqual(result.exit_code, 0, result.stderr)
        # SDK progress lines precede the indented JSON document on stdout
        data = json.loads(result.stdout[result.stdout.index("\n{") + 1 :])
        self.assertEqual(data["status"], "imported")
        self.assertEqual(data["count"], 3)
        self.assertEqual(data["uploaded"], 3)

    def test_zip_upload_failed_status_exits_nonzero(self):
        zip_path = os.path.join(self.tmpdir, "dataset.zip")
        with open(zip_path, "wb") as f:
            f.write(b"PK")
        responses.add(
            responses.POST,
            f"{API_URL}/{WORKSPACE_NAME}/{PROJECT_NAME}/upload/zip",
            json={"taskId": "t1", "signedUrl": "https://storage.example.com/zip"},
        )
        responses.add(responses.PUT, "https://storage.example.com/zip")
        responses.add(
            responses.GET,
            f"{API_URL}/{WORKSPACE_NAME}/upload/zip/t1",
            json={"status": "failed", "error": "No images found in zip"},
        )

        result = runner.invoke(
            app, ["--json", "--api-key", ROBOFLOW_API_KEY, "image", "upload", zip_path, "-p", PROJECT_NAME]
        )

        self.assertEqual(result.exit_code, 1, result.stdout)
        error = json.loads(result.stderr.strip().splitlines()[-1])["error"]
        self.assertIn("No images found in zip", error["message"])
