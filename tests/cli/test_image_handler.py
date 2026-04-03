"""Unit tests for roboflow.cli.handlers.image."""

import io
import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from roboflow.cli import app

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
        result = json.loads(buf.getvalue())
        self.assertEqual(result["total"], 0)


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


class TestImageTagValidation(unittest.TestCase):
    """Test that tag command validates --add/--remove presence."""

    def test_tag_no_add_or_remove(self):
        from roboflow.cli.handlers.image import _handle_tag

        args = _make_args(
            image_id="img-1",
            project="proj",
            add_tags=None,
            remove_tags=None,
        )

        buf = io.StringIO()
        old = sys.stderr
        sys.stderr = buf
        try:
            with self.assertRaises(SystemExit):
                _handle_tag(args)
        finally:
            sys.stderr = old

        self.assertIn("Nothing to do", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
