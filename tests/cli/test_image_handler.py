"""Unit tests for roboflow.cli.handlers.image."""

import argparse
import io
import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch


def _make_args(**overrides):
    defaults = {
        "json": False,
        "api_key": "test-key",
        "workspace": "test-ws",
        "quiet": False,
    }
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def _build_image_parser():
    """Build a minimal parser with just the image handler registered."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", "-j", action="store_true", default=False)
    parser.add_argument("--api-key", "-k", dest="api_key", default=None)
    parser.add_argument("--workspace", "-w", dest="workspace", default=None)
    parser.add_argument("--quiet", "-q", action="store_true", default=False)
    sub = parser.add_subparsers(title="commands", dest="command")

    from roboflow.cli.handlers.image import register

    register(sub)
    return parser


class TestImageParserRegistration(unittest.TestCase):
    """Verify the image handler registers its subcommands."""

    def test_image_subcommand_exists(self):
        parser = _build_image_parser()
        args = parser.parse_args(["image", "upload", "test.jpg", "-p", "my-proj"])
        self.assertEqual(args.path, "test.jpg")
        self.assertEqual(args.project, "my-proj")

    def test_image_upload_defaults(self):
        parser = _build_image_parser()
        args = parser.parse_args(["image", "upload", "test.jpg", "-p", "proj"])
        self.assertEqual(args.split, "train")
        self.assertEqual(args.concurrency, 10)
        self.assertEqual(args.retries, 0)
        self.assertFalse(args.is_prediction)

    def test_image_get_parser(self):
        parser = _build_image_parser()
        args = parser.parse_args(["image", "get", "img-123", "-p", "proj"])
        self.assertEqual(args.image_id, "img-123")
        self.assertEqual(args.project, "proj")

    def test_image_search_parser(self):
        parser = _build_image_parser()
        args = parser.parse_args(["image", "search", "tag:review", "-p", "proj", "--limit", "10"])
        self.assertEqual(args.query, "tag:review")
        self.assertEqual(args.limit, 10)

    def test_image_tag_parser(self):
        parser = _build_image_parser()
        args = parser.parse_args(["image", "tag", "img-1", "-p", "proj", "--add", "a,b", "--remove", "c"])
        self.assertEqual(args.image_id, "img-1")
        self.assertEqual(args.add_tags, "a,b")
        self.assertEqual(args.remove_tags, "c")

    def test_image_delete_parser(self):
        parser = _build_image_parser()
        args = parser.parse_args(["image", "delete", "id1,id2", "-p", "proj"])
        self.assertEqual(args.image_ids, "id1,id2")

    def test_image_annotate_parser(self):
        parser = _build_image_parser()
        args = parser.parse_args(
            ["image", "annotate", "img-1", "-p", "proj", "--annotation-file", "ann.txt"]
        )
        self.assertEqual(args.image_id, "img-1")
        self.assertEqual(args.annotation_file, "ann.txt")


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

    @patch("roboflow.cli.handlers.image.rfapi")
    def test_delete_images(self, mock_rfapi):
        from roboflow.cli.handlers.image import _handle_delete

        mock_rfapi.workspace_delete_images.return_value = {"deleted": 2, "skipped": 0}

        args = _make_args(json=True, image_ids="id1,id2", project="proj")

        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            _handle_delete(args)
        finally:
            sys.stdout = old

        mock_rfapi.workspace_delete_images.assert_called_once_with(
            api_key="test-key",
            workspace_url="test-ws",
            image_ids=["id1", "id2"],
        )
        result = json.loads(buf.getvalue())
        self.assertEqual(result["deleted"], 2)


class TestImageSearch(unittest.TestCase):
    """Test the search handler."""

    @patch("roboflow.cli.handlers.image.rfapi")
    def test_search(self, mock_rfapi):
        from roboflow.cli.handlers.image import _handle_search

        mock_rfapi.workspace_search.return_value = {"results": [], "total": 0}

        args = _make_args(json=True, query="tag:test", project="proj", limit=10, cursor=None)

        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            _handle_search(args)
        finally:
            sys.stdout = old

        mock_rfapi.workspace_search.assert_called_once()
        result = json.loads(buf.getvalue())
        self.assertEqual(result["total"], 0)


class TestImageAnnotate(unittest.TestCase):
    """Test the annotate handler."""

    @patch("roboflow.cli.handlers.image.rfapi")
    def test_annotate(self, mock_rfapi):
        from roboflow.cli.handlers.image import _handle_annotate

        mock_rfapi.save_annotation.return_value = {"success": True}

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

            mock_rfapi.save_annotation.assert_called_once()
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


if __name__ == "__main__":
    unittest.main()
