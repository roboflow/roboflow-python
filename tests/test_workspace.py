"""Unit tests for module-level helpers in roboflow.core.workspace."""

import os
import tempfile
import unittest
import zipfile
from unittest.mock import patch

from roboflow.core.workspace import Workspace, _zip_directory


def _make_workspace():
    return Workspace(
        {"workspace": {"name": "Test", "projects": [], "url": "test-ws"}},
        api_key="test-key",
        default_workspace="test-ws",
        model_format="yolov8",
    )


class TestZipDirectory(unittest.TestCase):
    def test_filters_hidden_and_junk_entries(self):
        with tempfile.TemporaryDirectory() as src:
            # Real content
            with open(os.path.join(src, "sample.jpg"), "wb") as fh:
                fh.write(b"jpg bytes")
            # Hidden / junk files at the top level
            with open(os.path.join(src, ".DS_Store"), "wb") as fh:
                fh.write(b"x")
            with open(os.path.join(src, "Thumbs.db"), "wb") as fh:
                fh.write(b"x")
            # macOS junk directory
            mac_dir = os.path.join(src, "__MACOSX")
            os.mkdir(mac_dir)
            with open(os.path.join(mac_dir, "whatever.txt"), "wb") as fh:
                fh.write(b"x")
            # Hidden directory
            hidden_dir = os.path.join(src, ".hidden")
            os.mkdir(hidden_dir)
            with open(os.path.join(hidden_dir, "inside.txt"), "wb") as fh:
                fh.write(b"x")

            zip_path = _zip_directory(src)
            try:
                with zipfile.ZipFile(zip_path) as zf:
                    names = set(zf.namelist())
                self.assertEqual(names, {"sample.jpg"})
            finally:
                os.unlink(zip_path)


class TestWorkspaceAsyncTasks(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.fork_project")
    def test_fork_project_uses_workspace_destination(self, mock_fork):
        workspace = _make_workspace()
        mock_fork.return_value = {"taskId": "task-1", "url": "poll-url"}

        result = workspace.fork_project(url="source-ws/source-project")

        self.assertEqual(result, {"taskId": "task-1", "url": "poll-url"})
        mock_fork.assert_called_once_with(
            "test-key",
            "test-ws",
            url="source-ws/source-project",
            source_project_slug=None,
        )

    @patch("roboflow.adapters.rfapi.fork_project")
    def test_fork_project_accepts_explicit_source_slug(self, mock_fork):
        workspace = _make_workspace()
        mock_fork.return_value = {"taskId": "task-1", "url": "poll-url"}

        workspace.fork_project(source_project_slug="source-project")

        mock_fork.assert_called_once_with(
            "test-key",
            "test-ws",
            url=None,
            source_project_slug="source-project",
        )

    @patch("roboflow.adapters.rfapi.get_async_task")
    def test_get_async_task_uses_workspace_destination(self, mock_get):
        workspace = _make_workspace()
        mock_get.return_value = {"taskId": "task-1", "status": "running"}

        result = workspace.get_async_task("task-1")

        self.assertEqual(result, {"taskId": "task-1", "status": "running"})
        mock_get.assert_called_once_with("test-key", "test-ws", "task-1")


class TestWorkspaceImageMetadata(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.update_image_metadata")
    def test_update_image_metadata_delegates(self, mock_update):
        workspace = _make_workspace()
        mock_update.return_value = {"success": True}

        result = workspace.update_image_metadata(
            "img-1",
            metadata={"quality": 95},
            remove_metadata=["old"],
            add_tags=["reviewed"],
            remove_tags=["pending"],
        )

        self.assertEqual(result, {"success": True})
        mock_update.assert_called_once_with(
            api_key="test-key",
            workspace_url="test-ws",
            image_id="img-1",
            metadata={"quality": 95},
            remove_metadata=["old"],
            add_tags=["reviewed"],
            remove_tags=["pending"],
        )

    @patch("roboflow.adapters.rfapi.update_image_metadata")
    def test_update_image_metadata_propagates_error(self, mock_update):
        from roboflow.adapters.rfapi import RoboflowError

        workspace = _make_workspace()
        mock_update.side_effect = RoboflowError('{"error": {"message": "Invalid tag"}}')

        with self.assertRaises(RoboflowError):
            workspace.update_image_metadata("img-1", add_tags=["bad tag"])

    @patch("roboflow.adapters.rfapi.batch_update_image_metadata")
    def test_batch_update_returns_enqueue_response_without_wait(self, mock_batch):
        workspace = _make_workspace()
        mock_batch.return_value = {"taskId": "task-9", "url": "https://api.test/poll"}
        updates = [{"imageId": "img-1", "addTags": ["t"]}]

        result = workspace.batch_update_image_metadata(updates)

        self.assertEqual(result, {"taskId": "task-9", "url": "https://api.test/poll"})
        mock_batch.assert_called_once_with(
            api_key="test-key",
            workspace_url="test-ws",
            updates=updates,
        )

    @patch("roboflow.core.async_tasks.time.sleep", lambda *_: None)
    @patch("roboflow.adapters.rfapi.get_async_task_at")
    @patch("roboflow.adapters.rfapi.batch_update_image_metadata")
    def test_batch_update_wait_polls_until_terminal(self, mock_batch, mock_poll):
        workspace = _make_workspace()
        mock_batch.return_value = {"taskId": "task-9", "url": "https://api.test/poll"}
        final = {
            "taskId": "task-9",
            "status": "completed",
            "result": {"totalProcessed": 2, "succeeded": 2, "failed": 0, "failedItems": []},
        }
        mock_poll.side_effect = [
            {"taskId": "task-9", "status": "running"},
            final,
        ]

        result = workspace.batch_update_image_metadata([{"imageId": "img-1", "addTags": ["t"]}], wait=True)

        self.assertEqual(result, final)
        self.assertEqual(mock_poll.call_count, 2)
        mock_poll.assert_called_with("test-key", "https://api.test/poll")

    @patch("roboflow.core.async_tasks.time.sleep", lambda *_: None)
    @patch("roboflow.adapters.rfapi.get_async_task")
    @patch("roboflow.adapters.rfapi.batch_update_image_metadata")
    def test_batch_update_wait_falls_back_to_task_id_poll(self, mock_batch, mock_get):
        workspace = _make_workspace()
        mock_batch.return_value = {"taskId": "task-9"}  # no polling url in response
        mock_get.return_value = {"taskId": "task-9", "status": "completed", "result": {}}

        result = workspace.batch_update_image_metadata([{"imageId": "img-1", "addTags": ["t"]}], wait=True)

        self.assertEqual(result["status"], "completed")
        mock_get.assert_called_once_with("test-key", "test-ws", "task-9")


if __name__ == "__main__":
    unittest.main()
