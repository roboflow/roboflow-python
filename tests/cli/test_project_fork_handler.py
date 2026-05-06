"""Tests for the `roboflow project fork` CLI handler."""

import json
import unittest
from argparse import Namespace
from unittest.mock import patch

from typer.testing import CliRunner

from roboflow.cli import app

runner = CliRunner()


def _make_args(**kwargs):
    """Create a Namespace with CLI defaults and fork-command defaults."""
    defaults = {
        "json": False,
        "workspace": "test-ws",
        "api_key": "test-key",
        "quiet": False,
        "no_wait": False,
        "timeout": 1800,
    }
    defaults.update(kwargs)
    return Namespace(**defaults)


class TestProjectForkRegistration(unittest.TestCase):
    def test_fork_help_exists(self) -> None:
        result = runner.invoke(app, ["project", "fork", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Universe", result.output)
        self.assertIn("no", result.output.lower())
        self.assertIn("wait", result.output.lower())
        self.assertIn("timeout", result.output.lower())


class TestForkProjectNoWait(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.fork_project")
    @patch("roboflow.config.load_roboflow_api_key", return_value="test-key")
    def test_url_form_no_wait_text(self, _mock_key, mock_fork):
        from roboflow.cli.handlers.project import _fork_project

        mock_fork.return_value = {
            "taskId": "task-123",
            "url": "https://api.roboflow.com/test-ws/asynctasks/task-123",
        }
        args = _make_args(
            source="https://universe.roboflow.com/ws/proj",
            no_wait=True,
        )
        with patch("builtins.print") as mock_print:
            _fork_project(args)

        mock_fork.assert_called_once_with(
            "test-key",
            "test-ws",
            url="https://universe.roboflow.com/ws/proj",
        )
        printed = mock_print.call_args[0][0]
        self.assertIn("task-123", printed)

    @patch("roboflow.adapters.rfapi.fork_project")
    @patch("roboflow.config.load_roboflow_api_key", return_value="test-key")
    def test_shorthand_no_wait_json(self, _mock_key, mock_fork):
        from roboflow.cli.handlers.project import _fork_project

        mock_fork.return_value = {"taskId": "task-456", "url": "poll-url"}
        args = _make_args(json=True, source="ws/proj", no_wait=True)
        with patch("builtins.print") as mock_print:
            _fork_project(args)

        mock_fork.assert_called_once_with(
            "test-key",
            "test-ws",
            url="ws/proj",
        )
        out = json.loads(mock_print.call_args[0][0])
        # Server response is passed through verbatim.
        self.assertEqual(out, {"taskId": "task-456", "url": "poll-url"})


class TestForkProjectWait(unittest.TestCase):
    @patch("roboflow.core.async_tasks.time.sleep", lambda *_a, **_k: None)
    @patch("roboflow.adapters.rfapi.get_async_task")
    @patch("roboflow.adapters.rfapi.fork_project")
    @patch("roboflow.config.load_roboflow_api_key", return_value="test-key")
    def test_wait_until_completed_text(self, _mock_key, mock_fork, mock_get):
        from roboflow.cli.handlers.project import _fork_project

        mock_fork.return_value = {"taskId": "task-1", "url": "poll"}
        mock_get.side_effect = [
            {"taskId": "task-1", "status": "running", "progress": {"current": 1, "total": 2}},
            {
                "taskId": "task-1",
                "status": "completed",
                "result": {
                    "forked": True,
                    "datasetUrl": "license-plates",
                    "id": "test-ws/license-plates",
                    "name": "License Plates",
                    "url": "https://app.roboflow.com/test-ws/license-plates",
                },
            },
        ]
        args = _make_args(source="ws/proj")
        with patch("builtins.print") as mock_print:
            _fork_project(args)

        printed = mock_print.call_args[0][0]
        self.assertIn("Forked", printed)
        self.assertIn("Destination URL", printed)
        self.assertIn("https://app.roboflow.com/test-ws/license-plates", printed)
        mock_print.assert_any_call("Task progress: 1/2", flush=True)
        self.assertEqual(mock_get.call_count, 2)

    @patch("roboflow.core.async_tasks.time.sleep", lambda *_a, **_k: None)
    @patch("roboflow.adapters.rfapi.get_async_task")
    @patch("roboflow.adapters.rfapi.fork_project")
    @patch("roboflow.config.load_roboflow_api_key", return_value="test-key")
    def test_wait_builds_destination_url_from_dataset_url(self, _mock_key, mock_fork, mock_get):
        from roboflow.cli.handlers.project import _fork_project

        mock_fork.return_value = {"taskId": "task-1", "url": "poll"}
        mock_get.return_value = {
            "taskId": "task-1",
            "status": "completed",
            "result": {"forked": True, "datasetUrl": "license-plates"},
        }
        args = _make_args(source="ws/proj")
        with patch("builtins.print") as mock_print:
            _fork_project(args)

        printed = mock_print.call_args[0][0]
        self.assertIn("Destination URL", printed)
        self.assertIn("/test-ws/license-plates", printed)

    @patch("roboflow.core.async_tasks.time.sleep", lambda *_a, **_k: None)
    @patch("roboflow.adapters.rfapi.get_async_task")
    @patch("roboflow.adapters.rfapi.fork_project")
    @patch("roboflow.config.load_roboflow_api_key", return_value="test-key")
    def test_wait_until_completed_json(self, _mock_key, mock_fork, mock_get):
        from roboflow.cli.handlers.project import _fork_project

        mock_fork.return_value = {"taskId": "task-1", "url": "poll"}
        terminal_payload = {
            "taskId": "task-1",
            "status": "completed",
            "result": {"forked": True, "url": "https://app.roboflow.com/x/y"},
        }
        mock_get.return_value = terminal_payload
        args = _make_args(json=True, source="ws/proj")
        with patch("builtins.print") as mock_print:
            _fork_project(args)

        # Server payload is passed through unchanged in --json mode.
        out = json.loads(mock_print.call_args[0][0])
        self.assertEqual(out, terminal_payload)

    @patch("roboflow.core.async_tasks.time.sleep", lambda *_a, **_k: None)
    @patch("roboflow.adapters.rfapi.get_async_task")
    @patch("roboflow.adapters.rfapi.fork_project")
    @patch("roboflow.config.load_roboflow_api_key", return_value="test-key")
    def test_wait_until_failed_exits_one(self, _mock_key, mock_fork, mock_get):
        from roboflow.cli.handlers.project import _fork_project

        mock_fork.return_value = {"taskId": "task-1", "url": "poll"}
        mock_get.return_value = {
            "taskId": "task-1",
            "status": "failed",
            "error": "Source dataset is not public",
        }
        args = _make_args(source="ws/proj")
        with self.assertRaises(SystemExit) as ctx:
            _fork_project(args)
        self.assertEqual(ctx.exception.code, 1)


class TestForkProjectErrors(unittest.TestCase):
    def test_empty_source_exits(self):
        from roboflow.cli.handlers.project import _fork_project

        args = _make_args(source="")
        with self.assertRaises(SystemExit) as ctx:
            _fork_project(args)
        self.assertEqual(ctx.exception.code, 1)

    @patch("roboflow.adapters.rfapi.fork_project")
    @patch("roboflow.config.load_roboflow_api_key", return_value="test-key")
    def test_server_error_passes_through(self, _mock_key, mock_fork):
        from roboflow.adapters.rfapi import RoboflowError
        from roboflow.cli.handlers.project import _fork_project

        mock_fork.side_effect = RoboflowError('{"error":"You already own that dataset."}')
        args = _make_args(source="ws/proj")
        with self.assertRaises(SystemExit) as ctx:
            _fork_project(args)
        self.assertEqual(ctx.exception.code, 1)

    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value=None)
    def test_no_workspace_exits_two(self, _mock_resolve):
        from roboflow.cli.handlers.project import _fork_project

        args = _make_args(workspace=None, api_key=None, source="ws/proj")
        with self.assertRaises(SystemExit) as ctx:
            _fork_project(args)
        self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
