"""Tests for the `roboflow asynctasks` CLI handler."""

import json
import unittest
from argparse import Namespace
from unittest.mock import patch

from typer.testing import CliRunner

from roboflow.cli import app

runner = CliRunner()


def _make_args(**kwargs):
    defaults = {
        "json": False,
        "workspace": "test-ws",
        "api_key": "test-key",
        "quiet": False,
        "task_id": "task-123",
        "timeout": 1800,
        "poll_interval": 2.0,
    }
    defaults.update(kwargs)
    return Namespace(**defaults)


class TestAsyncTasksRegistration(unittest.TestCase):
    def test_app_exists(self) -> None:
        from roboflow.cli.handlers.asynctasks import asynctasks_app

        self.assertIsNotNone(asynctasks_app)

    def test_get_help(self) -> None:
        result = runner.invoke(app, ["asynctasks", "get", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_wait_help(self) -> None:
        result = runner.invoke(app, ["asynctasks", "wait", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--timeout", result.output)


class TestAsyncTaskGet(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.get_async_task")
    @patch("roboflow.config.load_roboflow_api_key", return_value="test-key")
    def test_get_text(self, _mock_key, mock_get):
        from roboflow.cli.handlers.asynctasks import _get_async_task

        mock_get.return_value = {
            "taskId": "task-123",
            "status": "running",
            "progress": {"percent": 42},
        }
        args = _make_args()
        with patch("builtins.print") as mock_print:
            _get_async_task(args)

        mock_get.assert_called_once_with("test-key", "test-ws", "task-123")
        printed = mock_print.call_args[0][0]
        self.assertIn("task-123", printed)
        self.assertIn("running", printed)

    @patch("roboflow.adapters.rfapi.get_async_task")
    @patch("roboflow.config.load_roboflow_api_key", return_value="test-key")
    def test_get_json(self, _mock_key, mock_get):
        from roboflow.cli.handlers.asynctasks import _get_async_task

        payload = {
            "taskId": "task-123",
            "status": "completed",
            "result": {"forked": True, "url": "https://app.roboflow.com/x/y"},
        }
        mock_get.return_value = payload
        args = _make_args(json=True)
        with patch("builtins.print") as mock_print:
            _get_async_task(args)

        # Server payload pass-through.
        out = json.loads(mock_print.call_args[0][0])
        self.assertEqual(out, payload)

    @patch("roboflow.adapters.rfapi.get_async_task")
    @patch("roboflow.config.load_roboflow_api_key", return_value="test-key")
    def test_get_404_exits_three(self, _mock_key, mock_get):
        from roboflow.adapters.rfapi import RoboflowError
        from roboflow.cli.handlers.asynctasks import _get_async_task

        mock_get.side_effect = RoboflowError('{"error":"Async task not found"}')
        args = _make_args()
        with self.assertRaises(SystemExit) as ctx:
            _get_async_task(args)
        self.assertEqual(ctx.exception.code, 3)


class TestAsyncTaskWait(unittest.TestCase):
    @patch("roboflow.cli._async_tasks.time.sleep", lambda *_a, **_k: None)
    @patch("roboflow.adapters.rfapi.get_async_task")
    @patch("roboflow.config.load_roboflow_api_key", return_value="test-key")
    def test_wait_until_completed(self, _mock_key, mock_get):
        from roboflow.cli.handlers.asynctasks import _wait_async_task

        mock_get.side_effect = [
            {"taskId": "task-1", "status": "pending", "progress": None},
            {"taskId": "task-1", "status": "running", "progress": None},
            {"taskId": "task-1", "status": "completed", "result": {"ok": True}},
        ]
        args = _make_args(task_id="task-1")
        with patch("builtins.print") as mock_print:
            _wait_async_task(args)

        self.assertEqual(mock_get.call_count, 3)
        printed = mock_print.call_args[0][0]
        self.assertIn("completed", printed)

    @patch("roboflow.cli._async_tasks.time.sleep", lambda *_a, **_k: None)
    @patch("roboflow.adapters.rfapi.get_async_task")
    @patch("roboflow.config.load_roboflow_api_key", return_value="test-key")
    def test_wait_until_failed_exits_one(self, _mock_key, mock_get):
        from roboflow.cli.handlers.asynctasks import _wait_async_task

        mock_get.return_value = {
            "taskId": "task-1",
            "status": "failed",
            "error": "Source dataset not public",
        }
        args = _make_args(task_id="task-1")
        with self.assertRaises(SystemExit) as ctx:
            _wait_async_task(args)
        self.assertEqual(ctx.exception.code, 1)

    @patch("roboflow.cli._async_tasks.time.sleep", lambda *_a, **_k: None)
    @patch("roboflow.cli._async_tasks.time.monotonic")
    @patch("roboflow.adapters.rfapi.get_async_task")
    @patch("roboflow.config.load_roboflow_api_key", return_value="test-key")
    def test_wait_timeout_exits_one(self, _mock_key, mock_get, mock_monotonic):
        from roboflow.cli.handlers.asynctasks import _wait_async_task

        # Two get_async_task calls, then deadline check trips.
        mock_get.return_value = {"taskId": "task-1", "status": "running"}
        # monotonic sequence: start, deadline-check-1 (still under), deadline-check-2 (over)
        mock_monotonic.side_effect = [0.0, 0.5, 99999.0]
        args = _make_args(task_id="task-1", timeout=10)
        with self.assertRaises(SystemExit) as ctx:
            _wait_async_task(args)
        self.assertEqual(ctx.exception.code, 1)

    @patch("roboflow.cli._async_tasks.time.sleep", lambda *_a, **_k: None)
    @patch("roboflow.adapters.rfapi.get_async_task")
    @patch("roboflow.config.load_roboflow_api_key", return_value="test-key")
    def test_wait_server_error_exits_three(self, _mock_key, mock_get):
        from roboflow.adapters.rfapi import RoboflowError
        from roboflow.cli.handlers.asynctasks import _wait_async_task

        mock_get.side_effect = RoboflowError('{"error":"Async task not found"}')
        args = _make_args(task_id="task-1")
        with self.assertRaises(SystemExit) as ctx:
            _wait_async_task(args)
        self.assertEqual(ctx.exception.code, 3)


class TestAsyncTaskNoWorkspace(unittest.TestCase):
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value=None)
    def test_get_no_workspace_exits_two(self, _mock_resolve):
        from roboflow.cli.handlers.asynctasks import _get_async_task

        args = _make_args(workspace=None, api_key=None)
        with self.assertRaises(SystemExit) as ctx:
            _get_async_task(args)
        self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
