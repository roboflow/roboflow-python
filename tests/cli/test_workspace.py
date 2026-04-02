"""Tests for the workspace CLI handler."""

import json
import unittest
from argparse import Namespace
from unittest.mock import patch


class TestWorkspaceRegistration(unittest.TestCase):
    """Verify workspace handler registers expected subcommands."""

    def test_register_callable(self) -> None:
        from roboflow.cli.handlers.workspace import register

        self.assertTrue(callable(register))

    def test_workspace_list_exists(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["workspace", "list"])
        self.assertIsNotNone(args.func)

    def test_workspace_get_positional(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["workspace", "get", "my-ws"])
        self.assertEqual(args.workspace_id, "my-ws")
        self.assertIsNotNone(args.func)

    def test_workspace_usage_exists(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["workspace", "usage"])
        self.assertIsNotNone(args.func)

    def test_workspace_plan_exists(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["workspace", "plan"])
        self.assertIsNotNone(args.func)

    def test_workspace_stats_exists(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["workspace", "stats", "--start-date", "2026-01-01", "--end-date", "2026-04-01"])
        self.assertIsNotNone(args.func)
        self.assertEqual(args.start_date, "2026-01-01")
        self.assertEqual(args.end_date, "2026-04-01")

    def test_handler_functions_exist(self) -> None:
        from roboflow.cli.handlers import workspace

        self.assertTrue(callable(workspace._list_workspaces))
        self.assertTrue(callable(workspace._get_workspace))
        self.assertTrue(callable(workspace._workspace_usage))
        self.assertTrue(callable(workspace._workspace_plan))
        self.assertTrue(callable(workspace._workspace_stats))


class TestWorkspaceUsageHandler(unittest.TestCase):
    """Test workspace usage command behavior."""

    @patch("roboflow.adapters.rfapi.get_billing_usage")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="test-ws")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_usage_json(self, _mock_key, _mock_ws, mock_usage):
        mock_usage.return_value = {"usage": {"inference_calls": 100, "images_uploaded": 50}}
        args = Namespace(json=True, workspace=None, api_key=None, quiet=False)

        from roboflow.cli.handlers.workspace import _workspace_usage

        with patch("builtins.print") as mock_print:
            _workspace_usage(args)
        printed = mock_print.call_args[0][0]
        data = json.loads(printed)
        self.assertIn("usage", data)

    @patch("roboflow.adapters.rfapi.get_billing_usage")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="test-ws")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_usage_text(self, _mock_key, _mock_ws, mock_usage):
        mock_usage.return_value = {"usage": {"inference_calls": 100}}
        args = Namespace(json=False, workspace=None, api_key=None, quiet=False)

        from roboflow.cli.handlers.workspace import _workspace_usage

        with patch("builtins.print") as mock_print:
            _workspace_usage(args)
        printed = mock_print.call_args[0][0]
        self.assertIn("Billing Usage", printed)

    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value=None)
    def test_usage_no_workspace(self, _mock_ws):
        args = Namespace(json=True, workspace=None, api_key=None, quiet=False)

        from roboflow.cli.handlers.workspace import _workspace_usage

        with self.assertRaises(SystemExit) as ctx:
            _workspace_usage(args)
        self.assertEqual(ctx.exception.code, 2)


class TestWorkspacePlanHandler(unittest.TestCase):
    """Test workspace plan command behavior."""

    @patch("roboflow.adapters.rfapi.get_plan_info")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="test-ws")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_plan_json(self, _mock_key, _mock_ws, mock_plan):
        mock_plan.return_value = {"plan": {"name": "Pro", "limit": 10000}}
        args = Namespace(json=True, workspace=None, api_key=None, quiet=False)

        from roboflow.cli.handlers.workspace import _workspace_plan

        with patch("builtins.print") as mock_print:
            _workspace_plan(args)
        printed = mock_print.call_args[0][0]
        data = json.loads(printed)
        self.assertIn("plan", data)

    @patch("roboflow.adapters.rfapi.get_plan_info")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="test-ws")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_plan_text(self, _mock_key, _mock_ws, mock_plan):
        mock_plan.return_value = {"plan": {"name": "Pro"}}
        args = Namespace(json=False, workspace=None, api_key=None, quiet=False)

        from roboflow.cli.handlers.workspace import _workspace_plan

        with patch("builtins.print") as mock_print:
            _workspace_plan(args)
        printed = mock_print.call_args[0][0]
        self.assertIn("Plan Info", printed)


class TestWorkspaceStatsHandler(unittest.TestCase):
    """Test workspace stats command behavior."""

    @patch("roboflow.adapters.rfapi.get_labeling_stats")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="test-ws")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_stats_json(self, _mock_key, _mock_ws, mock_stats):
        mock_stats.return_value = {"stats": {"total_annotations": 500}}
        args = Namespace(json=True, workspace=None, api_key=None, quiet=False, start_date="2026-01-01", end_date="2026-04-01")

        from roboflow.cli.handlers.workspace import _workspace_stats

        with patch("builtins.print") as mock_print:
            _workspace_stats(args)
        printed = mock_print.call_args[0][0]
        data = json.loads(printed)
        self.assertIn("stats", data)

    @patch("roboflow.adapters.rfapi.get_labeling_stats")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="test-ws")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_stats_passes_dates(self, _mock_key, _mock_ws, mock_stats):
        mock_stats.return_value = {"stats": {"total_annotations": 500}}
        args = Namespace(json=True, workspace=None, api_key=None, quiet=False, start_date="2026-01-01", end_date="2026-04-01")

        from roboflow.cli.handlers.workspace import _workspace_stats

        with patch("builtins.print"):
            _workspace_stats(args)
        mock_stats.assert_called_once_with("fake-key", "test-ws", start_date="2026-01-01", end_date="2026-04-01")

    @patch("roboflow.adapters.rfapi.get_labeling_stats")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="test-ws")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_stats_text(self, _mock_key, _mock_ws, mock_stats):
        mock_stats.return_value = {"stats": {"total_annotations": 500}}
        args = Namespace(json=False, workspace=None, api_key=None, quiet=False, start_date="2026-01-01", end_date="2026-04-01")

        from roboflow.cli.handlers.workspace import _workspace_stats

        with patch("builtins.print") as mock_print:
            _workspace_stats(args)
        printed = mock_print.call_args[0][0]
        self.assertIn("Labeling Stats", printed)

    @patch("roboflow.adapters.rfapi.get_labeling_stats", side_effect=Exception("server error"))
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="test-ws")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_stats_error_json(self, _mock_key, _mock_ws, _mock_stats):
        args = Namespace(json=True, workspace=None, api_key=None, quiet=False, start_date="2026-01-01", end_date="2026-04-01")

        from roboflow.cli.handlers.workspace import _workspace_stats

        with self.assertRaises(SystemExit) as ctx:
            _workspace_stats(args)
        self.assertEqual(ctx.exception.code, 3)


if __name__ == "__main__":
    unittest.main()
