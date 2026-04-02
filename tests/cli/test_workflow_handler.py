"""Tests for the workflow CLI handler."""

import json
import os
import tempfile
import unittest
from argparse import Namespace
from unittest.mock import patch


def _make_args(**kwargs):
    """Create a Namespace with CLI defaults."""
    defaults = {"json": False, "workspace": "test-ws", "api_key": "test-key", "quiet": False}
    defaults.update(kwargs)
    return Namespace(**defaults)


class TestWorkflowRegistration(unittest.TestCase):
    """Verify workflow handler registers expected subcommands."""

    def test_register_callable(self) -> None:
        from roboflow.cli.handlers.workflow import register

        self.assertTrue(callable(register))

    def test_workflow_list_exists(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["workflow", "list"])
        self.assertIsNotNone(args.func)

    def test_workflow_get_exists(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["workflow", "get", "my-workflow"])
        self.assertEqual(args.workflow_url, "my-workflow")
        self.assertIsNotNone(args.func)

    def test_workflow_create_exists(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["workflow", "create", "--name", "test-wf"])
        self.assertEqual(args.name, "test-wf")
        self.assertIsNotNone(args.func)

    def test_workflow_update_exists(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["workflow", "update", "my-wf"])
        self.assertIsNotNone(args.func)

    def test_workflow_version_list_exists(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["workflow", "version", "list", "my-wf"])
        self.assertIsNotNone(args.func)

    def test_workflow_fork_exists(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["workflow", "fork", "my-wf"])
        self.assertIsNotNone(args.func)

    def test_workflow_build_exists(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["workflow", "build", "detect objects in a video"])
        self.assertEqual(args.prompt, "detect objects in a video")
        self.assertIsNotNone(args.func)

    def test_workflow_run_exists(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["workflow", "run", "my-wf", "--input", "image.jpg"])
        self.assertEqual(args.input, "image.jpg")
        self.assertIsNotNone(args.func)

    def test_workflow_deploy_exists(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["workflow", "deploy", "my-wf"])
        self.assertIsNotNone(args.func)


class TestWorkflowList(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.list_workflows")
    @patch("roboflow.config.load_roboflow_api_key", return_value="test-key")
    def test_list_workflows_text(self, _mock_key, mock_list):
        from roboflow.cli.handlers.workflow import _list_workflows

        mock_list.return_value = {
            "workflows": [
                {"name": "My Workflow", "url": "my-workflow", "status": "active"},
            ]
        }
        args = _make_args()
        with patch("builtins.print") as mock_print:
            _list_workflows(args)
        mock_list.assert_called_once_with("test-key", "test-ws")
        printed = mock_print.call_args[0][0]
        self.assertIn("My Workflow", printed)

    @patch("roboflow.adapters.rfapi.list_workflows")
    @patch("roboflow.config.load_roboflow_api_key", return_value="test-key")
    def test_list_workflows_json(self, _mock_key, mock_list):
        from roboflow.cli.handlers.workflow import _list_workflows

        mock_list.return_value = {
            "workflows": [
                {"name": "WF1", "url": "wf-1", "status": "active"},
            ]
        }
        args = _make_args(json=True)
        with patch("builtins.print") as mock_print:
            _list_workflows(args)
        out = json.loads(mock_print.call_args[0][0])
        self.assertIsInstance(out, list)
        self.assertEqual(out[0]["name"], "WF1")

    @patch("roboflow.adapters.rfapi.list_workflows")
    @patch("roboflow.config.load_roboflow_api_key", return_value="test-key")
    def test_list_workflows_error(self, _mock_key, mock_list):
        from roboflow.adapters.rfapi import RoboflowError
        from roboflow.cli.handlers.workflow import _list_workflows

        mock_list.side_effect = RoboflowError("Not found")
        args = _make_args()
        with self.assertRaises(SystemExit) as ctx:
            _list_workflows(args)
        self.assertEqual(ctx.exception.code, 3)


class TestWorkflowGet(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.get_workflow")
    @patch("roboflow.config.load_roboflow_api_key", return_value="test-key")
    def test_get_workflow_text(self, _mock_key, mock_get):
        from roboflow.cli.handlers.workflow import _get_workflow

        mock_get.return_value = {
            "workflow": {
                "name": "My WF",
                "url": "my-wf",
                "description": "A test workflow",
                "blockCount": 5,
            }
        }
        args = _make_args(workflow_url="my-wf")
        with patch("builtins.print") as mock_print:
            _get_workflow(args)
        mock_get.assert_called_once_with("test-key", "test-ws", "my-wf")
        printed = mock_print.call_args[0][0]
        self.assertIn("My WF", printed)
        self.assertIn("5", printed)

    @patch("roboflow.adapters.rfapi.get_workflow")
    @patch("roboflow.config.load_roboflow_api_key", return_value="test-key")
    def test_get_workflow_json(self, _mock_key, mock_get):
        from roboflow.cli.handlers.workflow import _get_workflow

        mock_get.return_value = {"workflow": {"name": "My WF", "url": "my-wf"}}
        args = _make_args(json=True, workflow_url="my-wf")
        with patch("builtins.print") as mock_print:
            _get_workflow(args)
        out = json.loads(mock_print.call_args[0][0])
        self.assertIn("workflow", out)


class TestWorkflowCreate(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.create_workflow")
    @patch("roboflow.config.load_roboflow_api_key", return_value="test-key")
    def test_create_workflow_basic(self, _mock_key, mock_create):
        from roboflow.cli.handlers.workflow import _create_workflow

        mock_create.return_value = {"name": "New WF", "url": "new-wf"}
        args = _make_args(name="New WF", definition=None, description=None)
        with patch("builtins.print") as mock_print:
            _create_workflow(args)
        mock_create.assert_called_once_with("test-key", "test-ws", name="New WF", config="{}", template="{}")
        printed = mock_print.call_args[0][0]
        self.assertIn("Created workflow", printed)

    @patch("roboflow.adapters.rfapi.create_workflow")
    @patch("roboflow.config.load_roboflow_api_key", return_value="test-key")
    def test_create_workflow_with_definition(self, _mock_key, mock_create):
        from roboflow.cli.handlers.workflow import _create_workflow

        mock_create.return_value = {"name": "New WF", "url": "new-wf"}
        defn = {"blocks": [{"type": "input"}]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(defn, f)
            f.flush()
            tmp_path = f.name

        try:
            args = _make_args(name="New WF", definition=tmp_path, description="A desc")
            with patch("builtins.print"):
                _create_workflow(args)
            mock_create.assert_called_once_with(
                "test-key", "test-ws", name="New WF", config=json.dumps(defn), template="{}"
            )
        finally:
            os.unlink(tmp_path)

    def test_create_workflow_missing_file(self):
        from roboflow.cli.handlers.workflow import _create_workflow

        args = _make_args(name="New WF", definition="/nonexistent/file.json", description=None)
        with self.assertRaises(SystemExit) as ctx:
            _create_workflow(args)
        self.assertEqual(ctx.exception.code, 1)

    def test_create_workflow_invalid_json(self):
        from roboflow.cli.handlers.workflow import _create_workflow

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{bad json")
            f.flush()
            tmp_path = f.name

        try:
            args = _make_args(name="New WF", definition=tmp_path, description=None)
            with self.assertRaises(SystemExit) as ctx:
                _create_workflow(args)
            self.assertEqual(ctx.exception.code, 1)
        finally:
            os.unlink(tmp_path)


class TestWorkflowUpdate(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.update_workflow")
    @patch("roboflow.adapters.rfapi.get_workflow")
    @patch("roboflow.config.load_roboflow_api_key", return_value="test-key")
    def test_update_workflow(self, _mock_key, mock_get, mock_update):
        from roboflow.cli.handlers.workflow import _update_workflow

        mock_get.return_value = {"workflow": {"id": "wf-123", "name": "My WF", "url": "my-wf", "config": "{}"}}
        mock_update.return_value = {"url": "my-wf", "status": "updated"}
        defn = {"blocks": []}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(defn, f)
            f.flush()
            tmp_path = f.name

        try:
            args = _make_args(workflow_url="my-wf", definition=tmp_path)
            with patch("builtins.print") as mock_print:
                _update_workflow(args)
            mock_get.assert_called_once_with("test-key", "test-ws", "my-wf")
            mock_update.assert_called_once_with(
                "test-key",
                "test-ws",
                workflow_id="wf-123",
                workflow_name="My WF",
                workflow_url="my-wf",
                config=json.dumps(defn),
            )
            printed = mock_print.call_args[0][0]
            self.assertIn("Updated workflow", printed)
        finally:
            os.unlink(tmp_path)

    @patch("roboflow.adapters.rfapi.update_workflow")
    @patch("roboflow.adapters.rfapi.get_workflow")
    @patch("roboflow.config.load_roboflow_api_key", return_value="test-key")
    def test_update_workflow_no_definition(self, _mock_key, mock_get, mock_update):
        """When no --definition is given, existing config is preserved."""
        from roboflow.cli.handlers.workflow import _update_workflow

        mock_get.return_value = {
            "workflow": {"id": "wf-123", "name": "My WF", "url": "my-wf", "config": '{"existing": true}'}
        }
        mock_update.return_value = {"url": "my-wf", "status": "updated"}
        args = _make_args(workflow_url="my-wf", definition=None)
        with patch("builtins.print"):
            _update_workflow(args)
        mock_update.assert_called_once_with(
            "test-key",
            "test-ws",
            workflow_id="wf-123",
            workflow_name="My WF",
            workflow_url="my-wf",
            config='{"existing": true}',
        )

    def test_update_workflow_missing_file(self):
        from roboflow.cli.handlers.workflow import _update_workflow

        args = _make_args(workflow_url="my-wf", definition="/nonexistent/file.json")
        with self.assertRaises(SystemExit) as ctx:
            _update_workflow(args)
        self.assertEqual(ctx.exception.code, 1)


class TestWorkflowVersionList(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.list_workflow_versions")
    @patch("roboflow.config.load_roboflow_api_key", return_value="test-key")
    def test_list_versions(self, _mock_key, mock_versions):
        from roboflow.cli.handlers.workflow import _list_workflow_versions

        mock_versions.return_value = {
            "versions": [
                {"version": "1", "created": "2026-01-01"},
                {"version": "2", "created": "2026-02-01"},
            ]
        }
        args = _make_args(workflow_url="my-wf")
        with patch("builtins.print") as mock_print:
            _list_workflow_versions(args)
        mock_versions.assert_called_once_with("test-key", "test-ws", "my-wf")
        printed = mock_print.call_args[0][0]
        self.assertIn("1", printed)
        self.assertIn("2", printed)

    @patch("roboflow.adapters.rfapi.list_workflow_versions")
    @patch("roboflow.config.load_roboflow_api_key", return_value="test-key")
    def test_list_versions_json(self, _mock_key, mock_versions):
        from roboflow.cli.handlers.workflow import _list_workflow_versions

        mock_versions.return_value = {"versions": [{"version": "1", "created": "2026-01-01"}]}
        args = _make_args(json=True, workflow_url="my-wf")
        with patch("builtins.print") as mock_print:
            _list_workflow_versions(args)
        out = json.loads(mock_print.call_args[0][0])
        self.assertIsInstance(out, list)


class TestWorkflowFork(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.fork_workflow")
    @patch("roboflow.config.load_roboflow_api_key", return_value="test-key")
    def test_fork_workflow_same_workspace(self, _mock_key, mock_fork):
        """When workflow_url is just a slug, source_workspace defaults to current ws."""
        from roboflow.cli.handlers.workflow import _fork_workflow

        mock_fork.return_value = {"url": "my-wf-fork", "workflow_url": "my-wf-fork"}
        args = _make_args(workflow_url="my-wf")
        with patch("builtins.print") as mock_print:
            _fork_workflow(args)
        mock_fork.assert_called_once_with("test-key", "test-ws", source_workspace="test-ws", source_workflow="my-wf")
        printed = mock_print.call_args[0][0]
        self.assertIn("Forked workflow", printed)

    @patch("roboflow.adapters.rfapi.fork_workflow")
    @patch("roboflow.config.load_roboflow_api_key", return_value="test-key")
    def test_fork_workflow_cross_workspace(self, _mock_key, mock_fork):
        """When workflow_url is 'other-ws/my-wf', source_workspace is parsed."""
        from roboflow.cli.handlers.workflow import _fork_workflow

        mock_fork.return_value = {"url": "my-wf-fork"}
        args = _make_args(workflow_url="other-ws/my-wf")
        with patch("builtins.print"):
            _fork_workflow(args)
        mock_fork.assert_called_once_with("test-key", "test-ws", source_workspace="other-ws", source_workflow="my-wf")

    @patch("roboflow.adapters.rfapi.fork_workflow")
    @patch("roboflow.config.load_roboflow_api_key", return_value="test-key")
    def test_fork_workflow_json(self, _mock_key, mock_fork):
        from roboflow.cli.handlers.workflow import _fork_workflow

        mock_fork.return_value = {"url": "my-wf-fork"}
        args = _make_args(json=True, workflow_url="my-wf")
        with patch("builtins.print") as mock_print:
            _fork_workflow(args)
        out = json.loads(mock_print.call_args[0][0])
        self.assertEqual(out["status"], "forked")
        self.assertEqual(out["source"], "my-wf")
        self.assertEqual(out["new_url"], "my-wf-fork")


class TestWorkflowStubs(unittest.TestCase):
    def test_build_stub(self):
        from roboflow.cli.handlers.workflow import _stub_build

        args = _make_args()
        with self.assertRaises(SystemExit):
            _stub_build(args)

    def test_run_stub(self):
        from roboflow.cli.handlers.workflow import _stub_run

        args = _make_args()
        with self.assertRaises(SystemExit):
            _stub_run(args)

    def test_deploy_stub(self):
        from roboflow.cli.handlers.workflow import _stub_deploy

        args = _make_args()
        with self.assertRaises(SystemExit):
            _stub_deploy(args)


class TestWorkflowNoWorkspace(unittest.TestCase):
    """Verify proper error when no workspace is available."""

    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value=None)
    def test_list_no_workspace(self, _mock_resolve):
        from roboflow.cli.handlers.workflow import _list_workflows

        args = _make_args(workspace=None, api_key=None)
        with self.assertRaises(SystemExit) as ctx:
            _list_workflows(args)
        self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
