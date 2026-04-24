"""Tests for the trash CLI handler."""

import unittest
from argparse import Namespace
from unittest.mock import patch

from typer.testing import CliRunner

from roboflow.cli import app

runner = CliRunner()


class TestTrashRegistration(unittest.TestCase):
    """Verify trash handler registers expected subcommands."""

    def test_trash_app_exists(self) -> None:
        from roboflow.cli.handlers.trash import trash_app

        self.assertIsNotNone(trash_app)

    def test_trash_list_exists(self) -> None:
        result = runner.invoke(app, ["trash", "list", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_permanent_delete_commands_not_exposed(self) -> None:
        # empty / delete immediately are intentionally not available on the
        # SDK/CLI — they exist only in the web UI. Guard against regression.
        empty_result = runner.invoke(app, ["trash", "empty", "--help"])
        self.assertNotEqual(empty_result.exit_code, 0)
        delete_result = runner.invoke(app, ["trash", "delete", "--help"])
        self.assertNotEqual(delete_result.exit_code, 0)

    def test_subcommands_visible(self) -> None:
        result = runner.invoke(app, ["trash", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("list", result.output)
        # empty / delete should NOT appear in the command group
        self.assertNotIn("empty", result.output.lower())


def _args(**overrides):
    base = {
        "json": False,
        "workspace": None,
        "api_key": "fake-key",
        "quiet": False,
    }
    base.update(overrides)
    return Namespace(**base)


class TestTrashListHandler(unittest.TestCase):
    """trash list calls rfapi.list_trash and formats the output."""

    def test_list_text_output(self) -> None:
        from roboflow.cli.handlers.trash import _list_trash

        trash_response = {
            "items": [
                {
                    "type": "dataset",
                    "id": "d1",
                    "name": "My Project",
                    "deletedAt": "2026-04-01",
                    "scheduledCleanupAt": "2026-05-01",
                    "deletedByName": "Alice",
                },
                {
                    "type": "version",
                    "id": "3",
                    "name": "v3",
                    "parentName": "My Project",
                    "parentUrl": "my-proj",
                    "deletedAt": "2026-04-02",
                    "scheduledCleanupAt": "2026-05-02",
                    "deletedByName": "Bob",
                },
            ],
            "sections": {},
        }
        with (
            patch("roboflow.cli._resolver.resolve_default_workspace", return_value="test-ws"),
            patch("roboflow.config.load_roboflow_api_key", return_value="fake-key"),
            patch("roboflow.adapters.rfapi.list_trash", return_value=trash_response) as mock_list,
            patch("builtins.print") as mock_print,
        ):
            _list_trash(_args())
            mock_list.assert_called_once_with("fake-key", "test-ws")
            mock_print.assert_called_once()
            printed = mock_print.call_args[0][0]
            self.assertIn("My Project", printed)
            self.assertIn("v3", printed)

    def test_list_empty(self) -> None:
        from roboflow.cli.handlers.trash import _list_trash

        with (
            patch("roboflow.cli._resolver.resolve_default_workspace", return_value="test-ws"),
            patch("roboflow.config.load_roboflow_api_key", return_value="fake-key"),
            patch(
                "roboflow.adapters.rfapi.list_trash",
                return_value={"items": [], "sections": {}},
            ),
            patch("builtins.print") as mock_print,
        ):
            _list_trash(_args())
            printed = mock_print.call_args[0][0]
            self.assertIn("empty", printed.lower())


class TestRfapiSurface(unittest.TestCase):
    """Guard: rfapi must not expose permanent-delete wrappers."""

    def test_no_trash_delete_immediately(self) -> None:
        from roboflow.adapters import rfapi

        self.assertFalse(hasattr(rfapi, "trash_delete_immediately"))

    def test_no_empty_trash(self) -> None:
        from roboflow.adapters import rfapi

        self.assertFalse(hasattr(rfapi, "empty_trash"))


class TestWorkspaceSurface(unittest.TestCase):
    """Guard: Workspace must not expose permanent-delete helpers."""

    def test_no_delete_from_trash(self) -> None:
        from roboflow.core.workspace import Workspace

        self.assertFalse(hasattr(Workspace, "delete_from_trash"))

    def test_no_empty_trash(self) -> None:
        from roboflow.core.workspace import Workspace

        self.assertFalse(hasattr(Workspace, "empty_trash"))


if __name__ == "__main__":
    unittest.main()
