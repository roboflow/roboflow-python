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

    def test_trash_empty_exists(self) -> None:
        result = runner.invoke(app, ["trash", "empty", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_trash_delete_exists(self) -> None:
        result = runner.invoke(app, ["trash", "delete", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_subcommands_visible(self) -> None:
        result = runner.invoke(app, ["trash", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("list", result.output)
        self.assertIn("empty", result.output)
        self.assertIn("delete", result.output)


def _args(**overrides):
    base = {
        "json": False,
        "workspace": None,
        "api_key": "fake-key",
        "quiet": False,
        "yes": True,
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


class TestTrashEmptyHandler(unittest.TestCase):
    """trash empty calls rfapi.empty_trash and honors --yes."""

    def test_empty_calls_rfapi(self) -> None:
        from roboflow.cli.handlers.trash import _empty_trash

        with (
            patch("roboflow.cli._resolver.resolve_default_workspace", return_value="test-ws"),
            patch("roboflow.config.load_roboflow_api_key", return_value="fake-key"),
            patch("roboflow.adapters.rfapi.empty_trash", return_value={"dispatched": 5}) as mock_empty,
            patch("builtins.print"),
        ):
            _empty_trash(_args())
            mock_empty.assert_called_once_with("fake-key", "test-ws")


class TestTrashDeleteImmediatelyHandler(unittest.TestCase):
    """trash delete calls rfapi.trash_delete_immediately."""

    def test_delete_dataset(self) -> None:
        from roboflow.cli.handlers.trash import _delete_immediately

        with (
            patch("roboflow.cli._resolver.resolve_default_workspace", return_value="test-ws"),
            patch("roboflow.config.load_roboflow_api_key", return_value="fake-key"),
            patch(
                "roboflow.adapters.rfapi.trash_delete_immediately",
                return_value={"deleted": True},
            ) as mock_del,
            patch("builtins.print"),
        ):
            _delete_immediately(_args(item_type="dataset", item_id="abc123", parent_id=None))
            mock_del.assert_called_once_with("fake-key", "test-ws", "dataset", "abc123", None)

    def test_delete_version_with_parent(self) -> None:
        from roboflow.cli.handlers.trash import _delete_immediately

        with (
            patch("roboflow.cli._resolver.resolve_default_workspace", return_value="test-ws"),
            patch("roboflow.config.load_roboflow_api_key", return_value="fake-key"),
            patch(
                "roboflow.adapters.rfapi.trash_delete_immediately",
                return_value={"deleted": True},
            ) as mock_del,
            patch("builtins.print"),
        ):
            _delete_immediately(_args(item_type="version", item_id="3", parent_id="dataset-123"))
            mock_del.assert_called_once_with("fake-key", "test-ws", "version", "3", "dataset-123")


if __name__ == "__main__":
    unittest.main()
