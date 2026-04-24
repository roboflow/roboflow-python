"""Tests for the project CLI handler."""

import unittest

from typer.testing import CliRunner

from roboflow.cli import app

runner = CliRunner()


class TestProjectHandlerRegistration(unittest.TestCase):
    """Verify that the project handler registers correctly."""

    def test_project_list_exists(self) -> None:
        result = runner.invoke(app, ["project", "list", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_project_list_help_shows_type(self) -> None:
        result = runner.invoke(app, ["project", "list", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("type", result.output.lower())

    def test_project_get_exists(self) -> None:
        result = runner.invoke(app, ["project", "get", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_project_create_exists(self) -> None:
        result = runner.invoke(app, ["project", "create", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("type", result.output.lower())

    def test_project_delete_exists(self) -> None:
        result = runner.invoke(app, ["project", "delete", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("trash", result.output.lower())

    def test_project_restore_exists(self) -> None:
        result = runner.invoke(app, ["project", "restore", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("trash", result.output.lower())

    def test_subcommands_visible(self) -> None:
        result = runner.invoke(app, ["project", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("list", result.output)
        self.assertIn("get", result.output)
        self.assertIn("create", result.output)
        self.assertIn("delete", result.output)
        self.assertIn("restore", result.output)


class TestProjectDeleteHandler(unittest.TestCase):
    """project delete calls rfapi.delete_project and honors --yes."""

    def _args(self, project_id="my-ws/my-proj"):
        from argparse import Namespace

        return Namespace(
            json=False,
            workspace=None,
            api_key="fake-key",
            quiet=False,
            project_id=project_id,
            yes=True,
        )

    def test_delete_calls_rfapi(self) -> None:
        from unittest.mock import patch

        from roboflow.cli.handlers.project import _delete_project

        with patch("roboflow.adapters.rfapi.delete_project", return_value={"deleted": True}) as mock_del:
            _delete_project(self._args())
            mock_del.assert_called_once_with("fake-key", "my-ws", "my-proj")


class TestProjectRestoreHandler(unittest.TestCase):
    """project restore looks up the item in Trash by URL, then restores."""

    def _args(self, project_id="my-ws/my-proj"):
        from argparse import Namespace

        return Namespace(
            json=False,
            workspace=None,
            api_key="fake-key",
            quiet=False,
            project_id=project_id,
        )

    def test_restore_found(self) -> None:
        from unittest.mock import patch

        from roboflow.cli.handlers.project import _restore_project

        trash = {"sections": {"datasets": [{"id": "abc123", "url": "my-proj", "name": "My Project"}]}}
        with (
            patch("roboflow.adapters.rfapi.list_trash", return_value=trash),
            patch(
                "roboflow.adapters.rfapi.restore_trash_item",
                return_value={"restored": True, "type": "dataset", "id": "abc123"},
            ) as mock_restore,
        ):
            _restore_project(self._args())
            mock_restore.assert_called_once_with("fake-key", "my-ws", "dataset", "abc123")

    def test_restore_not_in_trash(self) -> None:
        from unittest.mock import patch

        from roboflow.cli.handlers.project import _restore_project

        # Trash doesn't contain this project — handler should error without
        # calling restore_trash_item.
        with (
            patch(
                "roboflow.adapters.rfapi.list_trash",
                return_value={"sections": {"datasets": []}},
            ),
            patch("roboflow.adapters.rfapi.restore_trash_item") as mock_restore,
            patch("sys.exit"),
        ):
            _restore_project(self._args())
            mock_restore.assert_not_called()


if __name__ == "__main__":
    unittest.main()
