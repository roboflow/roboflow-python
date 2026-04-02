"""Tests for the folder CLI handler."""

import json
import unittest
from argparse import Namespace
from unittest.mock import patch


class TestFolderRegistration(unittest.TestCase):
    """Verify folder handler registers expected subcommands."""

    def test_register_callable(self) -> None:
        from roboflow.cli.handlers.folder import register

        self.assertTrue(callable(register))

    def test_folder_list_exists(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["folder", "list"])
        self.assertIsNotNone(args.func)

    def test_folder_get_exists(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["folder", "get", "folder-123"])
        self.assertIsNotNone(args.func)
        self.assertEqual(args.folder_id, "folder-123")

    def test_folder_create_exists(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["folder", "create", "My Folder"])
        self.assertIsNotNone(args.func)
        self.assertEqual(args.name, "My Folder")

    def test_folder_create_with_flags(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["folder", "create", "My Folder", "--parent", "p1", "--projects", "a,b"])
        self.assertEqual(args.parent, "p1")
        self.assertEqual(args.projects, "a,b")

    def test_folder_update_exists(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["folder", "update", "folder-123", "--name", "New Name"])
        self.assertIsNotNone(args.func)

    def test_folder_delete_exists(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["folder", "delete", "folder-123"])
        self.assertIsNotNone(args.func)


class TestFolderListHandler(unittest.TestCase):
    """Test folder list command behavior."""

    @patch("roboflow.adapters.rfapi.list_folders")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="test-ws")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_list_folders_text(self, _mock_key, _mock_ws, mock_list):
        mock_list.return_value = {"data": [{"name": "Folder1", "id": "f1", "projects": ["p1", "p2"]}]}
        args = Namespace(json=False, workspace=None, api_key=None, quiet=False)

        from roboflow.cli.handlers.folder import _list_folders

        with patch("builtins.print") as mock_print:
            _list_folders(args)
        mock_print.assert_called_once()
        printed = mock_print.call_args[0][0]
        self.assertIn("Folder1", printed)

    @patch("roboflow.adapters.rfapi.list_folders")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="test-ws")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_list_folders_json(self, _mock_key, _mock_ws, mock_list):
        mock_list.return_value = {"data": [{"name": "Folder1", "id": "f1", "projects": []}]}
        args = Namespace(json=True, workspace=None, api_key=None, quiet=False)

        from roboflow.cli.handlers.folder import _list_folders

        with patch("builtins.print") as mock_print:
            _list_folders(args)
        printed = mock_print.call_args[0][0]
        data = json.loads(printed)
        self.assertIsInstance(data, list)
        self.assertEqual(data[0]["name"], "Folder1")

    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value=None)
    def test_list_folders_no_workspace(self, _mock_ws):
        args = Namespace(json=True, workspace=None, api_key=None, quiet=False)

        from roboflow.cli.handlers.folder import _list_folders

        with self.assertRaises(SystemExit) as ctx:
            _list_folders(args)
        self.assertEqual(ctx.exception.code, 2)


class TestFolderGetHandler(unittest.TestCase):
    """Test folder get command behavior."""

    @patch("roboflow.adapters.rfapi.get_folder")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="test-ws")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_get_folder_text(self, _mock_key, _mock_ws, mock_get):
        mock_get.return_value = {"data": [{"name": "MyFolder", "id": "f1", "projects": []}]}
        args = Namespace(json=False, workspace=None, api_key=None, quiet=False, folder_id="f1")

        from roboflow.cli.handlers.folder import _get_folder

        with patch("builtins.print") as mock_print:
            _get_folder(args)
        printed = mock_print.call_args[0][0]
        self.assertIn("MyFolder", printed)


class TestFolderCreateHandler(unittest.TestCase):
    """Test folder create command behavior."""

    @patch("roboflow.adapters.rfapi.create_folder")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="test-ws")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_create_folder_json(self, _mock_key, _mock_ws, mock_create):
        mock_create.return_value = {"id": "new-folder-id"}
        args = Namespace(
            json=True, workspace=None, api_key=None, quiet=False, name="NewFolder", parent=None, projects=None
        )

        from roboflow.cli.handlers.folder import _create_folder

        with patch("builtins.print") as mock_print:
            _create_folder(args)
        printed = mock_print.call_args[0][0]
        data = json.loads(printed)
        self.assertEqual(data["status"], "created")
        self.assertEqual(data["id"], "new-folder-id")

    @patch("roboflow.adapters.rfapi.create_folder")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="test-ws")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_create_folder_with_projects(self, _mock_key, _mock_ws, mock_create):
        mock_create.return_value = {"id": "f2"}
        args = Namespace(json=False, workspace=None, api_key=None, quiet=False, name="F", parent="p1", projects="a,b,c")

        from roboflow.cli.handlers.folder import _create_folder

        with patch("builtins.print"):
            _create_folder(args)
        mock_create.assert_called_once_with("fake-key", "test-ws", "F", parent_id="p1", project_ids=["a", "b", "c"])


class TestFolderUpdateHandler(unittest.TestCase):
    """Test folder update command behavior."""

    @patch("roboflow.adapters.rfapi.update_folder")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="test-ws")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_update_folder_json(self, _mock_key, _mock_ws, mock_update):
        mock_update.return_value = {}
        args = Namespace(json=True, workspace=None, api_key=None, quiet=False, folder_id="f1", name="Renamed")

        from roboflow.cli.handlers.folder import _update_folder

        with patch("builtins.print") as mock_print:
            _update_folder(args)
        printed = mock_print.call_args[0][0]
        data = json.loads(printed)
        self.assertEqual(data["status"], "updated")


class TestFolderDeleteHandler(unittest.TestCase):
    """Test folder delete command behavior."""

    @patch("roboflow.adapters.rfapi.delete_folder")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="test-ws")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_delete_folder_json(self, _mock_key, _mock_ws, mock_delete):
        mock_delete.return_value = {}
        args = Namespace(json=True, workspace=None, api_key=None, quiet=False, folder_id="f1")

        from roboflow.cli.handlers.folder import _delete_folder

        with patch("builtins.print") as mock_print:
            _delete_folder(args)
        printed = mock_print.call_args[0][0]
        data = json.loads(printed)
        self.assertEqual(data["status"], "deleted")

    @patch("roboflow.adapters.rfapi.delete_folder", side_effect=Exception("Not found"))
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="test-ws")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_delete_folder_error_json(self, _mock_key, _mock_ws, _mock_delete):
        args = Namespace(json=True, workspace=None, api_key=None, quiet=False, folder_id="bad-id")

        from roboflow.cli.handlers.folder import _delete_folder

        with self.assertRaises(SystemExit) as ctx:
            _delete_folder(args)
        self.assertEqual(ctx.exception.code, 3)


if __name__ == "__main__":
    unittest.main()
