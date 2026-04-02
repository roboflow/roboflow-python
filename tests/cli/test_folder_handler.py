"""Tests for the folder CLI handler."""

import unittest


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


if __name__ == "__main__":
    unittest.main()
