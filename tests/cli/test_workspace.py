"""Tests for the workspace CLI handler."""

import unittest


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

    def test_handler_functions_exist(self) -> None:
        from roboflow.cli.handlers import workspace

        self.assertTrue(callable(workspace._list_workspaces))
        self.assertTrue(callable(workspace._get_workspace))


if __name__ == "__main__":
    unittest.main()
