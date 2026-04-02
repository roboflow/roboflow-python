"""Tests for the auth CLI handler."""

import unittest


class TestAuthRegistration(unittest.TestCase):
    """Verify auth handler registers expected subcommands."""

    def test_register_callable(self) -> None:
        from roboflow.cli.handlers.auth import register

        self.assertTrue(callable(register))

    def test_auth_subcommand_exists(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["auth", "status"])
        self.assertIsNotNone(args.func)

    def test_auth_login_defaults(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["auth", "login"])
        self.assertFalse(args.force)
        self.assertIsNone(args.login_api_key)
        self.assertIsNone(args.login_workspace)

    def test_auth_login_flags(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["auth", "login", "--api-key", "test123", "--force"])
        self.assertEqual(args.login_api_key, "test123")
        self.assertTrue(args.force)

    def test_auth_set_workspace_positional(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["auth", "set-workspace", "my-ws"])
        self.assertEqual(args.workspace_id, "my-ws")

    def test_auth_logout_has_func(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["auth", "logout"])
        self.assertIsNotNone(args.func)

    def test_handler_functions_exist(self) -> None:
        from roboflow.cli.handlers import auth

        # All handler functions should be importable
        self.assertTrue(callable(auth._login))
        self.assertTrue(callable(auth._status))
        self.assertTrue(callable(auth._set_workspace))
        self.assertTrue(callable(auth._logout))

    def test_mask_key(self) -> None:
        from roboflow.cli.handlers.auth import _mask_key

        self.assertEqual(_mask_key("abcdefgh"), "ab****gh")
        self.assertEqual(_mask_key("ab"), "****")
        self.assertEqual(_mask_key(""), "****")


if __name__ == "__main__":
    unittest.main()
