"""Tests for the auth CLI handler."""

import re
import types
import unittest

from typer.testing import CliRunner

from roboflow.cli import app
from roboflow.cli.handlers import auth as auth_module

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


class TestAuthRegistration(unittest.TestCase):
    """Verify auth handler registers expected subcommands."""

    def test_auth_app_exists(self) -> None:
        from roboflow.cli.handlers.auth import auth_app

        self.assertIsNotNone(auth_app)

    def test_auth_subcommand_exists(self) -> None:
        result = runner.invoke(app, ["auth", "status", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_auth_login_exists(self) -> None:
        result = runner.invoke(app, ["auth", "login", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_auth_login_help_shows_flags(self) -> None:
        result = runner.invoke(app, ["auth", "login", "--help"])
        self.assertEqual(result.exit_code, 0)
        output = _strip_ansi(result.output).lower()
        self.assertIn("api-key", output)
        self.assertIn("force", output)

    def test_auth_set_workspace_exists(self) -> None:
        result = runner.invoke(app, ["auth", "set-workspace", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_auth_logout_exists(self) -> None:
        result = runner.invoke(app, ["auth", "logout", "--help"])
        self.assertEqual(result.exit_code, 0)

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


class TestCompletionTip(unittest.TestCase):
    """Verify the post-login completion tip honours --json and --quiet."""

    def _capture(self, args_ns) -> str:  # noqa: ANN001
        import io
        import sys

        buf = io.StringIO()
        prev = sys.stdout
        sys.stdout = buf
        try:
            auth_module._print_completion_tip(args_ns)
        finally:
            sys.stdout = prev
        return buf.getvalue()

    def test_tip_printed_in_normal_mode(self) -> None:
        out = self._capture(types.SimpleNamespace(json=False, quiet=False))
        self.assertIn("roboflow completion install", out)

    def test_tip_suppressed_in_json_mode(self) -> None:
        out = self._capture(types.SimpleNamespace(json=True, quiet=False))
        self.assertEqual(out, "")

    def test_tip_suppressed_in_quiet_mode(self) -> None:
        out = self._capture(types.SimpleNamespace(json=False, quiet=True))
        self.assertEqual(out, "")


if __name__ == "__main__":
    unittest.main()
