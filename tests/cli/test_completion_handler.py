"""Tests for the completion CLI handler."""

import unittest

from typer.testing import CliRunner

from roboflow.cli import app

runner = CliRunner()


class TestCompletionRegistration(unittest.TestCase):
    """Verify completion handler registers expected subcommands."""

    def test_completion_app_exists(self) -> None:
        from roboflow.cli.handlers.completion import completion_app

        self.assertIsNotNone(completion_app)

    def test_completion_bash_exists(self) -> None:
        result = runner.invoke(app, ["completion", "bash", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_completion_zsh_exists(self) -> None:
        result = runner.invoke(app, ["completion", "zsh", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_completion_fish_exists(self) -> None:
        result = runner.invoke(app, ["completion", "fish", "--help"])
        self.assertEqual(result.exit_code, 0)


if __name__ == "__main__":
    unittest.main()
