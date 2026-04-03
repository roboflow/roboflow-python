"""Tests for the search CLI handler."""

import unittest

from typer.testing import CliRunner

from roboflow.cli import app

runner = CliRunner()


class TestSearchRegistration(unittest.TestCase):
    """Verify search handler registers expected subcommands."""

    def test_search_command_callable(self) -> None:
        from roboflow.cli.handlers.search import search_command

        self.assertTrue(callable(search_command))

    def test_search_subcommand_exists(self) -> None:
        result = runner.invoke(app, ["search", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_search_help_shows_options(self) -> None:
        result = runner.invoke(app, ["search", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("limit", result.output.lower())
        self.assertIn("cursor", result.output.lower())
        self.assertIn("export", result.output.lower())


if __name__ == "__main__":
    unittest.main()
