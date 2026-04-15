"""Tests that the roboflowpy.py backwards-compatibility shim works.

Ensures that existing scripts and integrations that import from the old
monolithic module continue to work after the CLI modularization and
typer migration.
"""

import unittest


class TestRoboflowpyShim(unittest.TestCase):
    """Verify the roboflowpy.py shim re-exports work."""

    def test_main_importable(self) -> None:
        from roboflow.roboflowpy import main

        self.assertTrue(callable(main))

    def test_argparser_importable(self) -> None:
        """debugme.py imports _argparser — this must not break."""
        from roboflow.roboflowpy import _argparser

        self.assertTrue(callable(_argparser))

    def test_argparser_returns_object_with_parse_args(self) -> None:
        """_argparser() must return an object with parse_args() method."""
        from roboflow.roboflowpy import _argparser

        parser = _argparser()
        self.assertIsNotNone(parser)
        self.assertTrue(hasattr(parser, "parse_args"))
        self.assertTrue(callable(parser.parse_args))

    def test_argparser_has_print_help(self) -> None:
        """The parser should support print_help() for interactive use."""
        from roboflow.roboflowpy import _argparser

        parser = _argparser()
        self.assertTrue(hasattr(parser, "print_help"))

    def test_cli_commands_work_via_typer_runner(self) -> None:
        """Verify commands execute through typer's CliRunner."""
        from typer.testing import CliRunner

        from roboflow.cli import app

        runner = CliRunner()

        # --version
        result = runner.invoke(app, ["--version"])
        self.assertEqual(result.exit_code, 0)

        # --help
        result = runner.invoke(app, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("project", result.output)

        # Legacy alias: login --help
        result = runner.invoke(app, ["login", "--help"])
        self.assertEqual(result.exit_code, 0)

        # Legacy alias: whoami --help
        result = runner.invoke(app, ["whoami", "--help"])
        self.assertEqual(result.exit_code, 0)

        # Legacy alias: download --help
        result = runner.invoke(app, ["download", "--help"])
        self.assertEqual(result.exit_code, 0)


if __name__ == "__main__":
    unittest.main()
