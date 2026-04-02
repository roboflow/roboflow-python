"""Tests that the roboflowpy.py backwards-compatibility shim works.

Ensures that existing scripts and integrations that import from the old
monolithic module continue to work after the CLI modularization.
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

    def test_argparser_returns_parser(self) -> None:
        import argparse

        from roboflow.roboflowpy import _argparser

        parser = _argparser()
        self.assertIsInstance(parser, argparse.ArgumentParser)

    def test_argparser_has_subcommands(self) -> None:
        """The parser returned by _argparser should have the new CLI subcommands."""
        from roboflow.roboflowpy import _argparser

        parser = _argparser()
        # Parse a known new-style command (--json must come before subcommand
        # when using parse_args directly; _reorder_argv handles end-position
        # in the real main() entry point)
        args = parser.parse_args(["--json", "project", "list"])
        self.assertTrue(args.json)

    def test_argparser_has_legacy_aliases(self) -> None:
        """Legacy command names should still parse."""
        from roboflow.roboflowpy import _argparser

        parser = _argparser()

        # 'login' was a top-level command in the old CLI
        args = parser.parse_args(["login"])
        self.assertIsNotNone(args.func)

        # 'whoami' was a top-level command
        args = parser.parse_args(["whoami"])
        self.assertIsNotNone(args.func)

        # 'download' was a top-level command
        args = parser.parse_args(["download", "ws/proj/1"])
        self.assertIsNotNone(args.func)


if __name__ == "__main__":
    unittest.main()
