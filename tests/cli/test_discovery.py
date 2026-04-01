"""Tests that the CLI auto-discovery mechanism works correctly."""

import unittest


class TestCLIDiscovery(unittest.TestCase):
    """Verify build_parser discovers handlers and creates expected subcommands."""

    def test_build_parser_returns_parser(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        self.assertIsNotNone(parser)

    def test_parser_has_global_flags(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        # Parse with no args should work (defaults to help / version)
        args = parser.parse_args(["--json"])
        self.assertTrue(args.json)

    def test_version_flag(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["--version"])
        self.assertTrue(args.version)

    def test_handlers_package_importable(self) -> None:
        import roboflow.cli.handlers

        self.assertIsNotNone(roboflow.cli.handlers)

    def test_output_module_importable(self) -> None:
        from roboflow.cli._output import output, output_error

        self.assertTrue(callable(output))
        self.assertTrue(callable(output_error))

    def test_resolver_module_importable(self) -> None:
        from roboflow.cli._resolver import resolve_resource

        self.assertTrue(callable(resolve_resource))

    def test_table_module_importable(self) -> None:
        from roboflow.cli._table import format_table

        self.assertTrue(callable(format_table))


if __name__ == "__main__":
    unittest.main()
