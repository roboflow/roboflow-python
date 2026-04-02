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


class TestReorderArgv(unittest.TestCase):
    """Verify _reorder_argv moves global flags before subcommands."""

    def _reorder(self, argv: list[str]) -> list[str]:
        from roboflow.cli import _reorder_argv

        return _reorder_argv(argv)

    def test_no_flags(self) -> None:
        self.assertEqual(self._reorder(["project", "list"]), ["project", "list"])

    def test_empty(self) -> None:
        self.assertEqual(self._reorder([]), [])

    def test_bool_flag_after_subcommand(self) -> None:
        result = self._reorder(["project", "list", "--json"])
        self.assertEqual(result, ["--json", "project", "list"])

    def test_bool_flag_already_first(self) -> None:
        result = self._reorder(["--json", "project", "list"])
        self.assertEqual(result, ["--json", "project", "list"])

    def test_short_bool_flag(self) -> None:
        result = self._reorder(["project", "list", "-j"])
        self.assertEqual(result, ["-j", "project", "list"])

    def test_value_flag_after_subcommand(self) -> None:
        result = self._reorder(["project", "list", "--api-key", "abc123"])
        self.assertEqual(result, ["--api-key", "abc123", "project", "list"])

    def test_short_value_flag(self) -> None:
        result = self._reorder(["project", "list", "-k", "abc123"])
        self.assertEqual(result, ["-k", "abc123", "project", "list"])

    def test_multiple_flags_mixed(self) -> None:
        result = self._reorder(["project", "list", "--json", "-w", "my-ws"])
        self.assertEqual(result, ["--json", "-w", "my-ws", "project", "list"])

    def test_value_flag_at_end_without_value(self) -> None:
        """A value flag at the very end with no following arg should still be moved."""
        result = self._reorder(["project", "list", "--api-key"])
        self.assertEqual(result, ["--api-key", "project", "list"])

    def test_non_global_flags_preserved(self) -> None:
        """Flags not in the global set stay in place."""
        result = self._reorder(["image", "upload", "--project", "my-proj", "--json"])
        self.assertEqual(result, ["--json", "image", "upload", "--project", "my-proj"])

    def test_quiet_and_version_flags(self) -> None:
        result = self._reorder(["project", "list", "--quiet", "--version"])
        self.assertEqual(result, ["--quiet", "--version", "project", "list"])

    def test_workspace_flag(self) -> None:
        result = self._reorder(["project", "list", "--workspace", "ws-1"])
        self.assertEqual(result, ["--workspace", "ws-1", "project", "list"])

    def test_preserves_subcommand_positional_args(self) -> None:
        result = self._reorder(["version", "download", "ws/proj/3", "--json", "-f", "yolov8"])
        self.assertEqual(result, ["--json", "version", "download", "ws/proj/3", "-f", "yolov8"])


class TestAliases(unittest.TestCase):
    """Verify top-level aliases parse correctly and delegate to the right handler."""

    def _parse(self, argv: list[str]):
        from roboflow.cli import build_parser

        parser = build_parser()
        return parser.parse_args(argv)

    def test_login_alias_exists(self) -> None:
        args = self._parse(["login"])
        self.assertIsNotNone(args.func)

    def test_whoami_alias_exists(self) -> None:
        args = self._parse(["whoami"])
        self.assertIsNotNone(args.func)

    def test_upload_alias_exists(self) -> None:
        args = self._parse(["upload", "img.jpg", "-p", "my-project"])
        self.assertIsNotNone(args.func)
        self.assertEqual(args.path, "img.jpg")
        self.assertEqual(args.project, "my-project")

    def test_import_alias_exists(self) -> None:
        args = self._parse(["import", "/data/images", "-p", "my-project"])
        self.assertIsNotNone(args.func)
        self.assertEqual(args.path, "/data/images")
        self.assertEqual(args.project, "my-project")

    def test_download_alias_parses_url(self) -> None:
        """Regression: download alias must use url_or_id as dest, not datasetUrl."""
        args = self._parse(["download", "my-ws/my-proj/3"])
        self.assertIsNotNone(args.func)
        self.assertEqual(args.url_or_id, "my-ws/my-proj/3")

    def test_download_alias_delegates_to_version_download(self) -> None:
        """The download alias should use the same handler as 'version download'."""
        from roboflow.cli.handlers.version import _download

        args = self._parse(["download", "my-ws/my-proj/3"])
        self.assertIs(args.func, _download)


if __name__ == "__main__":
    unittest.main()
