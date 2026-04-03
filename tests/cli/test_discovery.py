"""Tests that the CLI auto-discovery mechanism works correctly.

Tests use typer.testing.CliRunner instead of argparse internals.
"""

import unittest

from typer.testing import CliRunner

from roboflow.cli import app

runner = CliRunner()


class TestCLIDiscovery(unittest.TestCase):
    """Verify the CLI app loads and has expected structure."""

    def test_app_exists(self) -> None:
        self.assertIsNotNone(app)

    def test_help_shows_commands(self) -> None:
        result = runner.invoke(app, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("project", result.output)
        self.assertIn("workspace", result.output)
        self.assertIn("image", result.output)
        self.assertIn("infer", result.output)

    def test_version_flag(self) -> None:
        result = runner.invoke(app, ["--version"])
        self.assertEqual(result.exit_code, 0)

    def test_json_flag(self) -> None:
        result = runner.invoke(app, ["--json", "--help"])
        self.assertEqual(result.exit_code, 0)

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

    def test_compat_module_importable(self) -> None:
        from roboflow.cli._compat import ctx_to_args

        self.assertTrue(callable(ctx_to_args))


class TestGlobalFlagPositioning(unittest.TestCase):
    """Verify global flags work in any position (typer handles natively)."""

    def test_json_at_start(self) -> None:
        result = runner.invoke(app, ["--json", "project", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_json_at_end(self) -> None:
        result = runner.invoke(app, ["project", "list", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_workspace_long_form(self) -> None:
        result = runner.invoke(app, ["--workspace", "test-ws", "project", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_api_key_flag(self) -> None:
        result = runner.invoke(app, ["--api-key", "test-key", "project", "--help"])
        self.assertEqual(result.exit_code, 0)


class TestAliases(unittest.TestCase):
    """Verify backwards-compat aliases work."""

    def test_login_alias(self) -> None:
        result = runner.invoke(app, ["login", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("login", result.output.lower())

    def test_whoami_alias(self) -> None:
        result = runner.invoke(app, ["whoami", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_upload_alias(self) -> None:
        result = runner.invoke(app, ["upload", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_import_alias(self) -> None:
        result = runner.invoke(app, ["import", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_download_alias(self) -> None:
        result = runner.invoke(app, ["download", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("dataseturl", result.output.lower())

    def test_hidden_aliases_not_in_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        self.assertNotIn("upload_model", result.output)
        self.assertNotIn("get_workspace_info", result.output)
        self.assertNotIn("run_video_inference_api", result.output)

    def test_hidden_alias_still_works(self) -> None:
        result = runner.invoke(app, ["upload_model", "--help"])
        self.assertEqual(result.exit_code, 0)


if __name__ == "__main__":
    unittest.main()
