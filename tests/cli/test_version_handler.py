"""Tests for the version CLI handler."""

import argparse
import unittest


def _make_parser() -> argparse.ArgumentParser:
    """Build a minimal parser with just the version handler."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", default=False)
    parser.add_argument("--api-key", dest="api_key", default=None)
    parser.add_argument("--workspace", "-w", dest="workspace", default=None)
    subs = parser.add_subparsers(dest="command")

    from roboflow.cli.handlers.version import register

    register(subs)
    return parser


class TestVersionHandlerRegistration(unittest.TestCase):
    """Verify that the version handler registers correctly."""

    def test_register_creates_version_subcommand(self) -> None:
        parser = _make_parser()
        args = parser.parse_args(["version", "list", "-p", "my-project"])
        self.assertIsNotNone(args.func)

    def test_version_list_requires_project(self) -> None:
        parser = _make_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["version", "list"])

    def test_version_list_parses_project(self) -> None:
        parser = _make_parser()
        args = parser.parse_args(["version", "list", "-p", "my-project"])
        self.assertEqual(args.project, "my-project")

    def test_version_get_requires_version_num(self) -> None:
        parser = _make_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["version", "get"])

    def test_version_get_parses_args(self) -> None:
        parser = _make_parser()
        args = parser.parse_args(["version", "get", "3", "-p", "my-project"])
        self.assertEqual(args.version_num, "3")
        self.assertEqual(args.project, "my-project")

    def test_version_get_shorthand(self) -> None:
        parser = _make_parser()
        args = parser.parse_args(["version", "get", "my-project/3"])
        self.assertEqual(args.version_num, "my-project/3")

    def test_version_download_parses_args(self) -> None:
        parser = _make_parser()
        args = parser.parse_args(["version", "download", "ws/proj/1", "-f", "coco"])
        self.assertEqual(args.url_or_id, "ws/proj/1")
        self.assertEqual(args.format, "coco")

    def test_version_download_default_format(self) -> None:
        parser = _make_parser()
        args = parser.parse_args(["version", "download", "ws/proj/1"])
        self.assertEqual(args.format, "voc")

    def test_version_export_parses_args(self) -> None:
        parser = _make_parser()
        args = parser.parse_args(["version", "export", "2", "-p", "my-project", "-f", "yolov8"])
        self.assertEqual(args.version_num, "2")
        self.assertEqual(args.project, "my-project")
        self.assertEqual(args.format, "yolov8")

    def test_version_create_parses_args(self) -> None:
        parser = _make_parser()
        args = parser.parse_args(["version", "create", "-p", "my-project", "--settings", "config.json"])
        self.assertIsNotNone(args.func)
        self.assertEqual(args.project, "my-project")
        self.assertEqual(args.settings, "config.json")

    def test_version_create_requires_settings(self) -> None:
        parser = _make_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["version", "create", "-p", "my-project"])

    def test_subcommands_have_func(self) -> None:
        parser = _make_parser()
        subcmds = [
            "list -p proj",
            "get 3 -p proj",
            "download ws/proj/1",
            "export 1 -p proj",
            "create -p proj --settings s.json",
        ]
        for subcmd in subcmds:
            args = parser.parse_args(["version"] + subcmd.split())
            self.assertIsNotNone(args.func, f"version {subcmd} has no func")


class TestVersionCreate(unittest.TestCase):
    """Test version create handler."""

    def test_create_missing_settings_file(self) -> None:
        from unittest.mock import patch

        parser = _make_parser()
        args = parser.parse_args(
            ["--json", "version", "create", "-p", "my-ws/my-project", "--settings", "/nonexistent/file.json"]
        )
        args.api_key = "fake-key"
        with patch("roboflow.config.load_roboflow_api_key", return_value="fake-key"):
            with self.assertRaises(SystemExit) as ctx:
                args.func(args)
            self.assertEqual(ctx.exception.code, 1)

    def test_create_invalid_json_file(self) -> None:
        import tempfile
        from unittest.mock import patch

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json")
            f.flush()
            parser = _make_parser()
            args = parser.parse_args(
                ["--json", "version", "create", "-p", "my-ws/my-project", "--settings", f.name]
            )
            args.api_key = "fake-key"
            with patch("roboflow.config.load_roboflow_api_key", return_value="fake-key"):
                with self.assertRaises(SystemExit) as ctx:
                    args.func(args)
                self.assertEqual(ctx.exception.code, 1)

    def test_create_no_api_key(self) -> None:
        import json
        import tempfile

        settings = {"augmentation": {}, "preprocessing": {}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(settings, f)
            f.flush()
            parser = _make_parser()
            args = parser.parse_args(
                ["--json", "version", "create", "-p", "my-ws/my-project", "--settings", f.name]
            )
            # Patch load_roboflow_api_key to return None
            from unittest.mock import patch

            with patch("roboflow.config.load_roboflow_api_key", return_value=None):
                with self.assertRaises(SystemExit) as ctx:
                    args.func(args)
                self.assertEqual(ctx.exception.code, 2)

    def test_create_json_error_output(self) -> None:
        import io
        import sys

        parser = _make_parser()
        args = parser.parse_args(
            ["--json", "version", "create", "-p", "my-ws/my-project", "--settings", "/nonexistent/file.json"]
        )
        captured = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = captured
        try:
            with self.assertRaises(SystemExit):
                args.func(args)
        finally:
            sys.stderr = old_stderr
        import json

        err = json.loads(captured.getvalue())
        self.assertIn("error", err)
        self.assertIn("message", err["error"])


class TestParseUrl(unittest.TestCase):
    """Test the _parse_url helper."""

    def test_shorthand(self) -> None:
        from roboflow.cli.handlers.version import _parse_url

        w, p, v = _parse_url("my-ws/my-project/3")
        self.assertEqual(w, "my-ws")
        self.assertEqual(p, "my-project")
        self.assertEqual(v, "3")

    def test_full_url(self) -> None:
        from roboflow.cli.handlers.version import _parse_url

        w, p, v = _parse_url("https://universe.roboflow.com/my-ws/my-project/3")
        self.assertEqual(w, "my-ws")
        self.assertEqual(p, "my-project")
        self.assertEqual(v, "3")

    def test_no_version(self) -> None:
        from roboflow.cli.handlers.version import _parse_url

        w, p, v = _parse_url("my-ws/my-project")
        self.assertEqual(w, "my-ws")
        self.assertEqual(p, "my-project")
        self.assertIsNone(v)


if __name__ == "__main__":
    unittest.main()
