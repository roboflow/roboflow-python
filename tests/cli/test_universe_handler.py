"""Tests for the universe CLI handler."""

import unittest


class TestUniverseRegistration(unittest.TestCase):
    """Verify universe handler registers expected subcommands."""

    def test_register_callable(self) -> None:
        from roboflow.cli.handlers.universe import register

        self.assertTrue(callable(register))

    def test_universe_search_exists(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["universe", "search", "cats"])
        self.assertIsNotNone(args.func)
        self.assertEqual(args.query, "cats")

    def test_universe_search_with_flags(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["universe", "search", "dogs", "--type", "model", "--limit", "5"])
        self.assertEqual(args.type, "model")
        self.assertEqual(args.limit, 5)

    def test_universe_search_default_limit(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["universe", "search", "cats"])
        self.assertEqual(args.limit, 12)


class TestUniverseSearch(unittest.TestCase):
    """Test universe search handler."""

    def test_search_success(self) -> None:
        import io
        import sys

        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["universe", "search", "cats"])
        from unittest.mock import patch

        mock_data = {
            "results": [
                {"name": "cats-dataset", "type": "dataset", "images": 1000, "url": "https://example.com/cats"},
            ]
        }
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            with patch("roboflow.adapters.rfapi.search_universe", return_value=mock_data):
                with patch("roboflow.config.load_roboflow_api_key", return_value="test-key"):
                    args.func(args)
        finally:
            sys.stdout = old_stdout
        out = captured.getvalue()
        self.assertIn("cats-dataset", out)

    def test_search_passes_api_key(self) -> None:
        import io
        import sys

        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["universe", "search", "cats"])
        from unittest.mock import call, patch

        mock_data = {"results": []}
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            with patch("roboflow.adapters.rfapi.search_universe", return_value=mock_data) as mock_search:
                with patch("roboflow.config.load_roboflow_api_key", return_value="my-key"):
                    args.func(args)
        finally:
            sys.stdout = old_stdout
        mock_search.assert_called_once_with("cats", api_key="my-key", project_type=None, limit=12)

    def test_search_json_output(self) -> None:
        import io
        import json
        import sys

        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["--json", "universe", "search", "dogs"])
        from unittest.mock import patch

        mock_data = {
            "results": [
                {"name": "dogs-dataset", "type": "dataset", "images": 500, "url": "https://example.com/dogs"},
            ]
        }
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            with patch("roboflow.adapters.rfapi.search_universe", return_value=mock_data):
                args.func(args)
        finally:
            sys.stdout = old_stdout
        result = json.loads(captured.getvalue())
        self.assertIsInstance(result, list)
        self.assertEqual(result[0]["name"], "dogs-dataset")

    def test_search_api_error_json(self) -> None:
        import io
        import sys

        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["--json", "universe", "search", "fail"])
        from unittest.mock import patch

        from roboflow.adapters.rfapi import RoboflowError

        captured = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = captured
        try:
            with patch("roboflow.adapters.rfapi.search_universe", side_effect=RoboflowError("API down")):
                with self.assertRaises(SystemExit) as ctx:
                    args.func(args)
                self.assertEqual(ctx.exception.code, 1)
        finally:
            sys.stderr = old_stderr
        import json

        err = json.loads(captured.getvalue())
        self.assertIn("error", err)


if __name__ == "__main__":
    unittest.main()
