"""Unhandled errors in the CLI entry point must not print the API key."""

import contextlib
import io
import json
import unittest
from unittest.mock import patch

import requests

from roboflow.cli import main

API_KEY = "abc-DEF_123"
CONNECTION_ERROR = requests.ConnectionError(
    f"HTTPSConnectionPool(host='api.roboflow.com', port=443): Max retries exceeded with url: /ws/proj?api_key={API_KEY}"
)


def _run_main(argv: list) -> tuple:
    stderr = io.StringIO()
    with (
        patch("sys.argv", ["roboflow", *argv]),
        patch("roboflow.adapters.rfapi.get_project", side_effect=CONNECTION_ERROR),
        contextlib.redirect_stderr(stderr),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        try:
            main()
        except SystemExit as exc:
            return exc.code, stderr.getvalue()
    return 0, stderr.getvalue()


class TestUnhandledErrorsAreRedacted(unittest.TestCase):
    def test_json_mode(self) -> None:
        code, stderr = _run_main(["--json", "-k", API_KEY, "-w", "ws", "project", "get", "proj"])

        self.assertEqual(code, 1)
        self.assertNotIn(API_KEY, stderr)
        self.assertIn("api_key=***", json.loads(stderr)["error"]["message"])

    def test_text_mode_prints_redacted_traceback(self) -> None:
        code, stderr = _run_main(["-k", API_KEY, "-w", "ws", "project", "get", "proj"])

        self.assertEqual(code, 1)
        self.assertNotIn(API_KEY, stderr)
        self.assertIn("Traceback (most recent call last)", stderr)
        self.assertIn("requests.exceptions.ConnectionError", stderr)
        self.assertIn("api_key=***", stderr)


if __name__ == "__main__":
    unittest.main()
