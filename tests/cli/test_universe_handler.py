"""Tests for the universe CLI handler."""

import io
import json
import sys
import unittest
from unittest.mock import patch

from typer.testing import CliRunner

from roboflow.cli import app

runner = CliRunner()


class TestUniverseRegistration(unittest.TestCase):
    """Verify universe handler registers expected subcommands."""

    def test_universe_app_exists(self) -> None:
        from roboflow.cli.handlers.universe import universe_app

        self.assertIsNotNone(universe_app)

    def test_universe_search_exists(self) -> None:
        result = runner.invoke(app, ["universe", "search", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_universe_search_help_shows_options(self) -> None:
        result = runner.invoke(app, ["universe", "search", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("type", result.output.lower())
        self.assertIn("limit", result.output.lower())


class TestUniverseSearch(unittest.TestCase):
    """Test universe search handler."""

    @patch("roboflow.adapters.rfapi.search_universe")
    @patch("roboflow.config.load_roboflow_api_key", return_value="test-key")
    def test_search_success(self, _mock_key, mock_search) -> None:
        mock_search.return_value = {
            "results": [
                {"name": "cats-dataset", "type": "dataset", "images": 1000, "url": "https://example.com/cats"},
            ]
        }
        result = runner.invoke(app, ["universe", "search", "cats"])
        self.assertIn("cats-dataset", result.output)

    @patch("roboflow.adapters.rfapi.search_universe")
    @patch("roboflow.config.load_roboflow_api_key", return_value="my-key")
    def test_search_passes_api_key(self, _mock_key, mock_search) -> None:
        mock_search.return_value = {"results": []}
        runner.invoke(app, ["universe", "search", "cats"])
        mock_search.assert_called_once_with("cats", api_key="my-key", project_type=None, limit=12)

    @patch("roboflow.adapters.rfapi.search_universe")
    @patch("roboflow.config.load_roboflow_api_key", return_value="k")
    def test_search_passes_custom_limit(self, _mock_key, mock_search) -> None:
        mock_search.return_value = {"results": []}
        runner.invoke(app, ["universe", "search", "dogs", "--limit", "5"])
        mock_search.assert_called_once_with("dogs", api_key="k", project_type=None, limit=5)

    @patch("roboflow.adapters.rfapi.search_universe")
    @patch("roboflow.config.load_roboflow_api_key", return_value="test-key")
    def test_search_json_output(self, _mock_key, mock_search) -> None:
        mock_search.return_value = {
            "results": [
                {"name": "dogs-dataset", "type": "dataset", "images": 500, "url": "https://example.com/dogs"},
            ]
        }
        result = runner.invoke(app, ["--json", "universe", "search", "dogs"])
        data = json.loads(result.output)
        self.assertIsInstance(data, list)
        self.assertEqual(data[0]["name"], "dogs-dataset")

    @patch("roboflow.adapters.rfapi.search_universe")
    @patch("roboflow.config.load_roboflow_api_key", return_value="test-key")
    def test_search_api_error_json(self, _mock_key, mock_search) -> None:
        from roboflow.adapters.rfapi import RoboflowError

        mock_search.side_effect = RoboflowError("API down")
        result = runner.invoke(app, ["--json", "universe", "search", "fail"])
        self.assertNotEqual(result.exit_code, 0)


if __name__ == "__main__":
    unittest.main()
