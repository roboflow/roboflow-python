"""CLI `--media-types` selection on `roboflow search` and `roboflow image search`."""

import json
import re
import unittest
from unittest.mock import patch

import responses
from typer.testing import CliRunner

from roboflow.cli import app
from roboflow.config import API_URL

runner = CliRunner()

API_KEY = "cli_test_key"
WORKSPACE = "cli-test-ws"
PROJECT = "cli-test-project"
SEARCH_URL = f"{API_URL}/{WORKSPACE}/search/v1?api_key={API_KEY}"
WORKSPACE_URL = f"{API_URL}/{WORKSPACE}?api_key={API_KEY}"


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """Rich styles each `-` of an option name separately, so strip ANSI before matching."""
    return _ANSI.sub("", text)


def _search_bodies():
    return [json.loads(c.request.body) for c in responses.calls if c.request.url == SEARCH_URL and c.request.body]


def _stub_workspace_init():
    """`roboflow search` authenticates and builds a Workspace first; `image search -p` does not."""
    responses.add(
        responses.POST,
        f"{API_URL}/?api_key={API_KEY}",
        json={"welcome": "Welcome to the Roboflow API.", "workspace": WORKSPACE},
        status=200,
    )
    responses.add(
        responses.GET,
        WORKSPACE_URL,
        json={"workspace": {"name": "CLI Test", "url": WORKSPACE, "projects": [], "members": []}},
        status=200,
    )


def _stub_search(results, total=None, token=None):
    body = {"results": results, "total": total if total is not None else len(results), "continuationToken": token}
    responses.add(responses.POST, SEARCH_URL, json=body, status=200)


class TestSearchHelpAdvertisesMediaTypes(unittest.TestCase):
    def test_top_level_search_help(self):
        result = runner.invoke(app, ["search", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--media-types", _plain(result.output))

    def test_image_search_help(self):
        result = runner.invoke(app, ["image", "search", "--help"])
        self.assertEqual(result.exit_code, 0)
        plain = _plain(result.output)
        self.assertIn("--media-types", plain)
        self.assertIn("--fields", plain)


class TestTopLevelSearchMediaTypes(unittest.TestCase):
    """`roboflow search` routes the selection through Workspace.search."""

    @responses.activate
    def test_video_selection_sends_media_types(self):
        _stub_workspace_init()
        _stub_search([{"id": "vid1", "filename": "clip.mp4", "mediaType": "video", "videoUrl": "https://s/clip.mp4"}])

        result = runner.invoke(
            app,
            [
                "--json",
                "--api-key",
                API_KEY,
                "--workspace",
                WORKSPACE,
                "search",
                "*",
                "--media-types",
                "video",
                "--fields",
                "id,filename,url",
            ],
        )

        self.assertEqual(result.exit_code, 0, result.output)
        body = _search_bodies()[0]
        self.assertEqual(body["mediaTypes"], ["video"])
        self.assertEqual(body["fields"], ["id", "filename", "url"])

        payload = json.loads(result.output)
        self.assertEqual(payload["mediaTypes"], ["video"])
        self.assertEqual(payload["results"][0]["videoUrl"], "https://s/clip.mp4")

    @responses.activate
    def test_mixed_selection_sends_both(self):
        _stub_workspace_init()
        _stub_search([])

        result = runner.invoke(
            app,
            ["--json", "--api-key", API_KEY, "--workspace", WORKSPACE, "search", "*", "--media-types", "image,video"],
        )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(_search_bodies()[0]["mediaTypes"], ["image", "video"])

    @responses.activate
    def test_omitted_selection_keeps_image_default(self):
        _stub_workspace_init()
        _stub_search([{"id": "img1", "filename": "a.jpg"}])

        result = runner.invoke(app, ["--json", "--api-key", API_KEY, "--workspace", WORKSPACE, "search", "tag:review"])

        self.assertEqual(result.exit_code, 0, result.output)
        body = _search_bodies()[0]
        self.assertNotIn("mediaTypes", body)
        self.assertNotIn("mediaTypes", json.loads(result.output))

    @responses.activate
    def test_text_output_shows_media_type_and_video_url(self):
        _stub_workspace_init()
        _stub_search([{"id": "vid1", "filename": "clip.mp4", "mediaType": "video", "videoUrl": "https://s/clip.mp4"}])

        result = runner.invoke(
            app, ["--api-key", API_KEY, "--workspace", WORKSPACE, "search", "*", "--media-types", "video"]
        )

        self.assertEqual(result.exit_code, 0, result.output)
        plain = _plain(result.output)
        self.assertIn("clip.mp4", plain)
        self.assertIn("[video]", plain)
        self.assertIn("https://s/clip.mp4", plain)

    @responses.activate
    def test_invalid_selection_is_a_structured_error_with_no_request(self):
        _stub_workspace_init()

        result = runner.invoke(
            app, ["--json", "--api-key", API_KEY, "--workspace", WORKSPACE, "search", "*", "--media-types", "audio"]
        )

        self.assertEqual(result.exit_code, 1)
        self.assertEqual(_search_bodies(), [])
        error = json.loads(result.output)["error"]
        self.assertIn("audio", error["message"])
        self.assertIn("image, video", error["hint"])

    @responses.activate
    def test_export_rejects_media_types(self):
        _stub_workspace_init()

        result = runner.invoke(
            app,
            [
                "--json",
                "--api-key",
                API_KEY,
                "--workspace",
                WORKSPACE,
                "search",
                "*",
                "--export",
                "--media-types",
                "video",
            ],
        )

        self.assertEqual(result.exit_code, 1)
        error = json.loads(result.output)["error"]
        self.assertIn("--media-types is not supported with --export", error["message"])


class TestImageSearchMediaTypes(unittest.TestCase):
    """`roboflow image search` covers both the workspace and project-scoped paths."""

    @responses.activate
    def test_workspace_path_sends_media_types(self):
        _stub_workspace_init()
        _stub_search([{"id": "vid1", "filename": "clip.mp4", "mediaType": "video"}])

        result = runner.invoke(
            app,
            [
                "--json",
                "--api-key",
                API_KEY,
                "--workspace",
                WORKSPACE,
                "image",
                "search",
                "*",
                "--media-types",
                "video",
            ],
        )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(_search_bodies()[0]["mediaTypes"], ["video"])

    @responses.activate
    @patch("roboflow.cli.handlers.image._default_workspace", return_value=WORKSPACE)
    def test_project_path_sends_media_types_with_project_filter(self, _mock_ws):
        _stub_search([{"id": "vid1", "filename": "clip.mp4", "mediaType": "video"}])

        result = runner.invoke(
            app,
            [
                "--json",
                "--api-key",
                API_KEY,
                "--workspace",
                WORKSPACE,
                "image",
                "search",
                "*",
                "-p",
                PROJECT,
                "--media-types",
                "image,video",
                "--fields",
                "id,url",
            ],
        )

        self.assertEqual(result.exit_code, 0, result.output)
        body = _search_bodies()[0]
        self.assertEqual(body["mediaTypes"], ["image", "video"])
        self.assertEqual(body["fields"], ["id", "url"])
        self.assertEqual(body["query"], f"project:{PROJECT} *")

    @responses.activate
    @patch("roboflow.cli.handlers.image._default_workspace", return_value=WORKSPACE)
    def test_project_path_omitted_selection_keeps_image_default(self, _mock_ws):
        _stub_search([{"id": "img1"}])

        result = runner.invoke(
            app,
            ["--json", "--api-key", API_KEY, "--workspace", WORKSPACE, "image", "search", "*", "-p", PROJECT],
        )

        self.assertEqual(result.exit_code, 0, result.output)
        body = _search_bodies()[0]
        self.assertNotIn("mediaTypes", body)
        self.assertNotIn("fields", body)

    @responses.activate
    @patch("roboflow.cli.handlers.image._default_workspace", return_value=WORKSPACE)
    def test_project_path_invalid_selection_is_a_structured_error(self, _mock_ws):
        result = runner.invoke(
            app,
            [
                "--json",
                "--api-key",
                API_KEY,
                "--workspace",
                WORKSPACE,
                "image",
                "search",
                "*",
                "-p",
                PROJECT,
                "--media-types",
                "clip",
            ],
        )

        self.assertEqual(result.exit_code, 1)
        self.assertEqual(_search_bodies(), [])
        self.assertIn("clip", json.loads(result.output)["error"]["message"])

    @responses.activate
    @patch("roboflow.cli.handlers.image._default_workspace", return_value=WORKSPACE)
    def test_project_path_forwards_cursor(self, _mock_ws):
        _stub_search([{"id": "vid1"}], total=2, token="tok2")

        result = runner.invoke(
            app,
            [
                "--json",
                "--api-key",
                API_KEY,
                "--workspace",
                WORKSPACE,
                "image",
                "search",
                "*",
                "-p",
                PROJECT,
                "--media-types",
                "video",
                "--cursor",
                "tok1",
            ],
        )

        self.assertEqual(result.exit_code, 0, result.output)
        body = _search_bodies()[0]
        self.assertEqual(body["continuationToken"], "tok1")
        self.assertEqual(body["mediaTypes"], ["video"])


if __name__ == "__main__":
    unittest.main()
