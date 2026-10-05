"""Explicit media type selection on the project and workspace search surfaces."""

import json
import unittest

import responses

from roboflow.adapters import rfapi
from roboflow.config import API_URL
from roboflow.util.search_utils import normalize_media_types, parse_media_types_option
from tests import PROJECT_NAME, ROBOFLOW_API_KEY, WORKSPACE_NAME, RoboflowTest


class TestNormalizeMediaTypes(unittest.TestCase):
    """The shared validator mirrors the API contract: non-empty list of image/video."""

    def test_none_passes_through(self):
        self.assertIsNone(normalize_media_types(None))

    def test_lowercases_and_dedupes_preserving_order(self):
        self.assertEqual(normalize_media_types(["VIDEO", "Image", "video"]), ["video", "image"])

    def test_accepts_tuple(self):
        self.assertEqual(normalize_media_types(("video",)), ["video"])

    def test_bare_string_names_the_fix(self):
        with self.assertRaises(ValueError) as ctx:
            normalize_media_types("video")
        self.assertIn("must be a list", str(ctx.exception))
        self.assertIn("media_types=['video']", str(ctx.exception))

    def test_empty_list_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            normalize_media_types([])
        self.assertIn("non-empty list", str(ctx.exception))

    def test_unknown_media_type_rejected_with_valid_set(self):
        with self.assertRaises(ValueError) as ctx:
            normalize_media_types(["audio"])
        self.assertIn("'audio'", str(ctx.exception))
        self.assertIn("'image', 'video'", str(ctx.exception))

    def test_non_string_entry_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            normalize_media_types([1])
        self.assertIn("must be strings", str(ctx.exception))


class TestParseMediaTypesOption(unittest.TestCase):
    """The CLI option parser turns a comma-separated value into a normalized list."""

    def test_none_passes_through(self):
        self.assertIsNone(parse_media_types_option(None))

    def test_parses_and_strips(self):
        self.assertEqual(parse_media_types_option("image, video"), ["image", "video"])

    def test_single_value(self):
        self.assertEqual(parse_media_types_option("video"), ["video"])

    def test_blank_value_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            parse_media_types_option(" , ")
        self.assertIn("at least one of", str(ctx.exception))

    def test_unknown_value_rejected(self):
        with self.assertRaises(ValueError):
            parse_media_types_option("image,audio")


class TestProjectSearchMediaTypes(RoboflowTest):
    """Project.search/search_all send `mediaTypes` only when explicitly selected."""

    SEARCH_URL = f"{API_URL}/{WORKSPACE_NAME}/{PROJECT_NAME}/search?api_key={ROBOFLOW_API_KEY}"

    def _stub(self, results):
        responses.add(responses.POST, self.SEARCH_URL, json={"results": results}, status=200)

    def _search_bodies(self):
        """Only the /search POSTs — RoboflowTest.setUp mocks workspace/project init calls too."""
        return [
            json.loads(call.request.body)
            for call in responses.calls
            if call.request.url == self.SEARCH_URL and call.request.body
        ]

    def _sent(self, index=0):
        return self._search_bodies()[index]

    @responses.activate
    def test_omitted_media_types_keeps_image_default_off_the_wire(self):
        self._stub([{"id": "img1"}])

        self.project.search(prompt="cat")

        self.assertNotIn("mediaTypes", self._sent())

    @responses.activate
    def test_video_selection_sends_media_types(self):
        self._stub([{"id": "vid1", "mediaType": "video", "videoUrl": "https://signed/vid1.mp4"}])

        results = self.project.search(media_types=["video"], fields=["id", "url"])

        self.assertEqual(self._sent()["mediaTypes"], ["video"])
        self.assertEqual(results[0]["mediaType"], "video")

    @responses.activate
    def test_mixed_selection_sends_both(self):
        self._stub([{"id": "img1"}, {"id": "vid1"}])

        self.project.search(media_types=["image", "video"])

        self.assertEqual(self._sent()["mediaTypes"], ["image", "video"])

    @responses.activate
    def test_selection_is_normalized_before_sending(self):
        self._stub([])

        self.project.search(media_types=["VIDEO", "video"])

        self.assertEqual(self._sent()["mediaTypes"], ["video"])

    @responses.activate
    def test_invalid_selection_raises_before_any_request(self):
        with self.assertRaises(ValueError):
            self.project.search(media_types=["audio"])

        self.assertEqual(self._search_bodies(), [])

    @responses.activate
    def test_search_all_forwards_media_types_on_every_page(self):
        page_one = [{"id": f"vid{i}"} for i in range(2)]
        responses.add(responses.POST, self.SEARCH_URL, json={"results": page_one}, status=200)
        responses.add(responses.POST, self.SEARCH_URL, json={"results": [{"id": "vid2"}]}, status=200)

        pages = list(self.project.search_all(limit=2, media_types=["video"]))

        self.assertEqual(len(pages), 2)
        self.assertEqual(len(self._search_bodies()), 2)
        for index in range(2):
            self.assertEqual(self._sent(index)["mediaTypes"], ["video"])
        # Offset still advances by `limit` while the selection is carried along.
        self.assertEqual(self._sent(0)["offset"], 0)
        self.assertEqual(self._sent(1)["offset"], 2)

    @responses.activate
    def test_search_all_without_selection_stays_image_default(self):
        responses.add(responses.POST, self.SEARCH_URL, json={"results": [{"id": "img0"}]}, status=200)

        list(self.project.search_all(limit=2))

        self.assertNotIn("mediaTypes", self._sent())


class TestWorkspaceSearchMediaTypes(unittest.TestCase):
    """Workspace.search/search_all and the adapter forward `mediaTypes` to search/v1."""

    API_KEY = "test_key"
    WORKSPACE = "test-ws"
    SEARCH_URL = f"{API_URL}/{WORKSPACE}/search/v1?api_key={API_KEY}"

    def _make_workspace(self):
        from roboflow.core.workspace import Workspace

        info = {"workspace": {"name": "Test", "url": self.WORKSPACE, "projects": [], "members": []}}
        return Workspace(info, api_key=self.API_KEY, default_workspace=self.WORKSPACE, model_format="yolov8")

    @staticmethod
    def _sent(index=0):
        return json.loads(responses.calls[index].request.body)

    @responses.activate
    def test_omitted_media_types_keeps_image_default_off_the_wire(self):
        responses.add(responses.POST, self.SEARCH_URL, json={"results": [], "total": 0}, status=200)

        self._make_workspace().search("tag:review")

        self.assertNotIn("mediaTypes", self._sent())

    @responses.activate
    def test_video_selection_sends_media_types(self):
        body = {
            "results": [{"id": "vid1", "mediaType": "video", "videoUrl": "https://signed/vid1.mp4"}],
            "total": 1,
            "continuationToken": None,
        }
        responses.add(responses.POST, self.SEARCH_URL, json=body, status=200)

        page = self._make_workspace().search("*", media_types=["video"], fields=["id", "url"])

        self.assertEqual(self._sent()["mediaTypes"], ["video"])
        self.assertEqual(page["results"][0]["videoUrl"], "https://signed/vid1.mp4")

    @responses.activate
    def test_mixed_selection_sends_both(self):
        responses.add(responses.POST, self.SEARCH_URL, json={"results": [], "total": 0}, status=200)

        self._make_workspace().search("*", media_types=["image", "video"])

        self.assertEqual(self._sent()["mediaTypes"], ["image", "video"])

    @responses.activate
    def test_invalid_selection_raises_before_any_request(self):
        with self.assertRaises(ValueError):
            self._make_workspace().search("*", media_types=[])

        self.assertEqual(len(responses.calls), 0)

    @responses.activate
    def test_search_all_forwards_media_types_across_continuation_pages(self):
        responses.add(
            responses.POST,
            self.SEARCH_URL,
            json={"results": [{"id": "vid0"}], "total": 2, "continuationToken": "tok1"},
            status=200,
        )
        responses.add(
            responses.POST,
            self.SEARCH_URL,
            json={"results": [{"id": "vid1"}], "total": 2, "continuationToken": None},
            status=200,
        )

        pages = list(self._make_workspace().search_all("*", media_types=["video"]))

        self.assertEqual(len(pages), 2)
        self.assertEqual(len(responses.calls), 2)
        self.assertEqual(self._sent(0)["mediaTypes"], ["video"])
        self.assertEqual(self._sent(1)["mediaTypes"], ["video"])
        self.assertEqual(self._sent(1)["continuationToken"], "tok1")

    @responses.activate
    def test_adapter_sends_media_types(self):
        responses.add(responses.POST, self.SEARCH_URL, json={"results": [], "total": 0}, status=200)

        rfapi.workspace_search(
            api_key=self.API_KEY,
            workspace_url=self.WORKSPACE,
            query="*",
            media_types=["Video"],
        )

        self.assertEqual(self._sent()["mediaTypes"], ["video"])

    @responses.activate
    def test_adapter_without_selection_omits_media_types(self):
        responses.add(responses.POST, self.SEARCH_URL, json={"results": [], "total": 0}, status=200)

        rfapi.workspace_search(api_key=self.API_KEY, workspace_url=self.WORKSPACE, query="*")

        self.assertNotIn("mediaTypes", self._sent())


if __name__ == "__main__":
    unittest.main()
