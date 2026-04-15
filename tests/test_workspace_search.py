import json
import unittest

import responses

from roboflow.adapters.rfapi import RoboflowError
from roboflow.config import API_URL


class TestWorkspaceSearch(unittest.TestCase):
    API_KEY = "test_key"
    WORKSPACE = "test-ws"
    SEARCH_URL = f"{API_URL}/{WORKSPACE}/search/v1?api_key={API_KEY}"

    def _make_workspace(self):
        from roboflow.core.workspace import Workspace

        info = {
            "workspace": {
                "name": "Test",
                "url": self.WORKSPACE,
                "projects": [],
                "members": [],
            }
        }
        return Workspace(info, api_key=self.API_KEY, default_workspace=self.WORKSPACE, model_format="yolov8")

    # --- search() tests ---

    @responses.activate
    def test_search_basic(self):
        body = {
            "results": [{"filename": "a.jpg"}, {"filename": "b.jpg"}],
            "total": 2,
            "continuationToken": None,
        }
        responses.add(responses.POST, self.SEARCH_URL, json=body, status=200)

        ws = self._make_workspace()
        result = ws.search("tag:review")

        self.assertEqual(result["total"], 2)
        self.assertEqual(len(result["results"]), 2)
        self.assertIsNone(result["continuationToken"])

        # Verify request payload
        sent = json.loads(responses.calls[0].request.body)
        self.assertEqual(sent["query"], "tag:review")
        self.assertEqual(sent["pageSize"], 50)
        self.assertEqual(sent["fields"], ["tags", "projects", "filename"])
        self.assertNotIn("continuationToken", sent)

    @responses.activate
    def test_search_with_continuation_token(self):
        body = {"results": [{"filename": "c.jpg"}], "total": 3, "continuationToken": None}
        responses.add(responses.POST, self.SEARCH_URL, json=body, status=200)

        ws = self._make_workspace()
        ws.search("*", continuation_token="tok_abc")

        sent = json.loads(responses.calls[0].request.body)
        self.assertEqual(sent["continuationToken"], "tok_abc")

    @responses.activate
    def test_search_custom_fields(self):
        body = {"results": [], "total": 0, "continuationToken": None}
        responses.add(responses.POST, self.SEARCH_URL, json=body, status=200)

        ws = self._make_workspace()
        ws.search("*", fields=["filename", "embedding"])

        sent = json.loads(responses.calls[0].request.body)
        self.assertEqual(sent["fields"], ["filename", "embedding"])

    @responses.activate
    def test_search_api_error(self):
        responses.add(responses.POST, self.SEARCH_URL, json={"error": "unauthorized"}, status=401)

        ws = self._make_workspace()
        with self.assertRaises(RoboflowError):
            ws.search("tag:review")

    # --- search_all() tests ---

    @responses.activate
    def test_search_all_single_page(self):
        body = {
            "results": [{"filename": "a.jpg"}, {"filename": "b.jpg"}],
            "total": 2,
            "continuationToken": None,
        }
        responses.add(responses.POST, self.SEARCH_URL, json=body, status=200)

        ws = self._make_workspace()
        pages = list(ws.search_all("*"))

        self.assertEqual(len(pages), 1)
        self.assertEqual(len(pages[0]), 2)

    @responses.activate
    def test_search_all_multiple_pages(self):
        page1 = {
            "results": [{"filename": "a.jpg"}],
            "total": 2,
            "continuationToken": "tok_page2",
        }
        page2 = {
            "results": [{"filename": "b.jpg"}],
            "total": 2,
            "continuationToken": None,
        }
        responses.add(responses.POST, self.SEARCH_URL, json=page1, status=200)
        responses.add(responses.POST, self.SEARCH_URL, json=page2, status=200)

        ws = self._make_workspace()
        pages = list(ws.search_all("*", page_size=1))

        self.assertEqual(len(pages), 2)
        self.assertEqual(pages[0][0]["filename"], "a.jpg")
        self.assertEqual(pages[1][0]["filename"], "b.jpg")

        # Verify second request used the continuation token
        sent2 = json.loads(responses.calls[1].request.body)
        self.assertEqual(sent2["continuationToken"], "tok_page2")

    @responses.activate
    def test_search_all_empty_results(self):
        body = {"results": [], "total": 0, "continuationToken": None}
        responses.add(responses.POST, self.SEARCH_URL, json=body, status=200)

        ws = self._make_workspace()
        pages = list(ws.search_all("*"))

        self.assertEqual(len(pages), 0)


if __name__ == "__main__":
    unittest.main()
