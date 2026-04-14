import json
import os
import tempfile
import unittest

import responses

from roboflow.adapters.rfapi import RoboflowError
from roboflow.config import API_URL

# The vision events API does not include workspace in the URL.
# Auth is via Bearer token; workspace is derived server-side from the API key.
_BASE = f"{API_URL}/vision-events"


class TestVisionEvents(unittest.TestCase):
    API_KEY = "test_key"
    WORKSPACE = "test-ws"

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

    def _assert_bearer_auth(self, call_index=0):
        auth = responses.calls[call_index].request.headers.get("Authorization")
        self.assertEqual(auth, f"Bearer {self.API_KEY}")

    # --- write_vision_event ---

    @responses.activate
    def test_write_event(self):
        responses.add(responses.POST, _BASE, json={"eventId": "evt-001"}, status=201)

        ws = self._make_workspace()
        event = {
            "eventId": "evt-001",
            "eventType": "quality_check",
            "useCaseId": "uc-1",
            "timestamp": "2024-01-15T10:00:00Z",
            "eventData": {"result": "pass"},
        }
        result = ws.write_vision_event(event)

        self.assertEqual(result["eventId"], "evt-001")
        self._assert_bearer_auth()
        sent = json.loads(responses.calls[0].request.body)
        self.assertEqual(sent["eventId"], "evt-001")
        self.assertEqual(sent["eventType"], "quality_check")
        self.assertEqual(sent["useCaseId"], "uc-1")
        self.assertEqual(sent["eventData"], {"result": "pass"})

    @responses.activate
    def test_write_event_passthrough(self):
        """The event dict must be sent to the server unchanged (no filtering or transformation)."""
        responses.add(responses.POST, _BASE, json={"eventId": "e1"}, status=201)

        ws = self._make_workspace()
        event = {
            "eventId": "e1",
            "eventType": "safety_alert",
            "useCaseId": "warehouse-safety",
            "timestamp": "2024-06-01T12:00:00Z",
            "deviceId": "cam-5",
            "streamId": "stream-a",
            "workflowId": "wf-1",
            "images": [{"sourceId": "src-1", "label": "frame"}],
            "eventData": {"alertType": "fire", "severity": "high"},
            "customMetadata": {"zone": "B3", "temperature": 42.5, "active": True},
        }
        ws.write_vision_event(event)

        sent = json.loads(responses.calls[0].request.body)
        self.assertEqual(sent, event)

    @responses.activate
    def test_write_event_error(self):
        responses.add(responses.POST, _BASE, json={"error": "forbidden"}, status=403)

        ws = self._make_workspace()
        with self.assertRaises(RoboflowError):
            ws.write_vision_event({"eventId": "x", "eventType": "custom", "useCaseId": "s", "timestamp": "t"})

    # --- write_vision_events_batch ---

    @responses.activate
    def test_write_batch(self):
        responses.add(responses.POST, f"{_BASE}/batch", json={"created": 2, "eventIds": ["e1", "e2"]}, status=201)

        ws = self._make_workspace()
        events = [
            {"eventId": "e1", "eventType": "custom", "useCaseId": "s", "timestamp": "2024-01-15T10:00:00Z"},
            {"eventId": "e2", "eventType": "custom", "useCaseId": "s", "timestamp": "2024-01-15T10:01:00Z"},
        ]
        result = ws.write_vision_events_batch(events)

        self.assertEqual(result["created"], 2)
        self.assertEqual(result["eventIds"], ["e1", "e2"])
        self._assert_bearer_auth()
        sent = json.loads(responses.calls[0].request.body)
        self.assertEqual(len(sent["events"]), 2)

    @responses.activate
    def test_write_batch_error(self):
        responses.add(responses.POST, f"{_BASE}/batch", json={"error": "validation"}, status=400)

        ws = self._make_workspace()
        with self.assertRaises(RoboflowError):
            ws.write_vision_events_batch([{"bad": "event"}])

    # --- query_vision_events ---

    @responses.activate
    def test_query_basic(self):
        body = {
            "events": [{"eventId": "e1"}, {"eventId": "e2"}],
            "nextCursor": None,
            "hasMore": False,
            "lookbackDays": 14,
        }
        responses.add(responses.POST, f"{_BASE}/query", json=body, status=200)

        ws = self._make_workspace()
        result = ws.query_vision_events("my-use-case")

        self.assertEqual(len(result["events"]), 2)
        self.assertFalse(result["hasMore"])
        self._assert_bearer_auth()
        sent = json.loads(responses.calls[0].request.body)
        self.assertEqual(sent["useCaseId"], "my-use-case")

    @responses.activate
    def test_query_with_filters(self):
        body = {"events": [], "nextCursor": None, "hasMore": False, "lookbackDays": 14}
        responses.add(responses.POST, f"{_BASE}/query", json=body, status=200)

        ws = self._make_workspace()
        ws.query_vision_events(
            "my-uc",
            event_type="quality_check",
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-02-01T00:00:00Z",
            limit=10,
            cursor="abc123",
            deviceId={"operator": "eq", "value": "cam-01"},
        )

        sent = json.loads(responses.calls[0].request.body)
        self.assertEqual(sent["useCaseId"], "my-uc")
        self.assertEqual(sent["eventType"], "quality_check")
        self.assertEqual(sent["startTime"], "2024-01-01T00:00:00Z")
        self.assertEqual(sent["endTime"], "2024-02-01T00:00:00Z")
        self.assertEqual(sent["limit"], 10)
        self.assertEqual(sent["cursor"], "abc123")
        self.assertEqual(sent["deviceId"], {"operator": "eq", "value": "cam-01"})

    @responses.activate
    def test_query_with_event_types_plural(self):
        body = {"events": [], "nextCursor": None, "hasMore": False, "lookbackDays": 14}
        responses.add(responses.POST, f"{_BASE}/query", json=body, status=200)

        ws = self._make_workspace()
        ws.query_vision_events("uc", event_types=["quality_check", "safety_alert"])

        sent = json.loads(responses.calls[0].request.body)
        self.assertEqual(sent["eventTypes"], ["quality_check", "safety_alert"])
        self.assertNotIn("eventType", sent)

    @responses.activate
    def test_query_omits_none_params(self):
        """Optional params that are None must not appear in the payload."""
        body = {"events": [], "nextCursor": None, "hasMore": False, "lookbackDays": 14}
        responses.add(responses.POST, f"{_BASE}/query", json=body, status=200)

        ws = self._make_workspace()
        ws.query_vision_events("uc")

        sent = json.loads(responses.calls[0].request.body)
        self.assertEqual(sent, {"useCaseId": "uc"})

    @responses.activate
    def test_query_error(self):
        responses.add(responses.POST, f"{_BASE}/query", json={"error": "unauthorized"}, status=401)

        ws = self._make_workspace()
        with self.assertRaises(RoboflowError):
            ws.query_vision_events("my-uc")

    # --- query_all_vision_events ---

    @responses.activate
    def test_query_all_single_page(self):
        body = {
            "events": [{"eventId": "e1"}],
            "nextCursor": None,
            "hasMore": False,
            "lookbackDays": 14,
        }
        responses.add(responses.POST, f"{_BASE}/query", json=body, status=200)

        ws = self._make_workspace()
        pages = list(ws.query_all_vision_events("my-uc"))

        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0][0]["eventId"], "e1")

    @responses.activate
    def test_query_all_multiple_pages(self):
        page1 = {"events": [{"eventId": "e1"}], "nextCursor": "cursor2", "hasMore": True, "lookbackDays": 14}
        page2 = {"events": [{"eventId": "e2"}], "nextCursor": None, "hasMore": False, "lookbackDays": 14}
        responses.add(responses.POST, f"{_BASE}/query", json=page1, status=200)
        responses.add(responses.POST, f"{_BASE}/query", json=page2, status=200)

        ws = self._make_workspace()
        pages = list(ws.query_all_vision_events("my-uc"))

        self.assertEqual(len(pages), 2)
        self.assertEqual(pages[0][0]["eventId"], "e1")
        self.assertEqual(pages[1][0]["eventId"], "e2")

        # Verify cursor was sent in second request
        sent2 = json.loads(responses.calls[1].request.body)
        self.assertEqual(sent2["cursor"], "cursor2")

    @responses.activate
    def test_query_all_forwards_filters(self):
        """Filters must be forwarded to every page request, not just the first."""
        page1 = {"events": [{"eventId": "e1"}], "nextCursor": "c2", "hasMore": True, "lookbackDays": 14}
        page2 = {"events": [{"eventId": "e2"}], "nextCursor": None, "hasMore": False, "lookbackDays": 14}
        responses.add(responses.POST, f"{_BASE}/query", json=page1, status=200)
        responses.add(responses.POST, f"{_BASE}/query", json=page2, status=200)

        ws = self._make_workspace()
        list(ws.query_all_vision_events("uc", event_type="quality_check", limit=1))

        sent1 = json.loads(responses.calls[0].request.body)
        sent2 = json.loads(responses.calls[1].request.body)

        # Both requests should have the filter
        self.assertEqual(sent1["eventType"], "quality_check")
        self.assertEqual(sent2["eventType"], "quality_check")
        # Second request should also have the cursor
        self.assertNotIn("cursor", sent1)
        self.assertEqual(sent2["cursor"], "c2")

    @responses.activate
    def test_query_all_empty(self):
        body = {"events": [], "nextCursor": None, "hasMore": False, "lookbackDays": 14}
        responses.add(responses.POST, f"{_BASE}/query", json=body, status=200)

        ws = self._make_workspace()
        pages = list(ws.query_all_vision_events("my-uc"))

        self.assertEqual(len(pages), 0)

    # --- list_vision_event_use_cases ---

    @responses.activate
    def test_list_use_cases(self):
        body = {
            "useCases": [
                {"id": "uc-1", "name": "QA", "status": "active"},
            ],
            "lookbackDays": 14,
        }
        responses.add(responses.GET, f"{_BASE}/use-cases", json=body, status=200)

        ws = self._make_workspace()
        result = ws.list_vision_event_use_cases()

        self.assertEqual(len(result["useCases"]), 1)
        self.assertEqual(result["useCases"][0]["name"], "QA")
        self._assert_bearer_auth()

    @responses.activate
    def test_list_use_cases_with_status(self):
        body = {"useCases": [], "lookbackDays": 14}
        responses.add(responses.GET, f"{_BASE}/use-cases", json=body, status=200)

        ws = self._make_workspace()
        result = ws.list_vision_event_use_cases(status="inactive")

        self.assertEqual(len(result["useCases"]), 0)
        # Verify status was sent as query param
        self.assertIn("status=inactive", responses.calls[0].request.url)

    @responses.activate
    def test_list_use_cases_legacy_solutions_response(self):
        responses.add(
            responses.GET,
            f"{_BASE}/use-cases",
            json={"solutions": [{"id": "uc-legacy", "name": "Legacy"}], "lookbackDays": 14},
            status=200,
        )

        ws = self._make_workspace()
        result = ws.list_vision_event_use_cases()
        self.assertEqual(result["useCases"][0]["id"], "uc-legacy")

    @responses.activate
    def test_list_use_cases_error(self):
        responses.add(responses.GET, f"{_BASE}/use-cases", json={"error": "forbidden"}, status=403)

        ws = self._make_workspace()
        with self.assertRaises(RoboflowError):
            ws.list_vision_event_use_cases()

    # --- get_vision_event_metadata_schema ---

    @responses.activate
    def test_get_metadata_schema(self):
        body = {
            "useCaseId": "manufacturing-qa",
            "fields": {
                "temperature": {"types": ["number"]},
                "zone": {"types": ["string"]},
                "active": {"types": ["boolean"]},
            },
        }
        responses.add(
            responses.GET,
            f"{_BASE}/custom-metadata-schema/manufacturing-qa",
            json=body,
            status=200,
        )

        ws = self._make_workspace()
        result = ws.get_vision_event_metadata_schema("manufacturing-qa")

        self.assertEqual(len(result["fields"]), 3)
        self.assertEqual(result["fields"]["temperature"]["types"], ["number"])
        self._assert_bearer_auth()

    @responses.activate
    def test_get_metadata_schema_error(self):
        responses.add(
            responses.GET,
            f"{_BASE}/custom-metadata-schema/nonexistent",
            json={"error": "not found"},
            status=404,
        )

        ws = self._make_workspace()
        with self.assertRaises(RoboflowError):
            ws.get_vision_event_metadata_schema("nonexistent")

    # --- upload_vision_event_image ---

    @responses.activate
    def test_upload_image(self):
        responses.add(responses.POST, f"{_BASE}/upload", json={"success": True, "sourceId": "src-123"}, status=201)

        ws = self._make_workspace()

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8\xff\xe0fake-jpeg-data")
            tmp_path = f.name

        try:
            result = ws.upload_vision_event_image(tmp_path)
            self.assertEqual(result["sourceId"], "src-123")
            self._assert_bearer_auth()
        finally:
            os.unlink(tmp_path)

    @responses.activate
    def test_upload_image_uses_basename(self):
        """When no name is provided, the multipart filename should be the basename of the path."""
        responses.add(responses.POST, f"{_BASE}/upload", json={"success": True, "sourceId": "src-789"}, status=201)

        ws = self._make_workspace()

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, prefix="myimage_") as f:
            f.write(b"\xff\xd8\xff\xe0fake")
            tmp_path = f.name

        try:
            ws.upload_vision_event_image(tmp_path)
            request_body = responses.calls[0].request.body
            basename = os.path.basename(tmp_path).encode()
            if isinstance(request_body, bytes):
                self.assertIn(basename, request_body)
        finally:
            os.unlink(tmp_path)

    @responses.activate
    def test_upload_image_with_metadata(self):
        responses.add(responses.POST, f"{_BASE}/upload", json={"success": True, "sourceId": "src-456"}, status=201)

        ws = self._make_workspace()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNGfake-png-data")
            tmp_path = f.name

        try:
            result = ws.upload_vision_event_image(
                tmp_path,
                name="custom-name.png",
                metadata={"camera_id": "cam-01"},
            )
            self.assertEqual(result["sourceId"], "src-456")

            request_body = responses.calls[0].request.body
            # Verify metadata and name were included in the multipart body
            if isinstance(request_body, bytes):
                self.assertIn(b"cam-01", request_body)
                self.assertIn(b"custom-name.png", request_body)
        finally:
            os.unlink(tmp_path)

    @responses.activate
    def test_upload_image_error(self):
        responses.add(responses.POST, f"{_BASE}/upload", json={"error": "forbidden"}, status=403)

        ws = self._make_workspace()

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"data")
            tmp_path = f.name

        try:
            with self.assertRaises(RoboflowError):
                ws.upload_vision_event_image(tmp_path)
        finally:
            os.unlink(tmp_path)


class TestVisionEventsAdapter(unittest.TestCase):
    """Tests that call the adapter directly (no Workspace). Work in slim installs."""

    API_KEY = "test_key"

    @responses.activate
    def test_adapter_write_event(self):
        from roboflow.adapters import vision_events_api

        responses.add(responses.POST, _BASE, json={"eventId": "e1", "created": True}, status=201)
        result = vision_events_api.write_event(
            self.API_KEY, {"eventId": "e1", "eventType": "custom", "useCaseId": "uc"}
        )
        self.assertEqual(result["eventId"], "e1")
        self.assertEqual(responses.calls[0].request.headers["Authorization"], f"Bearer {self.API_KEY}")

    @responses.activate
    def test_adapter_write_batch(self):
        from roboflow.adapters import vision_events_api

        responses.add(responses.POST, f"{_BASE}/batch", json={"created": 1, "eventIds": ["e1"]}, status=201)
        result = vision_events_api.write_batch(self.API_KEY, [{"eventId": "e1"}])
        self.assertEqual(result["created"], 1)

    @responses.activate
    def test_adapter_query(self):
        from roboflow.adapters import vision_events_api

        body = {"events": [{"eventId": "e1"}], "nextCursor": None, "hasMore": False, "lookbackDays": 14}
        responses.add(responses.POST, f"{_BASE}/query", json=body, status=200)
        result = vision_events_api.query(self.API_KEY, {"useCaseId": "uc"})
        self.assertEqual(len(result["events"]), 1)

    @responses.activate
    def test_adapter_list_use_cases(self):
        from roboflow.adapters import vision_events_api

        body = {"useCases": [{"id": "uc-1", "name": "QA"}], "lookbackDays": 14}
        responses.add(responses.GET, f"{_BASE}/use-cases", json=body, status=200)
        result = vision_events_api.list_use_cases(self.API_KEY)
        self.assertEqual(len(result["useCases"]), 1)

    @responses.activate
    def test_adapter_get_metadata_schema(self):
        from roboflow.adapters import vision_events_api

        body = {"useCaseId": "uc-1", "fields": {"temp": {"types": ["number"]}}}
        responses.add(responses.GET, f"{_BASE}/custom-metadata-schema/uc-1", json=body, status=200)
        result = vision_events_api.get_custom_metadata_schema(self.API_KEY, "uc-1")
        self.assertEqual(result["fields"]["temp"]["types"], ["number"])

    @responses.activate
    def test_adapter_error_raises_roboflow_error(self):
        from roboflow.adapters import vision_events_api

        responses.add(responses.POST, _BASE, json={"error": "forbidden"}, status=403)
        with self.assertRaises(RoboflowError):
            vision_events_api.write_event(self.API_KEY, {"eventId": "x"})


if __name__ == "__main__":
    unittest.main()
