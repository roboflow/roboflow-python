"""Tests for Device, devicesapi adapter, and Workspace device methods."""

from __future__ import annotations

import unittest
from typing import Any, Dict
from unittest.mock import MagicMock, patch

from roboflow.adapters import devicesapi
from roboflow.adapters.devicesapi import (
    DeviceApiError,
    DeviceAuthError,
    DeviceBadRequestError,
    DeviceNotFoundError,
    DeviceRateLimitedError,
)
from roboflow.core.device import Device

API_KEY = "fake-key"
WORKSPACE = "ws-1"
DEVICE_ID = "dev-abc"


def _mock_response(status: int, payload: Any) -> MagicMock:
    response = MagicMock()
    response.status_code = status
    response.json.return_value = payload
    response.text = "" if isinstance(payload, dict) else str(payload)
    return response


class TestDevicesApiUrlBuilding(unittest.TestCase):
    """The adapter must build correct workspace-scoped /devices/v2/* URLs."""

    @patch("roboflow.adapters.devicesapi.requests.get")
    def test_list_devices_url(self, mock_get):
        mock_get.return_value = _mock_response(200, {"data": []})
        result = devicesapi.list_devices(API_KEY, WORKSPACE)
        called_url = mock_get.call_args[0][0]
        self.assertIn(f"/{WORKSPACE}/devices/v2", called_url)
        self.assertIn(f"api_key={API_KEY}", called_url)
        self.assertEqual(result, {"data": []})

    @patch("roboflow.adapters.devicesapi.requests.get")
    def test_get_device_url(self, mock_get):
        mock_get.return_value = _mock_response(200, {"id": DEVICE_ID})
        devicesapi.get_device(API_KEY, WORKSPACE, DEVICE_ID)
        called_url = mock_get.call_args[0][0]
        self.assertIn(f"/{WORKSPACE}/devices/v2/{DEVICE_ID}", called_url)

    @patch("roboflow.adapters.devicesapi.requests.get")
    def test_list_device_streams_returns_envelope(self, mock_get):
        mock_get.return_value = _mock_response(200, {"data": [{"id": "s1"}]})
        result = devicesapi.list_device_streams(API_KEY, WORKSPACE, DEVICE_ID)
        called_url = mock_get.call_args[0][0]
        self.assertIn(f"/{WORKSPACE}/devices/v2/{DEVICE_ID}/streams", called_url)
        self.assertEqual(result, {"data": [{"id": "s1"}]})

    @patch("roboflow.adapters.devicesapi.requests.get")
    def test_logs_csv_serialization(self, mock_get):
        mock_get.return_value = _mock_response(200, {"data": [], "pagination": {}})
        devicesapi.get_device_logs(
            API_KEY,
            WORKSPACE,
            DEVICE_ID,
            service=["a", "b"],
            severity=["INFO", "WARN"],
            limit=50,
        )
        called_url = mock_get.call_args[0][0]
        # csv-serialized list params; characters URL-encoded by urllib
        self.assertIn("service=a%2Cb", called_url)
        self.assertIn("severity=INFO%2CWARN", called_url)
        self.assertIn("limit=50", called_url)

    @patch("roboflow.adapters.devicesapi.requests.get")
    def test_telemetry_time_period(self, mock_get):
        mock_get.return_value = _mock_response(200, {"buckets": []})
        devicesapi.get_device_telemetry(API_KEY, WORKSPACE, DEVICE_ID, time_period="7d")
        called_url = mock_get.call_args[0][0]
        self.assertIn("time_period=7d", called_url)

    @patch("roboflow.adapters.devicesapi.requests.get")
    def test_events_passes_cursor_unparsed(self, mock_get):
        mock_get.return_value = _mock_response(200, {"data": [], "pagination": {}})
        # Cursors are opaque base64url strings; must round-trip without parsing.
        cursor = "eyJ0aW1lc3RhbXAiOiAiMjAyNi0wNC0yMyAxMDowMDowMCJ9"
        devicesapi.get_device_events(API_KEY, WORKSPACE, DEVICE_ID, cursor=cursor, direction="forward")
        called_url = mock_get.call_args[0][0]
        self.assertIn(f"cursor={cursor}", called_url)
        self.assertIn("direction=forward", called_url)

    @patch("roboflow.adapters.devicesapi.requests.post")
    def test_create_device_body_field_names(self, mock_post):
        mock_post.return_value = _mock_response(201, {"deviceId": "d1", "installId": "i1"})
        devicesapi.create_device(
            API_KEY,
            WORKSPACE,
            device_name="Cam 1",
            device_type="edge",
            workflow_id="wf-1",
            tags=["a"],
            offline_mode=True,
            source_device_id="other",
        )
        body = mock_post.call_args.kwargs["json"]
        self.assertEqual(body["device_name"], "Cam 1")
        self.assertEqual(body["device_type"], "edge")
        self.assertEqual(body["workflow_id"], "wf-1")
        self.assertEqual(body["tags"], ["a"])
        self.assertTrue(body["offline_mode"])
        # Body field is camelCase per docs/api/deployments/overview.md
        self.assertEqual(body["sourceDeviceId"], "other")

    @patch("roboflow.adapters.devicesapi.requests.post")
    @patch("roboflow.adapters.devicesapi.requests.get")
    def test_requests_use_default_timeout(self, mock_get, mock_post):
        mock_get.return_value = _mock_response(200, {"data": [], "pagination": {}})
        mock_post.return_value = _mock_response(201, {"deviceId": "d1", "installId": "i1"})

        devicesapi.list_devices(API_KEY, WORKSPACE)
        devicesapi.create_device(API_KEY, WORKSPACE, device_name="Cam 1")
        devicesapi.get_device(API_KEY, WORKSPACE, DEVICE_ID)
        devicesapi.get_device_config(API_KEY, WORKSPACE, DEVICE_ID)
        devicesapi.get_device_config_history(API_KEY, WORKSPACE, DEVICE_ID)
        devicesapi.list_device_streams(API_KEY, WORKSPACE, DEVICE_ID)
        devicesapi.get_device_stream(API_KEY, WORKSPACE, DEVICE_ID, "s1")
        devicesapi.get_device_logs(API_KEY, WORKSPACE, DEVICE_ID)
        devicesapi.get_device_telemetry(API_KEY, WORKSPACE, DEVICE_ID)
        devicesapi.get_device_events(API_KEY, WORKSPACE, DEVICE_ID)

        for call in mock_get.call_args_list:
            self.assertEqual(call.kwargs["timeout"], devicesapi.DEFAULT_TIMEOUT)
        for call in mock_post.call_args_list:
            self.assertEqual(call.kwargs["timeout"], devicesapi.DEFAULT_TIMEOUT)


class TestDevicesApiErrors(unittest.TestCase):
    """Each non-2xx HTTP status maps to a typed exception."""

    def _expect(self, status: int, expected_cls: type) -> None:
        with patch("roboflow.adapters.devicesapi.requests.get") as mock_get:
            mock_get.return_value = _mock_response(status, {"error": "bad"})
            with self.assertRaises(expected_cls) as ctx:
                devicesapi.get_device(API_KEY, WORKSPACE, DEVICE_ID)
            self.assertEqual(ctx.exception.status_code, status)

    def test_400_bad_request(self) -> None:
        self._expect(400, DeviceBadRequestError)

    def test_401_auth(self) -> None:
        self._expect(401, DeviceAuthError)

    def test_403_auth(self) -> None:
        self._expect(403, DeviceAuthError)

    def test_404_not_found(self) -> None:
        self._expect(404, DeviceNotFoundError)

    def test_404_missing_scope_is_auth(self) -> None:
        # validateToken.js returns 404 + GraphMethodException when the api_key
        # is valid for the workspace but lacks the device:read/update scope.
        body = {"error": {"type": "GraphMethodException", "message": "scope missing"}}
        with patch("roboflow.adapters.devicesapi.requests.get") as mock_get:
            mock_get.return_value = _mock_response(404, body)
            with self.assertRaises(DeviceAuthError) as ctx:
                devicesapi.get_device(API_KEY, WORKSPACE, DEVICE_ID)
            self.assertEqual(ctx.exception.status_code, 404)

    def test_429_rate_limit(self) -> None:
        self._expect(429, DeviceRateLimitedError)

    def test_500_generic(self) -> None:
        self._expect(500, DeviceApiError)

    def test_500_truncates_huge_response_body(self) -> None:
        # Server-side 500s sometimes return a multi-KB HTML stack trace. The
        # adapter must cap that before it lands in str(exc).
        huge_body = "X" * 10_000  # 10x the cap
        with patch("roboflow.adapters.devicesapi.requests.get") as mock_get:
            response = MagicMock()
            response.status_code = 500
            response.json.side_effect = ValueError("not JSON")
            response.text = huge_body
            mock_get.return_value = response
            with self.assertRaises(DeviceApiError) as ctx:
                devicesapi.get_device(API_KEY, WORKSPACE, DEVICE_ID)
            msg = str(ctx.exception)
            self.assertLess(len(msg), len(huge_body))
            self.assertTrue(msg.endswith("…[truncated]"))


class TestDeviceClass(unittest.TestCase):
    """Device exposes the per-device sub-resources."""

    def setUp(self) -> None:
        self.info: Dict[str, Any] = {
            "id": DEVICE_ID,
            "name": "Cam 1",
            "status": "online",
            "type": "edge",
            "tags": ["floor-1"],
        }
        self.device = Device(API_KEY, WORKSPACE, self.info)

    def test_init_caches_summary_fields(self) -> None:
        self.assertEqual(self.device.id, DEVICE_ID)
        self.assertEqual(self.device.name, "Cam 1")
        self.assertEqual(self.device.status, "online")
        self.assertEqual(self.device.type, "edge")
        self.assertEqual(self.device.tags, ["floor-1"])

    @patch("roboflow.adapters.devicesapi.get_device_config")
    def test_config_calls_adapter(self, mock_config) -> None:
        mock_config.return_value = {"device_id": DEVICE_ID, "config": {}}
        result = self.device.config()
        mock_config.assert_called_once_with(API_KEY, WORKSPACE, DEVICE_ID)
        self.assertEqual(result["device_id"], DEVICE_ID)

    @patch("roboflow.adapters.devicesapi.get_device_config_history")
    def test_config_history_passes_cursor(self, mock_hist) -> None:
        mock_hist.return_value = {"data": [], "pagination": {}}
        self.device.config_history(limit=20, cursor="2026-04-23T10:00:00Z")
        mock_hist.assert_called_once_with(API_KEY, WORKSPACE, DEVICE_ID, limit=20, cursor="2026-04-23T10:00:00Z")

    @patch("roboflow.adapters.devicesapi.list_device_streams")
    def test_streams(self, mock_streams) -> None:
        mock_streams.return_value = {"data": [{"id": "s1"}]}
        self.assertEqual(self.device.streams(), [{"id": "s1"}])

    @patch("roboflow.adapters.devicesapi.get_device_stream")
    def test_stream(self, mock_stream) -> None:
        mock_stream.return_value = {"id": "s1"}
        self.device.stream("s1")
        mock_stream.assert_called_once_with(API_KEY, WORKSPACE, DEVICE_ID, "s1")

    @patch("roboflow.adapters.devicesapi.get_device_logs")
    def test_logs_forwards_kwargs(self, mock_logs) -> None:
        mock_logs.return_value = {"data": [], "pagination": {}}
        self.device.logs(severity=["ERROR"], limit=10)
        kwargs = mock_logs.call_args.kwargs
        self.assertEqual(kwargs["severity"], ["ERROR"])
        self.assertEqual(kwargs["limit"], 10)

    @patch("roboflow.adapters.devicesapi.get_device_telemetry")
    def test_telemetry(self, mock_tel) -> None:
        mock_tel.return_value = {"buckets": []}
        self.device.telemetry("1h")
        mock_tel.assert_called_once_with(API_KEY, WORKSPACE, DEVICE_ID, time_period="1h")

    @patch("roboflow.adapters.devicesapi.get_device_events")
    def test_events_forwards_all_filters(self, mock_events) -> None:
        mock_events.return_value = {"data": [], "pagination": {}}
        self.device.events(
            entity_type="stream",
            entity_id="pipe-1",
            event="stream_started",
            start_time="2026-04-01T00:00:00Z",
            end_time="2026-04-30T00:00:00Z",
            limit=200,
            cursor="opaque",
            direction="forward",
        )
        kwargs = mock_events.call_args.kwargs
        self.assertEqual(kwargs["entity_type"], "stream")
        self.assertEqual(kwargs["entity_id"], "pipe-1")
        self.assertEqual(kwargs["event"], "stream_started")
        self.assertEqual(kwargs["limit"], 200)
        self.assertEqual(kwargs["cursor"], "opaque")
        self.assertEqual(kwargs["direction"], "forward")

    @patch("roboflow.adapters.devicesapi.get_device")
    def test_refresh_updates_fields(self, mock_get) -> None:
        mock_get.return_value = {"id": DEVICE_ID, "name": "Cam 1 (renamed)", "status": "offline", "tags": []}
        self.device.refresh()
        self.assertEqual(self.device.name, "Cam 1 (renamed)")
        self.assertEqual(self.device.status, "offline")
        self.assertEqual(self.device.tags, [])


class TestWorkspaceDeviceMethods(unittest.TestCase):
    """Workspace.devices() / .device() / .create_device() route through the adapter."""

    def setUp(self) -> None:
        from roboflow.core.workspace import Workspace

        info = {"workspace": {"name": "Test", "url": WORKSPACE, "projects": []}}
        self.workspace = Workspace(info=info, api_key=API_KEY, default_workspace=WORKSPACE, model_format="yolov8")

    @patch("roboflow.adapters.devicesapi.list_devices")
    def test_devices_returns_device_objects(self, mock_list) -> None:
        mock_list.return_value = {"data": [{"id": "a"}, {"id": "b"}]}
        devices = self.workspace.devices()
        self.assertEqual(len(devices), 2)
        self.assertIsInstance(devices[0], Device)
        self.assertEqual(devices[0].id, "a")

    @patch("roboflow.adapters.devicesapi.get_device")
    def test_device_returns_single(self, mock_get) -> None:
        mock_get.return_value = {"id": DEVICE_ID, "name": "Cam"}
        device = self.workspace.device(DEVICE_ID)
        self.assertIsInstance(device, Device)
        self.assertEqual(device.id, DEVICE_ID)
        self.assertEqual(device.name, "Cam")

    @patch("roboflow.adapters.devicesapi.create_device")
    def test_create_device_forwards_kwargs(self, mock_create) -> None:
        mock_create.return_value = {"deviceId": "d1", "installId": "i1"}
        result = self.workspace.create_device("Cam 1", device_type="edge", workflow_id="wf-1", tags=["a"])
        self.assertEqual(result["deviceId"], "d1")
        kwargs = mock_create.call_args.kwargs
        self.assertEqual(kwargs["device_name"], "Cam 1")
        self.assertEqual(kwargs["device_type"], "edge")
        self.assertEqual(kwargs["workflow_id"], "wf-1")
        self.assertEqual(kwargs["tags"], ["a"])


if __name__ == "__main__":
    unittest.main()
