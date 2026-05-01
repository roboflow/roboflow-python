"""Adapter for the workspace-scoped device management API.

Wraps the read-only external observability endpoints plus device create
served by the ``light.v2.device`` Cloud Function. Routes are documented in
``docs/api/deployments/overview.md`` of the ``roboflow/roboflow`` repo.

Read endpoints require the ``device:read`` scope; create requires
``device:update``. Authentication is via the workspace api_key.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import requests

from roboflow.adapters.rfapi import RoboflowError
from roboflow.config import API_URL


class DeviceApiError(RoboflowError):
    """Raised when a device API call returns a non-success status."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        self.status_code = status_code
        super().__init__(message)


class DeviceNotFoundError(DeviceApiError):
    """404 — device or stream does not exist or is owned by a different workspace."""


class DeviceAuthError(DeviceApiError):
    """401/403 — missing key, wrong scope, or device-bound key targeting a sibling."""


class DeviceRateLimitedError(DeviceApiError):
    """429 — logs (5/min/IP) or telemetry (60/min) limit hit."""


class DeviceBadRequestError(DeviceApiError):
    """400 — malformed cursor, unparseable date, unknown ``time_period``."""


def _build_url(workspace: str, path: str, api_key: str, query: Optional[Dict[str, Any]] = None) -> str:
    base = f"{API_URL}/{workspace}/devices/v2{path}"
    params: Dict[str, Any] = {"api_key": api_key}
    if query:
        for key, value in query.items():
            if value is None:
                continue
            if isinstance(value, list):
                if not value:
                    continue
                params[key] = ",".join(str(v) for v in value)
            else:
                params[key] = value
    return f"{base}?{urlencode(params, doseq=False)}"


def _raise_for_status(response: requests.Response) -> None:
    if response.status_code < 400:
        return
    error_type: Optional[str] = None
    try:
        payload = response.json()
        err = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(err, dict):
            message = err.get("message") or response.text
            raw_type = err.get("type")
            error_type = raw_type if isinstance(raw_type, str) else None
        elif isinstance(err, str):
            message = err
        else:
            message = response.text
    except Exception:  # noqa: BLE001
        message = response.text
    code = response.status_code
    if code == 400:
        raise DeviceBadRequestError(message or "Bad request", status_code=code)
    if code in (401, 403):
        raise DeviceAuthError(message or "Unauthorized", status_code=code)
    if code == 404:
        # validateToken.js returns 404 + GraphMethodException when an api_key
        # is valid for this workspace but lacks the required scope
        # (device:read / device:update). Surface that as auth so the CLI
        # exits 2 with the scope hint instead of 3 ("not found").
        if error_type == "GraphMethodException":
            raise DeviceAuthError(message or "Forbidden", status_code=code)
        raise DeviceNotFoundError(message or "Not found", status_code=code)
    if code == 429:
        raise DeviceRateLimitedError(message or "Rate limited", status_code=code)
    raise DeviceApiError(message or f"HTTP {code}", status_code=code)


def list_devices(api_key: str, workspace: str) -> Dict[str, Any]:
    """``GET /:workspace/devices/v2`` — returns the parsed JSON response."""
    response = requests.get(_build_url(workspace, "", api_key))
    _raise_for_status(response)
    return response.json()


def create_device(
    api_key: str,
    workspace: str,
    *,
    device_name: str,
    device_type: Optional[str] = None,
    workflow_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    offline_mode: Optional[bool] = None,
    source_device_id: Optional[str] = None,
) -> Dict[str, Any]:
    """``POST /:workspace/devices/v2`` — returns ``{ deviceId, installId }``."""
    body: Dict[str, Any] = {"device_name": device_name}
    if device_type is not None:
        body["device_type"] = device_type
    if workflow_id is not None:
        body["workflow_id"] = workflow_id
    if tags is not None:
        body["tags"] = tags
    if offline_mode is not None:
        body["offline_mode"] = offline_mode
    if source_device_id is not None:
        # Body field is camelCase per docs/api/deployments/overview.md
        body["sourceDeviceId"] = source_device_id
    response = requests.post(_build_url(workspace, "", api_key), json=body)
    _raise_for_status(response)
    return response.json()


def get_device(api_key: str, workspace: str, device_id: str) -> Dict[str, Any]:
    """``GET /:workspace/devices/v2/:deviceId``."""
    response = requests.get(_build_url(workspace, f"/{device_id}", api_key))
    _raise_for_status(response)
    return response.json()


def get_device_config(api_key: str, workspace: str, device_id: str) -> Dict[str, Any]:
    """``GET /:workspace/devices/v2/:deviceId/config``.

    Note:
        The response can include ``environment_variables`` and integration
        credentials. Treat the returned dict as sensitive.
    """
    response = requests.get(_build_url(workspace, f"/{device_id}/config", api_key))
    _raise_for_status(response)
    return response.json()


def get_device_config_history(
    api_key: str,
    workspace: str,
    device_id: str,
    *,
    limit: Optional[int] = None,
    cursor: Optional[str] = None,
) -> Dict[str, Any]:
    """``GET /:workspace/devices/v2/:deviceId/config/history``."""
    response = requests.get(
        _build_url(
            workspace,
            f"/{device_id}/config/history",
            api_key,
            query={"limit": limit, "cursor": cursor},
        )
    )
    _raise_for_status(response)
    return response.json()


def list_device_streams(api_key: str, workspace: str, device_id: str) -> Dict[str, Any]:
    """``GET /:workspace/devices/v2/:deviceId/streams``."""
    response = requests.get(_build_url(workspace, f"/{device_id}/streams", api_key))
    _raise_for_status(response)
    return response.json()


def get_device_stream(api_key: str, workspace: str, device_id: str, stream_id: str) -> Dict[str, Any]:
    """``GET /:workspace/devices/v2/:deviceId/streams/:streamId``."""
    response = requests.get(_build_url(workspace, f"/{device_id}/streams/{stream_id}", api_key))
    _raise_for_status(response)
    return response.json()


def get_device_logs(
    api_key: str,
    workspace: str,
    device_id: str,
    *,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    service: Optional[List[str]] = None,
    severity: Optional[List[str]] = None,
    limit: Optional[int] = None,
    cursor: Optional[str] = None,
) -> Dict[str, Any]:
    """``GET /:workspace/devices/v2/:deviceId/logs``. Rate limited 5/min/IP."""
    response = requests.get(
        _build_url(
            workspace,
            f"/{device_id}/logs",
            api_key,
            query={
                "start_time": start_time,
                "end_time": end_time,
                "service": service,
                "severity": severity,
                "limit": limit,
                "cursor": cursor,
            },
        )
    )
    _raise_for_status(response)
    return response.json()


def get_device_telemetry(
    api_key: str,
    workspace: str,
    device_id: str,
    *,
    time_period: Optional[str] = None,
) -> Dict[str, Any]:
    """``GET /:workspace/devices/v2/:deviceId/telemetry``. Rate limited 60/min."""
    response = requests.get(
        _build_url(
            workspace,
            f"/{device_id}/telemetry",
            api_key,
            query={"time_period": time_period},
        )
    )
    _raise_for_status(response)
    return response.json()


def get_device_events(
    api_key: str,
    workspace: str,
    device_id: str,
    *,
    entity_type: Optional[str] = None,
    entity_id: Optional[str] = None,
    event: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    limit: Optional[int] = None,
    cursor: Optional[str] = None,
    direction: Optional[str] = None,
) -> Dict[str, Any]:
    """``GET /:workspace/devices/v2/:deviceId/events``."""
    response = requests.get(
        _build_url(
            workspace,
            f"/{device_id}/events",
            api_key,
            query={
                "entity_type": entity_type,
                "entity_id": entity_id,
                "event": event,
                "start_time": start_time,
                "end_time": end_time,
                "limit": limit,
                "cursor": cursor,
                "direction": direction,
            },
        )
    )
    _raise_for_status(response)
    return response.json()
