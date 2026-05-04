"""Workspace-scoped device handle.

Wraps the read endpoints of the external Deployments API
(``/:workspace/devices/v2/*``) added in roboflow/roboflow PR #11350. A
``Device`` is constructed by ``Workspace.device(id)`` or implicitly when
listing via ``Workspace.devices()``; it caches the device summary returned
by the API and exposes lazy methods for the per-device sub-resources.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from roboflow.adapters import devicesapi


class Device:
    """A v2 Roboflow device (RFDM, AI1, edge, …).

    Instances are created by :class:`roboflow.core.workspace.Workspace`.
    The ``info`` dict mirrors the entity documented in
    ``docs/api/deployments/overview.md`` of the platform repo (fields
    ``id``, ``name``, ``status``, ``last_heartbeat``, ``platform``,
    ``hardware``, ``tags``, …).

    Note:
        :meth:`config` returns the raw Firestore config doc, which can
        contain ``environment_variables`` and integration credentials.
    """

    def __init__(self, api_key: str, workspace_url: str, info: Dict[str, Any]) -> None:
        self.__api_key = api_key
        self.__workspace = workspace_url
        self.info: Dict[str, Any] = info
        self.id: str = info.get("id", "")
        self.name: Optional[str] = info.get("name")
        self.status: Optional[str] = info.get("status")
        self.type: Optional[str] = info.get("type")
        self.tags: List[str] = list(info.get("tags") or [])

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"Device(id={self.id!r}, name={self.name!r}, status={self.status!r})"

    def refresh(self) -> "Device":
        """Re-fetch the device summary from the API."""
        self.info = devicesapi.get_device(self.__api_key, self.__workspace, self.id)
        self.name = self.info.get("name")
        self.status = self.info.get("status")
        self.type = self.info.get("type")
        self.tags = list(self.info.get("tags") or [])
        return self

    def config(self) -> Dict[str, Any]:
        """Fetch the device's full runtime config (sensitive — see class docstring)."""
        return devicesapi.get_device_config(self.__api_key, self.__workspace, self.id)

    def config_history(
        self,
        *,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List prior config revisions, newest first.

        Args:
            limit: 1-500, default 10.
            cursor: ISO timestamp from a previous page's ``next_cursor``.
        """
        return devicesapi.get_device_config_history(
            self.__api_key, self.__workspace, self.id, limit=limit, cursor=cursor
        )

    def streams(self) -> List[Dict[str, Any]]:
        """List streams currently configured on this device."""
        return devicesapi.list_device_streams(self.__api_key, self.__workspace, self.id).get("data", [])

    def stream(self, stream_id: str) -> Dict[str, Any]:
        """Get a single stream by id."""
        return devicesapi.get_device_stream(self.__api_key, self.__workspace, self.id, stream_id)

    def logs(
        self,
        *,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        service: Optional[List[str]] = None,
        severity: Optional[List[str]] = None,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch device logs from Elasticsearch (5/min/IP rate limit).

        Args:
            start_time: ISO timestamp.
            end_time: ISO timestamp.
            service: List of service names; serialized as comma-separated string.
            severity: List of severity levels (``INFO``, ``WARN``, ``ERROR``, …).
            limit: 1-1000, default 100.
            cursor: ISO timestamp from a previous page's ``next_cursor``.
        """
        return devicesapi.get_device_logs(
            self.__api_key,
            self.__workspace,
            self.id,
            start_time=start_time,
            end_time=end_time,
            service=service,
            severity=severity,
            limit=limit,
            cursor=cursor,
        )

    def telemetry(self, time_period: Optional[str] = None) -> Dict[str, Any]:
        """Fetch aggregated hardware telemetry (60/min rate limit).

        Args:
            time_period: One of ``"1h"``, ``"24h"`` (default), ``"7d"``, ``"14d"``.
        """
        return devicesapi.get_device_telemetry(self.__api_key, self.__workspace, self.id, time_period=time_period)

    def events(
        self,
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
        """Query device/stream lifecycle events.

        Args:
            entity_type: Filter to a single entity type (``stream``, ``device``, …).
            entity_id: Filter to a single entity id.
            event: Filter by event name.
            start_time: ISO timestamp.
            end_time: ISO timestamp.
            limit: 1-1000, default 100.
            cursor: Opaque base64url cursor from a previous page (round-trip only;
                do not parse).
            direction: ``"forward"`` or ``"backward"`` (default ``"backward"``).
        """
        return devicesapi.get_device_events(
            self.__api_key,
            self.__workspace,
            self.id,
            entity_type=entity_type,
            entity_id=entity_id,
            event=event,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            cursor=cursor,
            direction=direction,
        )
