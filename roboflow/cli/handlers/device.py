"""Device management commands.

Wraps the workspace-scoped Deployments / Device Management API
(``/:workspace/devices/v2/*``). All commands honor ``--workspace`` /
``--api-key`` from the global callback and ``--json`` for stable output.

Exit codes:
    0  success
    1  general error (incl. 400 bad params, 429 rate limited)
    2  auth (401/403)
    3  not found (404)
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional

import typer

from roboflow.cli._compat import SortedGroup, ctx_to_args

device_app = typer.Typer(cls=SortedGroup, help="Manage RFDM devices", no_args_is_help=True)


def _resolve_ws_and_key(args):  # noqa: ANN001
    from roboflow.cli._resolver import resolve_ws_and_key

    return resolve_ws_and_key(args)


def _exit_code_for(exc: Exception) -> int:
    from roboflow.adapters.devicesapi import (
        DeviceAuthError,
        DeviceNotFoundError,
        DeviceRateLimitedError,
    )

    if isinstance(exc, DeviceAuthError):
        return 2
    if isinstance(exc, DeviceNotFoundError):
        return 3
    if isinstance(exc, DeviceRateLimitedError):
        return 1
    return 1


def _hint_for(exc: Exception) -> Optional[str]:
    from roboflow.adapters.devicesapi import DeviceAuthError, DeviceRateLimitedError

    if isinstance(exc, DeviceRateLimitedError):
        return "Logs are limited to 5 req/min/IP and telemetry to 60 req/min — wait and retry."
    if isinstance(exc, DeviceAuthError):
        return "Verify the api_key has the device:read scope, or device:update for create."
    return None


def _split_csv(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return parts or None


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@device_app.command("list")
def list_cmd(ctx: typer.Context) -> None:
    """List devices in the workspace."""
    args = ctx_to_args(ctx)
    _list(args)


@device_app.command("get")
def get_cmd(
    ctx: typer.Context,
    device_id: Annotated[str, typer.Argument(help="Device ID")],
) -> None:
    """Show a single device."""
    args = ctx_to_args(ctx, device_id=device_id)
    _get(args)


@device_app.command("create")
def create_cmd(
    ctx: typer.Context,
    device_name: Annotated[str, typer.Argument(help="Human-readable device name")],
    device_type: Annotated[Optional[str], typer.Option("--type", help="Device type: ai1, edge, or custom")] = None,
    workflow_id: Annotated[
        Optional[str], typer.Option("--workflow-id", help="Initial workflow assignment (AI1 only)")
    ] = None,
    tags: Annotated[Optional[str], typer.Option("--tags", help="Comma-separated tags")] = None,
    offline_mode: Annotated[
        Optional[bool], typer.Option("--offline-mode/--no-offline-mode", help="AI1 offline mode")
    ] = None,
    source_device_id: Annotated[
        Optional[str], typer.Option("--source-device-id", help="Duplicate config from this device")
    ] = None,
) -> None:
    """Create a v2 device. Requires the device:update scope."""
    args = ctx_to_args(
        ctx,
        device_name=device_name,
        device_type=device_type,
        workflow_id=workflow_id,
        tags=_split_csv(tags),
        offline_mode=offline_mode,
        source_device_id=source_device_id,
    )
    _create(args)


@device_app.command("config")
def config_cmd(
    ctx: typer.Context,
    device_id: Annotated[str, typer.Argument(help="Device ID")],
) -> None:
    """Show the device's full runtime config (sensitive — may contain credentials)."""
    args = ctx_to_args(ctx, device_id=device_id)
    _config(args)


@device_app.command("config-history")
def config_history_cmd(
    ctx: typer.Context,
    device_id: Annotated[str, typer.Argument(help="Device ID")],
    limit: Annotated[Optional[int], typer.Option("--limit", help="Max revisions (1-500, default 10)")] = None,
    cursor: Annotated[Optional[str], typer.Option("--cursor", help="ISO timestamp from previous next_cursor")] = None,
) -> None:
    """List prior config revisions, newest first."""
    args = ctx_to_args(ctx, device_id=device_id, limit=limit, cursor=cursor)
    _config_history(args)


@device_app.command("streams")
def streams_cmd(
    ctx: typer.Context,
    device_id: Annotated[str, typer.Argument(help="Device ID")],
) -> None:
    """List streams configured on the device."""
    args = ctx_to_args(ctx, device_id=device_id)
    _streams(args)


@device_app.command("stream")
def stream_cmd(
    ctx: typer.Context,
    device_id: Annotated[str, typer.Argument(help="Device ID")],
    stream_id: Annotated[str, typer.Argument(help="Stream ID")],
) -> None:
    """Show a single stream."""
    args = ctx_to_args(ctx, device_id=device_id, stream_id=stream_id)
    _stream(args)


@device_app.command("logs")
def logs_cmd(
    ctx: typer.Context,
    device_id: Annotated[str, typer.Argument(help="Device ID")],
    start_time: Annotated[Optional[str], typer.Option("--start-time", help="ISO timestamp")] = None,
    end_time: Annotated[Optional[str], typer.Option("--end-time", help="ISO timestamp")] = None,
    service: Annotated[Optional[str], typer.Option("--service", help="Comma-separated service names")] = None,
    severity: Annotated[
        Optional[str], typer.Option("--severity", help="Comma-separated levels (INFO,WARN,ERROR,...)")
    ] = None,
    limit: Annotated[Optional[int], typer.Option("--limit", help="1-1000, default 100")] = None,
    cursor: Annotated[Optional[str], typer.Option("--cursor", help="ISO timestamp from previous next_cursor")] = None,
) -> None:
    """Fetch device logs (5 req/min/IP)."""
    args = ctx_to_args(
        ctx,
        device_id=device_id,
        start_time=start_time,
        end_time=end_time,
        service=_split_csv(service),
        severity=_split_csv(severity),
        limit=limit,
        cursor=cursor,
    )
    _logs(args)


@device_app.command("telemetry")
def telemetry_cmd(
    ctx: typer.Context,
    device_id: Annotated[str, typer.Argument(help="Device ID")],
    time_period: Annotated[
        Optional[str], typer.Option("--time-period", help="One of 1h, 24h (default), 7d, 14d")
    ] = None,
) -> None:
    """Fetch aggregated hardware telemetry (60 req/min)."""
    args = ctx_to_args(ctx, device_id=device_id, time_period=time_period)
    _telemetry(args)


@device_app.command("events")
def events_cmd(
    ctx: typer.Context,
    device_id: Annotated[str, typer.Argument(help="Device ID")],
    entity_type: Annotated[Optional[str], typer.Option("--entity-type", help="Filter to a single entity type")] = None,
    entity_id: Annotated[Optional[str], typer.Option("--entity-id", help="Filter to a single entity id")] = None,
    event: Annotated[Optional[str], typer.Option("--event", help="Filter by event name")] = None,
    start_time: Annotated[Optional[str], typer.Option("--start-time", help="ISO timestamp")] = None,
    end_time: Annotated[Optional[str], typer.Option("--end-time", help="ISO timestamp")] = None,
    limit: Annotated[Optional[int], typer.Option("--limit", help="1-1000, default 100")] = None,
    cursor: Annotated[
        Optional[str], typer.Option("--cursor", help="Opaque base64url cursor from previous page")
    ] = None,
    direction: Annotated[
        Optional[str], typer.Option("--direction", help="forward or backward (default backward)")
    ] = None,
) -> None:
    """Query device/stream lifecycle events."""
    args = ctx_to_args(
        ctx,
        device_id=device_id,
        entity_type=entity_type,
        entity_id=entity_id,
        event=event,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
        cursor=cursor,
        direction=direction,
    )
    _events(args)


# ---------------------------------------------------------------------------
# Business logic
# ---------------------------------------------------------------------------


def _list(args) -> None:  # noqa: ANN001
    from roboflow.adapters import devicesapi
    from roboflow.cli._output import output, output_error
    from roboflow.cli._table import format_table

    resolved = _resolve_ws_and_key(args)
    if not resolved:
        return
    ws, api_key = resolved

    try:
        result = devicesapi.list_devices(api_key, ws)
    except Exception as exc:  # noqa: BLE001
        output_error(args, str(exc), hint=_hint_for(exc), exit_code=_exit_code_for(exc))
        return

    rows: List[Dict[str, Any]] = result.get("data", [])
    table_rows = [
        {
            "id": r.get("id", ""),
            "name": r.get("name", "") or "",
            "status": r.get("status", "") or "",
            "type": r.get("type", "") or "",
            "last_heartbeat": r.get("last_heartbeat", "") or "",
        }
        for r in rows
    ]
    table = format_table(
        table_rows,
        columns=["id", "name", "status", "type", "last_heartbeat"],
        headers=["ID", "NAME", "STATUS", "TYPE", "LAST HEARTBEAT"],
    )
    output(args, result, text=table)


def _get(args) -> None:  # noqa: ANN001
    from roboflow.adapters import devicesapi
    from roboflow.cli._output import output, output_error

    resolved = _resolve_ws_and_key(args)
    if not resolved:
        return
    ws, api_key = resolved

    try:
        device = devicesapi.get_device(api_key, ws, args.device_id)
    except Exception as exc:  # noqa: BLE001
        output_error(args, str(exc), hint=_hint_for(exc), exit_code=_exit_code_for(exc))
        return

    hardware = device.get("hardware") or {}
    lines = [
        f"Device: {device.get('name') or device.get('id')}",
        f"  ID: {device.get('id', '')}",
        f"  Status: {device.get('status', '')}",
        f"  Type: {device.get('type') or ''}",
        f"  Platform: {device.get('platform') or ''}",
        f"  RFDM Version: {device.get('rfdm_version') or ''}",
        f"  Last Heartbeat: {device.get('last_heartbeat') or ''}",
        f"  Memory: {hardware.get('total_memory_mb') or ''} MB",
        f"  Disk: {hardware.get('total_disk_space_mb') or ''} MB",
    ]
    tags = device.get("tags") or []
    if tags:
        lines.append(f"  Tags: {', '.join(tags)}")
    output(args, device, text="\n".join(lines))


def _create(args) -> None:  # noqa: ANN001
    from roboflow.adapters import devicesapi
    from roboflow.cli._output import output, output_error

    resolved = _resolve_ws_and_key(args)
    if not resolved:
        return
    ws, api_key = resolved

    try:
        result = devicesapi.create_device(
            api_key,
            ws,
            device_name=args.device_name,
            device_type=args.device_type,
            workflow_id=args.workflow_id,
            tags=args.tags,
            offline_mode=args.offline_mode,
            source_device_id=args.source_device_id,
        )
    except Exception as exc:  # noqa: BLE001
        output_error(args, str(exc), hint=_hint_for(exc), exit_code=_exit_code_for(exc))
        return

    output(
        args,
        result,
        text=(
            f"Created device '{args.device_name}'\n"
            f"  Device ID: {result.get('deviceId', '')}\n"
            f"  Install ID: {result.get('installId', '')}"
        ),
    )


def _config(args) -> None:  # noqa: ANN001
    import sys

    from roboflow.adapters import devicesapi
    from roboflow.cli._output import output, output_error

    resolved = _resolve_ws_and_key(args)
    if not resolved:
        return
    ws, api_key = resolved

    try:
        config = devicesapi.get_device_config(api_key, ws, args.device_id)
    except Exception as exc:  # noqa: BLE001
        output_error(args, str(exc), hint=_hint_for(exc), exit_code=_exit_code_for(exc))
        return

    # `GET .../config` is a documented passthrough of the Firestore config doc — it can
    # contain `environment_variables` and integration credentials. We deliberately do
    # NOT redact: that would silently corrupt round-trips (backup/restore/diff) and
    # diverge from what the API contract returns. Instead, surface a stderr warning
    # in interactive (non-JSON, non-quiet) mode so a human running `roboflow device
    # config <id>` is reminded before they paste the output anywhere. JSON mode stays
    # byte-identical to the API response.
    if not getattr(args, "json", False) and not getattr(args, "quiet", False):
        sys.stderr.write(
            "WARNING: Device config may contain environment variables, API keys, "
            "and integration credentials. Do not paste this output into chats, "
            "tickets, screenshots, or shared logs.\n"
        )
    output(args, config)


def _config_history(args) -> None:  # noqa: ANN001
    from roboflow.adapters import devicesapi
    from roboflow.cli._output import output, output_error
    from roboflow.cli._table import format_table

    resolved = _resolve_ws_and_key(args)
    if not resolved:
        return
    ws, api_key = resolved

    try:
        result = devicesapi.get_device_config_history(api_key, ws, args.device_id, limit=args.limit, cursor=args.cursor)
    except Exception as exc:  # noqa: BLE001
        output_error(args, str(exc), hint=_hint_for(exc), exit_code=_exit_code_for(exc))
        return

    revisions = result.get("data", [])
    rows = [
        {
            "revision_id": r.get("revision_id", "") or "",
            "created_at": r.get("created_at", "") or "",
            "created_by": r.get("created_by", "") or "",
        }
        for r in revisions
    ]
    table = format_table(
        rows,
        columns=["revision_id", "created_at", "created_by"],
        headers=["REVISION", "CREATED AT", "CREATED BY"],
    )
    output(args, result, text=table)


def _streams(args) -> None:  # noqa: ANN001
    from roboflow.adapters import devicesapi
    from roboflow.cli._output import output, output_error
    from roboflow.cli._table import format_table

    resolved = _resolve_ws_and_key(args)
    if not resolved:
        return
    ws, api_key = resolved

    try:
        result = devicesapi.list_device_streams(api_key, ws, args.device_id)
    except Exception as exc:  # noqa: BLE001
        output_error(args, str(exc), hint=_hint_for(exc), exit_code=_exit_code_for(exc))
        return

    streams = result.get("data", [])
    rows = [
        {
            "id": s.get("id", "") or "",
            "name": s.get("name", "") or "",
            "status": s.get("status", "") or "",
            "workflow_id": s.get("workflow_id", "") or "",
        }
        for s in streams
    ]
    table = format_table(
        rows,
        columns=["id", "name", "status", "workflow_id"],
        headers=["ID", "NAME", "STATUS", "WORKFLOW"],
    )
    output(args, result, text=table)


def _stream(args) -> None:  # noqa: ANN001
    from roboflow.adapters import devicesapi
    from roboflow.cli._output import output, output_error

    resolved = _resolve_ws_and_key(args)
    if not resolved:
        return
    ws, api_key = resolved

    try:
        stream = devicesapi.get_device_stream(api_key, ws, args.device_id, args.stream_id)
    except Exception as exc:  # noqa: BLE001
        output_error(args, str(exc), hint=_hint_for(exc), exit_code=_exit_code_for(exc))
        return

    lines = [
        f"Stream: {stream.get('name') or stream.get('id')}",
        f"  ID: {stream.get('id', '')}",
        f"  Status: {stream.get('status') or ''}",
        f"  Workflow: {stream.get('workflow_id') or ''}",
        f"  Pipeline: {stream.get('pipeline_id') or ''}",
        f"  Started: {stream.get('started_at') or ''}",
        f"  Last Event: {stream.get('last_event_at') or ''}",
    ]
    if stream.get("error"):
        lines.append(f"  Error: {stream['error']}")
    output(args, stream, text="\n".join(lines))


def _logs(args) -> None:  # noqa: ANN001
    from roboflow.adapters import devicesapi
    from roboflow.cli._output import output, output_error

    resolved = _resolve_ws_and_key(args)
    if not resolved:
        return
    ws, api_key = resolved

    try:
        result = devicesapi.get_device_logs(
            api_key,
            ws,
            args.device_id,
            start_time=args.start_time,
            end_time=args.end_time,
            service=args.service,
            severity=args.severity,
            limit=args.limit,
            cursor=args.cursor,
        )
    except Exception as exc:  # noqa: BLE001
        output_error(args, str(exc), hint=_hint_for(exc), exit_code=_exit_code_for(exc))
        return

    logs = result.get("data", [])
    text_lines = [
        f"{log.get('timestamp', '')}  [{log.get('severity', '')}]  {log.get('service', '')}  {log.get('message', '')}"
        for log in logs
    ]
    if not text_lines:
        text_lines = ["(no logs)"]
    output(args, result, text="\n".join(text_lines))


def _telemetry(args) -> None:  # noqa: ANN001
    from roboflow.adapters import devicesapi
    from roboflow.cli._output import output, output_error

    resolved = _resolve_ws_and_key(args)
    if not resolved:
        return
    ws, api_key = resolved

    try:
        result = devicesapi.get_device_telemetry(api_key, ws, args.device_id, time_period=args.time_period)
    except Exception as exc:  # noqa: BLE001
        output_error(args, str(exc), hint=_hint_for(exc), exit_code=_exit_code_for(exc))
        return

    buckets = result.get("buckets", [])
    lines = [
        f"Time period: {result.get('time_period', '')}  "
        f"Bucket: {result.get('bucket_interval', '')}  "
        f"Buckets: {len(buckets)}"
    ]
    output(args, result, text="\n".join(lines))


def _events(args) -> None:  # noqa: ANN001
    from roboflow.adapters import devicesapi
    from roboflow.cli._output import output, output_error

    resolved = _resolve_ws_and_key(args)
    if not resolved:
        return
    ws, api_key = resolved

    try:
        result = devicesapi.get_device_events(
            api_key,
            ws,
            args.device_id,
            entity_type=args.entity_type,
            entity_id=args.entity_id,
            event=args.event,
            start_time=args.start_time,
            end_time=args.end_time,
            limit=args.limit,
            cursor=args.cursor,
            direction=args.direction,
        )
    except Exception as exc:  # noqa: BLE001
        output_error(args, str(exc), hint=_hint_for(exc), exit_code=_exit_code_for(exc))
        return

    events = result.get("data", [])
    text_lines = [
        f"{e.get('server_timestamp', '')}  {e.get('event', '')}  "
        f"{e.get('entity_type', '')}/{e.get('entity_id', '')}  "
        f"{e.get('event_description', '') or ''}"
        for e in events
    ]
    if not text_lines:
        text_lines = ["(no events)"]
    output(args, result, text="\n".join(text_lines))
