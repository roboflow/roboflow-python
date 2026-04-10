"""Vision events commands: write, query, list use cases, and upload images."""

from __future__ import annotations

from typing import Annotated, Optional

import typer

from roboflow.cli._compat import SortedGroup, ctx_to_args

vision_events_app = typer.Typer(
    help="Create, query, and manage vision events.",
    cls=SortedGroup,
    no_args_is_help=True,
)


def _resolve(args):  # noqa: ANN001
    """Return api_key or call output_error and return None."""
    from roboflow.cli._resolver import resolve_ws_and_key

    resolved = resolve_ws_and_key(args)
    if resolved is None:
        return None
    _ws, api_key = resolved
    return api_key


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------


@vision_events_app.command("write")
def write(
    ctx: typer.Context,
    event: Annotated[str, typer.Argument(help="JSON string of the event payload")],
) -> None:
    """Create a single vision event."""
    args = ctx_to_args(ctx, event=event)
    _write(args)


def _write(args) -> None:  # noqa: ANN001
    import json

    from roboflow.adapters import vision_events_api
    from roboflow.adapters.rfapi import RoboflowError
    from roboflow.cli._output import output, output_error

    api_key = _resolve(args)
    if api_key is None:
        return

    try:
        event = json.loads(args.event)
    except (json.JSONDecodeError, TypeError) as exc:
        output_error(args, f"Invalid JSON: {exc}", hint="Pass a valid JSON string.")
        return

    try:
        result = vision_events_api.write_event(api_key, event)
    except RoboflowError as exc:
        output_error(args, str(exc))
        return

    output(args, result, text=f"Created event {result.get('eventId', '')}")


# ---------------------------------------------------------------------------
# write-batch
# ---------------------------------------------------------------------------


@vision_events_app.command("write-batch")
def write_batch(
    ctx: typer.Context,
    events: Annotated[str, typer.Argument(help="JSON string of the events array")],
) -> None:
    """Create multiple vision events in a single request."""
    args = ctx_to_args(ctx, events=events)
    _write_batch(args)


def _write_batch(args) -> None:  # noqa: ANN001
    import json

    from roboflow.adapters import vision_events_api
    from roboflow.adapters.rfapi import RoboflowError
    from roboflow.cli._output import output, output_error

    api_key = _resolve(args)
    if api_key is None:
        return

    try:
        events = json.loads(args.events)
    except (json.JSONDecodeError, TypeError) as exc:
        output_error(args, f"Invalid JSON: {exc}", hint="Pass a valid JSON array string.")
        return

    try:
        result = vision_events_api.write_batch(api_key, events)
    except RoboflowError as exc:
        output_error(args, str(exc))
        return

    output(args, result, text=f"Created {result.get('created', 0)} event(s)")


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------


@vision_events_app.command("query")
def query(
    ctx: typer.Context,
    use_case: Annotated[str, typer.Argument(help="Use case identifier to query")],
    event_type: Annotated[Optional[str], typer.Option("-t", "--event-type", help="Filter by event type")] = None,
    start_time: Annotated[Optional[str], typer.Option("--start", help="ISO 8601 start time")] = None,
    end_time: Annotated[Optional[str], typer.Option("--end", help="ISO 8601 end time")] = None,
    limit: Annotated[Optional[int], typer.Option("-l", "--limit", help="Max events to return")] = None,
    cursor: Annotated[Optional[str], typer.Option("--cursor", help="Pagination cursor")] = None,
) -> None:
    """Query vision events with filters and pagination."""
    args = ctx_to_args(
        ctx,
        use_case=use_case,
        event_type=event_type,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
        cursor=cursor,
    )
    _query(args)


def _query(args) -> None:  # noqa: ANN001
    from roboflow.adapters import vision_events_api
    from roboflow.adapters.rfapi import RoboflowError
    from roboflow.cli._output import output, output_error

    api_key = _resolve(args)
    if api_key is None:
        return

    payload = {"useCaseId": args.use_case}
    if args.event_type is not None:
        payload["eventType"] = args.event_type
    if args.start_time is not None:
        payload["startTime"] = args.start_time
    if args.end_time is not None:
        payload["endTime"] = args.end_time
    if args.limit is not None:
        payload["limit"] = args.limit
    if args.cursor is not None:
        payload["cursor"] = args.cursor

    try:
        result = vision_events_api.query(api_key, payload)
    except RoboflowError as exc:
        output_error(args, str(exc))
        return

    events = result.get("events", [])
    lines = [f"Found {len(events)} event(s)."]
    for evt in events:
        lines.append(f"  {evt.get('eventId', '')} [{evt.get('eventType', '')}]")
    if result.get("nextCursor"):
        lines.append(f"\nNext page: --cursor {result['nextCursor']}")

    output(args, result, text="\n".join(lines))


# ---------------------------------------------------------------------------
# use-cases
# ---------------------------------------------------------------------------


@vision_events_app.command("use-cases")
def use_cases(
    ctx: typer.Context,
    status: Annotated[Optional[str], typer.Option("-s", "--status", help="Filter by status (active, inactive)")] = None,
) -> None:
    """List vision event use cases for the workspace."""
    args = ctx_to_args(ctx, status=status)
    _use_cases(args)


def _use_cases(args) -> None:  # noqa: ANN001
    from roboflow.adapters import vision_events_api
    from roboflow.adapters.rfapi import RoboflowError
    from roboflow.cli._output import output, output_error

    api_key = _resolve(args)
    if api_key is None:
        return

    try:
        result = vision_events_api.list_use_cases(api_key, status=args.status)
    except RoboflowError as exc:
        output_error(args, str(exc))
        return

    items = result.get("useCases") or result.get("solutions", [])
    lines = [f"{len(items)} use case(s):"]
    for uc in items:
        name = uc.get("name", uc.get("id", ""))
        if uc.get("eventCount") is not None:
            detail = f" ({uc['eventCount']} events)"
        elif uc.get("status"):
            detail = f" [{uc['status']}]"
        else:
            detail = ""
        lines.append(f"  {name}{detail}")

    output(args, result, text="\n".join(lines))


# ---------------------------------------------------------------------------
# upload-image
# ---------------------------------------------------------------------------


@vision_events_app.command("upload-image")
def upload_image(
    ctx: typer.Context,
    image: Annotated[str, typer.Argument(help="Path to the image file")],
    name: Annotated[Optional[str], typer.Option("-n", "--name", help="Custom image name")] = None,
    metadata: Annotated[
        Optional[str],
        typer.Option("-M", "--metadata", help='JSON string of metadata (e.g. \'{"camera_id":"cam001"}\')'),
    ] = None,
) -> None:
    """Upload an image for use in vision events."""
    args = ctx_to_args(ctx, image=image, name=name, metadata=metadata)
    _upload_image(args)


def _upload_image(args) -> None:  # noqa: ANN001
    import json

    from roboflow.adapters import vision_events_api
    from roboflow.adapters.rfapi import RoboflowError
    from roboflow.cli._output import output, output_error

    api_key = _resolve(args)
    if api_key is None:
        return

    try:
        parsed_metadata = json.loads(args.metadata) if args.metadata else None
    except (json.JSONDecodeError, TypeError) as exc:
        output_error(args, f"Invalid metadata JSON: {exc}", hint="Pass a valid JSON string.")
        return

    try:
        result = vision_events_api.upload_image(
            api_key,
            image_path=args.image,
            name=args.name,
            metadata=parsed_metadata,
        )
    except RoboflowError as exc:
        output_error(args, str(exc))
        return

    output(args, result, text=f"Uploaded image: sourceId={result.get('sourceId', '')}")
