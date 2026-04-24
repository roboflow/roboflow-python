"""Trash management commands: list, empty, delete-immediately."""

from __future__ import annotations

from typing import Annotated, Optional

import typer

from roboflow.cli._compat import SortedGroup, ctx_to_args

trash_app = typer.Typer(cls=SortedGroup, help="Manage items in Trash", no_args_is_help=True)


@trash_app.command("list")
def list_trash_cmd(ctx: typer.Context) -> None:
    """List projects, versions, and workflows currently in Trash."""
    args = ctx_to_args(ctx)
    _list_trash(args)


@trash_app.command("empty")
def empty_trash_cmd(
    ctx: typer.Context,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation prompt.")] = False,
) -> None:
    """Permanently delete everything in Trash. Cannot be undone."""
    args = ctx_to_args(ctx, yes=yes)
    _empty_trash(args)


@trash_app.command("delete")
def delete_immediately_cmd(
    ctx: typer.Context,
    item_type: Annotated[str, typer.Argument(help="dataset, version, or workflow")],
    item_id: Annotated[str, typer.Argument(help="Firestore id of the item in Trash")],
    parent_id: Annotated[
        Optional[str],
        typer.Option("--parent-id", help="Parent dataset id (required for versions)."),
    ] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation prompt.")] = False,
) -> None:
    """Permanently delete a single Trash item. Cannot be undone."""
    args = ctx_to_args(ctx, item_type=item_type, item_id=item_id, parent_id=parent_id, yes=yes)
    _delete_immediately(args)


# ---------------------------------------------------------------------------
# Business logic
# ---------------------------------------------------------------------------


def _resolve_workspace(args):
    from roboflow.cli._output import output_error
    from roboflow.cli._resolver import resolve_default_workspace
    from roboflow.config import load_roboflow_api_key

    workspace_url = args.workspace or resolve_default_workspace(api_key=args.api_key)
    if not workspace_url:
        output_error(args, "No workspace specified.", hint="Use --workspace or run 'roboflow auth login'.")
        return None, None

    api_key = args.api_key or load_roboflow_api_key(workspace_url)
    if not api_key:
        output_error(
            args,
            "No API key found.",
            hint="Set ROBOFLOW_API_KEY or run 'roboflow auth login'.",
            exit_code=2,
        )
        return None, None

    return workspace_url, api_key


def _list_trash(args):  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error
    from roboflow.cli._table import format_table

    workspace_url, api_key = _resolve_workspace(args)
    if not workspace_url:
        return

    try:
        trash = rfapi.list_trash(api_key, workspace_url)
    except rfapi.RoboflowError as exc:
        output_error(args, str(exc), exit_code=3)
        return

    items = trash.get("items", [])
    rows = []
    for item in items:
        name = item.get("name", "")
        if item.get("type") == "version":
            parent = item.get("parentName") or item.get("parentUrl") or ""
            name = f"{parent} — {name} (v{item.get('id', '')})"
        rows.append(
            {
                "type": item.get("type", ""),
                "id": item.get("id", ""),
                "name": name,
                "deletedAt": item.get("deletedAt", ""),
                "scheduledCleanupAt": item.get("scheduledCleanupAt", ""),
                "deletedBy": item.get("deletedByName") or item.get("deletedBy", ""),
            }
        )

    table = format_table(
        rows,
        columns=["type", "id", "name", "deletedAt", "scheduledCleanupAt", "deletedBy"],
        headers=["TYPE", "ID", "NAME", "DELETED", "CLEANUP_AT", "BY"],
    )
    if not rows:
        table = "(Trash is empty)"
    output(args, trash, text=table)


def _empty_trash(args):  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error

    workspace_url, api_key = _resolve_workspace(args)
    if not workspace_url:
        return

    if not getattr(args, "yes", False) and not getattr(args, "json", False):
        import typer as _typer

        confirmed = _typer.confirm(
            f"Permanently delete ALL items in '{workspace_url}' Trash? This cannot be undone.",
            default=False,
        )
        if not confirmed:
            output(args, {"cancelled": True}, text="Cancelled.")
            return

    try:
        data = rfapi.empty_trash(api_key, workspace_url)
    except rfapi.RoboflowError as exc:
        output_error(args, str(exc), exit_code=3)
        return

    output(
        args,
        data,
        text=f"Emptying Trash — {data.get('dispatched', 0)} cleanup tasks dispatched.",
    )


def _delete_immediately(args):  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error

    workspace_url, api_key = _resolve_workspace(args)
    if not workspace_url:
        return

    if not getattr(args, "yes", False) and not getattr(args, "json", False):
        import typer as _typer

        confirmed = _typer.confirm(
            f"Permanently delete {args.item_type} '{args.item_id}'? This cannot be undone.",
            default=False,
        )
        if not confirmed:
            output(args, {"cancelled": True}, text="Cancelled.")
            return

    try:
        data = rfapi.trash_delete_immediately(api_key, workspace_url, args.item_type, args.item_id, args.parent_id)
    except rfapi.RoboflowError as exc:
        output_error(args, str(exc), exit_code=3)
        return

    output(args, data, text=f"Permanently deleted {args.item_type} {args.item_id}.")
