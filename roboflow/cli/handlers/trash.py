"""Trash management commands.

Only `list` is exposed here — permanent-delete actions (empty Trash, delete a
single Trash item immediately) destroy data irrecoverably and are available
only through the web UI's Trash view. Items left in Trash are cleaned up
automatically after 30 days.
"""

from __future__ import annotations

import typer

from roboflow.cli._compat import SortedGroup, ctx_to_args

trash_app = typer.Typer(cls=SortedGroup, help="Manage items in Trash", no_args_is_help=True)


@trash_app.command("list")
def list_trash_cmd(ctx: typer.Context) -> None:
    """List projects, versions, and workflows currently in Trash."""
    args = ctx_to_args(ctx)
    _list_trash(args)


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
        output_error(
            args,
            str(exc),
            hint="Check your API key has 'project:read' scope on this workspace.",
            exit_code=3,
        )
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
