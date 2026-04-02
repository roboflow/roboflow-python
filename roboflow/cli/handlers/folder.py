"""Folder management commands."""

from __future__ import annotations

from typing import Annotated, Optional

import typer

from roboflow.cli._compat import ctx_to_args

folder_app = typer.Typer(help="Manage workspace folders", no_args_is_help=True)


@folder_app.command("list")
def list_folders(ctx: typer.Context) -> None:
    """List folders."""
    args = ctx_to_args(ctx)
    _list_folders(args)


@folder_app.command("get")
def get_folder(
    ctx: typer.Context,
    folder_id: Annotated[str, typer.Argument(help="Folder ID")],
) -> None:
    """Show folder details."""
    args = ctx_to_args(ctx, folder_id=folder_id)
    _get_folder(args)


@folder_app.command("create")
def create_folder(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Folder name")],
    parent: Annotated[Optional[str], typer.Option(help="Parent folder ID")] = None,
    projects: Annotated[Optional[str], typer.Option(help="Comma-separated project IDs")] = None,
) -> None:
    """Create a folder."""
    args = ctx_to_args(ctx, name=name, parent=parent, projects=projects)
    _create_folder(args)


@folder_app.command("update")
def update_folder(
    ctx: typer.Context,
    folder_id: Annotated[str, typer.Argument(help="Folder ID")],
    name: Annotated[Optional[str], typer.Option(help="New folder name")] = None,
) -> None:
    """Update a folder."""
    args = ctx_to_args(ctx, folder_id=folder_id, name=name)
    _update_folder(args)


@folder_app.command("delete")
def delete_folder(
    ctx: typer.Context,
    folder_id: Annotated[str, typer.Argument(help="Folder ID")],
) -> None:
    """Delete a folder."""
    args = ctx_to_args(ctx, folder_id=folder_id)
    _delete_folder(args)


# ---------------------------------------------------------------------------
# Business logic (unchanged from argparse version)
# ---------------------------------------------------------------------------


def _resolve_ws_and_key(args):  # noqa: ANN001
    """Resolve workspace and API key, returning (ws, api_key) or None on error."""
    from roboflow.cli._resolver import resolve_ws_and_key

    return resolve_ws_and_key(args)


def _list_folders(args) -> None:  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error
    from roboflow.cli._table import format_table

    resolved = _resolve_ws_and_key(args)
    if not resolved:
        return
    ws, api_key = resolved

    try:
        result = rfapi.list_folders(api_key, ws)
    except rfapi.RoboflowError as exc:
        # The API returns 404 when there are no folders — treat as empty, not error
        if "Not Found" in str(exc):
            result = {"data": []}
        else:
            output_error(args, str(exc), exit_code=3)
            return
    except Exception as exc:
        output_error(args, str(exc), exit_code=3)
        return

    folders = result.get("data", result.get("groups", result if isinstance(result, list) else []))
    rows = []
    for f in folders:
        projects = f.get("projects", [])
        project_count = len(projects) if isinstance(projects, list) else projects
        rows.append({"name": f.get("name", ""), "id": f.get("id", ""), "projects": str(project_count)})

    table = format_table(rows, columns=["name", "id", "projects"], headers=["NAME", "ID", "PROJECTS"])
    output(args, folders, text=table)


def _get_folder(args) -> None:  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error

    resolved = _resolve_ws_and_key(args)
    if not resolved:
        return
    ws, api_key = resolved

    try:
        result = rfapi.get_folder(api_key, ws, args.folder_id)
    except Exception as exc:
        output_error(args, str(exc), exit_code=3)
        return

    # API returns {"data": [folder_obj]} — extract the first item
    data_list = result.get("data", [])
    folder = data_list[0] if isinstance(data_list, list) and data_list else result.get("group", result)
    lines = [
        f"Folder: {folder.get('name', '')}",
        f"  ID: {folder.get('id', '')}",
    ]
    projects = folder.get("projects", [])
    if isinstance(projects, list):
        lines.append(f"  Projects: {len(projects)}")
        for p in projects:
            if isinstance(p, dict):
                lines.append(f"    - {p.get('name', p.get('id', ''))}")
            else:
                lines.append(f"    - {p}")
    output(args, result, text="\n".join(lines))


def _create_folder(args) -> None:  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error

    resolved = _resolve_ws_and_key(args)
    if not resolved:
        return
    ws, api_key = resolved

    project_ids = None
    if args.projects:
        project_ids = [p.strip() for p in args.projects.split(",")]

    try:
        result = rfapi.create_folder(api_key, ws, args.name, parent_id=args.parent, project_ids=project_ids)
    except Exception as exc:
        output_error(args, str(exc), exit_code=1)
        return

    folder_id = result.get("id", result.get("group", {}).get("id", ""))
    data = {"status": "created", "id": folder_id}
    output(args, data, text=f"Created folder '{args.name}' (id: {folder_id})")


def _update_folder(args) -> None:  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error

    resolved = _resolve_ws_and_key(args)
    if not resolved:
        return
    ws, api_key = resolved

    try:
        rfapi.update_folder(api_key, ws, args.folder_id, name=args.name)
    except Exception as exc:
        output_error(args, str(exc), exit_code=1)
        return

    data = {"status": "updated"}
    output(args, data, text=f"Updated folder '{args.folder_id}'")


def _delete_folder(args) -> None:  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error

    resolved = _resolve_ws_and_key(args)
    if not resolved:
        return
    ws, api_key = resolved

    try:
        rfapi.delete_folder(api_key, ws, args.folder_id)
    except Exception as exc:
        output_error(args, str(exc), exit_code=3)
        return

    data = {"status": "deleted"}
    output(args, data, text=f"Deleted folder '{args.folder_id}'")
