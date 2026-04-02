"""Folder management commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``folder`` command group."""
    folder_parser = subparsers.add_parser("folder", help="Manage workspace folders")
    folder_subs = folder_parser.add_subparsers(title="folder commands", dest="folder_command")

    # --- folder list ---
    list_p = folder_subs.add_parser("list", help="List folders")
    list_p.set_defaults(func=_list_folders)

    # --- folder get ---
    get_p = folder_subs.add_parser("get", help="Show folder details")
    get_p.add_argument("folder_id", help="Folder ID")
    get_p.set_defaults(func=_get_folder)

    # --- folder create ---
    create_p = folder_subs.add_parser("create", help="Create a folder")
    create_p.add_argument("name", help="Folder name")
    create_p.add_argument("--parent", dest="parent", default=None, help="Parent folder ID")
    create_p.add_argument("--projects", dest="projects", default=None, help="Comma-separated project IDs")
    create_p.set_defaults(func=_create_folder)

    # --- folder update ---
    update_p = folder_subs.add_parser("update", help="Update a folder")
    update_p.add_argument("folder_id", help="Folder ID")
    update_p.add_argument("--name", help="New folder name")
    update_p.set_defaults(func=_update_folder)

    # --- folder delete ---
    delete_p = folder_subs.add_parser("delete", help="Delete a folder")
    delete_p.add_argument("folder_id", help="Folder ID")
    delete_p.set_defaults(func=_delete_folder)

    # Default
    folder_parser.set_defaults(func=lambda args: folder_parser.print_help())


def _resolve_ws_and_key(args: argparse.Namespace):
    """Resolve workspace and API key, returning (ws, api_key) or None on error."""
    from roboflow.cli._output import output_error
    from roboflow.cli._resolver import resolve_default_workspace
    from roboflow.config import load_roboflow_api_key

    ws = args.workspace or resolve_default_workspace(api_key=args.api_key)
    if not ws:
        output_error(args, "No workspace specified.", hint="Use --workspace or run 'roboflow auth login'.", exit_code=2)
        return None

    api_key = args.api_key or load_roboflow_api_key(ws)
    if not api_key:
        output_error(args, "No API key found.", hint="Set ROBOFLOW_API_KEY or run 'roboflow auth login'.", exit_code=2)
        return None

    return ws, api_key


def _list_folders(args: argparse.Namespace) -> None:
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error
    from roboflow.cli._table import format_table

    resolved = _resolve_ws_and_key(args)
    if not resolved:
        return
    ws, api_key = resolved

    try:
        result = rfapi.list_folders(api_key, ws)
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


def _get_folder(args: argparse.Namespace) -> None:
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


def _create_folder(args: argparse.Namespace) -> None:
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


def _update_folder(args: argparse.Namespace) -> None:
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


def _delete_folder(args: argparse.Namespace) -> None:
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error

    resolved = _resolve_ws_and_key(args)
    if not resolved:
        return
    ws, api_key = resolved

    try:
        rfapi.delete_folder(api_key, ws, args.folder_id)
    except Exception as exc:
        output_error(args, str(exc), exit_code=1)
        return

    data = {"status": "deleted"}
    output(args, data, text=f"Deleted folder '{args.folder_id}'")
