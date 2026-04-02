"""Workspace commands: list, get."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``workspace`` command group."""
    ws_parser = subparsers.add_parser("workspace", help="Manage workspaces")
    ws_sub = ws_parser.add_subparsers(title="workspace commands", dest="workspace_command")

    # --- workspace list ---
    list_p = ws_sub.add_parser("list", help="List configured workspaces")
    list_p.set_defaults(func=_list_workspaces)

    # --- workspace get ---
    get_p = ws_sub.add_parser("get", help="Get workspace details")
    get_p.add_argument("workspace_id", help="Workspace URL or ID")
    get_p.set_defaults(func=_get_workspace)

    # Default: show help
    ws_parser.set_defaults(func=lambda args: ws_parser.print_help())


def _list_workspaces(args: argparse.Namespace) -> None:
    import os

    from roboflow.cli._output import output
    from roboflow.cli._resolver import resolve_default_workspace
    from roboflow.cli._table import format_table
    from roboflow.config import APP_URL, get_conditional_configuration_variable

    workspaces = get_conditional_configuration_variable("workspaces", default={})
    default_ws_url = get_conditional_configuration_variable("RF_WORKSPACE", default=None)

    # When no workspaces in config, fall back to API using available API key
    if not workspaces:
        api_key = getattr(args, "api_key", None) or os.getenv("ROBOFLOW_API_KEY")
        ws_url = resolve_default_workspace(api_key=api_key)
        if ws_url:
            ws_name = ws_url
            if api_key:
                try:
                    from roboflow.adapters import rfapi

                    ws_json = rfapi.get_workspace(api_key, ws_url)
                    ws_detail = ws_json.get("workspace", ws_json)
                    ws_name = ws_detail.get("name", ws_url)
                except Exception:  # noqa: BLE001
                    pass
            workspaces = {ws_url: {"url": ws_url, "name": ws_name}}
            if not default_ws_url:
                default_ws_url = ws_url

    rows = []
    for w in workspaces.values():
        rows.append(
            {
                "name": w.get("name", ""),
                "url": w.get("url", ""),
                "link": f"{APP_URL}/{w.get('url', '')}",
                "default": "yes" if w.get("url") == default_ws_url else "",
            }
        )

    table = format_table(rows, columns=["name", "url", "default"], headers=["NAME", "ID", "DEFAULT"])
    output(args, rows, text=table)


def _get_workspace(args: argparse.Namespace) -> None:
    from roboflow.adapters import rfapi
    from roboflow.adapters.rfapi import RoboflowError
    from roboflow.cli._output import output, output_error
    from roboflow.config import APP_URL, load_roboflow_api_key

    workspace_id = args.workspace_id
    api_key = getattr(args, "api_key", None) or load_roboflow_api_key(workspace_id)

    if not api_key:
        output_error(
            args,
            "No API key found.",
            hint="Run 'roboflow auth login' or pass --api-key.",
            exit_code=2,
        )
        return  # unreachable, but helps mypy

    try:
        workspace_json = rfapi.get_workspace(api_key, workspace_id)
    except RoboflowError:
        output_error(
            args,
            f"Workspace '{workspace_id}' not found.",
            hint=f"Check the workspace ID and try again. Browse workspaces at {APP_URL}.",
            exit_code=3,
        )
        return  # unreachable, but helps mypy

    # Human-readable text for non-JSON mode
    ws = workspace_json.get("workspace", workspace_json)
    name = ws.get("name", workspace_id)
    members = ws.get("members", 0)
    projects = ws.get("projects", [])
    member_count = members if isinstance(members, int) else len(members)
    project_count = len(projects) if isinstance(projects, list) else projects
    lines = [
        f"Workspace: {name}",
        f"  URL: {workspace_id}",
        f"  Link: {APP_URL}/{workspace_id}",
        f"  Members: {member_count}",
        f"  Projects: {project_count}",
    ]
    output(args, workspace_json, text="\n".join(lines))
