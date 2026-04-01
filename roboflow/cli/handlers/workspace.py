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
    from roboflow.cli._output import output
    from roboflow.cli._table import format_table
    from roboflow.config import APP_URL, get_conditional_configuration_variable

    workspaces = get_conditional_configuration_variable("workspaces", default={})
    default_ws_url = get_conditional_configuration_variable("RF_WORKSPACE", default=None)

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
    from roboflow.cli._output import output, output_error
    from roboflow.config import load_roboflow_api_key

    workspace_id = args.workspace_id
    api_key = getattr(args, "api_key", None) or load_roboflow_api_key(workspace_id)

    if not api_key:
        output_error(
            args,
            "No API key found.",
            hint="Run 'roboflow auth login' or pass --api-key.",
            exit_code=2,
        )

    workspace_json = rfapi.get_workspace(api_key, workspace_id)
    output(args, workspace_json)
