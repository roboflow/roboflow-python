"""Workspace commands: list, get, usage, plan, stats."""

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

    # --- workspace usage ---
    usage_p = ws_sub.add_parser("usage", help="Show billing usage report")
    usage_p.set_defaults(func=_workspace_usage)

    # --- workspace plan ---
    plan_p = ws_sub.add_parser("plan", help="Show workspace plan info and limits")
    plan_p.set_defaults(func=_workspace_plan)

    # --- workspace stats ---
    stats_p = ws_sub.add_parser("stats", help="Show annotation/labeling statistics")
    stats_p.set_defaults(func=_workspace_stats)

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


def _resolve_ws_and_key(args: argparse.Namespace):
    """Resolve workspace and API key for workspace subcommands."""
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


def _workspace_usage(args: argparse.Namespace) -> None:
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error

    resolved = _resolve_ws_and_key(args)
    if not resolved:
        return
    ws, api_key = resolved

    try:
        result = rfapi.get_billing_usage(api_key, ws)
    except Exception as exc:
        output_error(args, str(exc), exit_code=3)
        return

    usage = result.get("usage", result)
    lines = ["Billing Usage:"]
    if isinstance(usage, dict):
        for key, val in usage.items():
            lines.append(f"  {key}: {val}")
    else:
        lines.append(f"  {usage}")
    output(args, result, text="\n".join(lines))


def _workspace_plan(args: argparse.Namespace) -> None:
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error

    resolved = _resolve_ws_and_key(args)
    if not resolved:
        return
    _ws, api_key = resolved

    try:
        result = rfapi.get_plan_info(api_key)
    except Exception as exc:
        output_error(args, str(exc), exit_code=3)
        return

    plan = result.get("plan", result)
    lines = ["Plan Info:"]
    if isinstance(plan, dict):
        for key, val in plan.items():
            lines.append(f"  {key}: {val}")
    else:
        lines.append(f"  {plan}")
    output(args, result, text="\n".join(lines))


def _workspace_stats(args: argparse.Namespace) -> None:
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error

    resolved = _resolve_ws_and_key(args)
    if not resolved:
        return
    ws, api_key = resolved

    try:
        result = rfapi.get_labeling_stats(api_key, ws)
    except Exception as exc:
        output_error(args, str(exc), exit_code=3)
        return

    stats = result.get("stats", result)
    lines = ["Labeling Stats:"]
    if isinstance(stats, dict):
        for key, val in stats.items():
            lines.append(f"  {key}: {val}")
    else:
        lines.append(f"  {stats}")
    output(args, result, text="\n".join(lines))
