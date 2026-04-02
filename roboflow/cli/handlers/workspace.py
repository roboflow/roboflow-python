"""Workspace commands: list, get, usage, plan, stats."""

from __future__ import annotations

from typing import Annotated

import typer

from roboflow.cli._compat import ctx_to_args

workspace_app = typer.Typer(help="Manage workspaces", no_args_is_help=True)


@workspace_app.command("list")
def list_workspaces(ctx: typer.Context) -> None:
    """List configured workspaces."""
    args = ctx_to_args(ctx)
    _list_workspaces(args)


@workspace_app.command("get")
def get_workspace(
    ctx: typer.Context,
    workspace_id: Annotated[str, typer.Argument(help="Workspace URL or ID")],
) -> None:
    """Get workspace details."""
    args = ctx_to_args(ctx, workspace_id=workspace_id)
    _get_workspace(args)


@workspace_app.command("usage")
def workspace_usage(ctx: typer.Context) -> None:
    """Show billing usage report."""
    args = ctx_to_args(ctx)
    _workspace_usage(args)


@workspace_app.command("plan")
def workspace_plan(ctx: typer.Context) -> None:
    """Show workspace plan info and limits."""
    args = ctx_to_args(ctx)
    _workspace_plan(args)


@workspace_app.command("stats")
def workspace_stats(
    ctx: typer.Context,
    start_date: Annotated[str, typer.Option("--start-date", help="Start date (YYYY-MM-DD)")],
    end_date: Annotated[str, typer.Option("--end-date", help="End date (YYYY-MM-DD)")],
) -> None:
    """Show annotation/labeling statistics."""
    args = ctx_to_args(ctx, start_date=start_date, end_date=end_date)
    _workspace_stats(args)


# ---------------------------------------------------------------------------
# Business logic (unchanged from argparse version)
# ---------------------------------------------------------------------------


def _list_workspaces(args):  # noqa: ANN001
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


def _get_workspace(args):  # noqa: ANN001
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


def _resolve_ws_and_key(args):  # noqa: ANN001
    """Resolve workspace and API key for workspace subcommands."""
    from roboflow.cli._resolver import resolve_ws_and_key

    return resolve_ws_and_key(args)


def _workspace_usage(args):  # noqa: ANN001
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


def _workspace_plan(args):  # noqa: ANN001
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


def _workspace_stats(args):  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error

    resolved = _resolve_ws_and_key(args)
    if not resolved:
        return
    ws, api_key = resolved

    try:
        result = rfapi.get_labeling_stats(api_key, ws, start_date=args.start_date, end_date=args.end_date)
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
