"""Workflow management commands."""

from __future__ import annotations

from typing import Annotated, Optional

import typer

from roboflow.cli._compat import SortedGroup, ctx_to_args

workflow_app = typer.Typer(cls=SortedGroup, help="Manage workflows", no_args_is_help=True)

# ---------------------------------------------------------------------------
# Sub-app for ``workflow version`` subcommands
# ---------------------------------------------------------------------------

_version_app = typer.Typer(cls=SortedGroup, help="Manage workflow versions", no_args_is_help=True)
workflow_app.add_typer(_version_app, name="version")


@workflow_app.command("list")
def list_workflows(ctx: typer.Context) -> None:
    """List workflows in a workspace."""
    args = ctx_to_args(ctx)
    _list_workflows(args)


@workflow_app.command("get")
def get_workflow(
    ctx: typer.Context,
    workflow_url: Annotated[str, typer.Argument(help="Workflow URL or ID")],
) -> None:
    """Show details for a workflow."""
    args = ctx_to_args(ctx, workflow_url=workflow_url)
    _get_workflow(args)


@workflow_app.command("create")
def create_workflow(
    ctx: typer.Context,
    name: Annotated[str, typer.Option("--name", help="Workflow name")],
    definition: Annotated[Optional[str], typer.Option(help="Path to JSON definition file")] = None,
    description: Annotated[Optional[str], typer.Option(help="Workflow description")] = None,
) -> None:
    """Create a new workflow."""
    args = ctx_to_args(ctx, name=name, definition=definition, description=description)
    _create_workflow(args)


@workflow_app.command("update")
def update_workflow(
    ctx: typer.Context,
    workflow_url: Annotated[str, typer.Argument(help="Workflow URL or ID")],
    definition: Annotated[Optional[str], typer.Option(help="Path to JSON definition file")] = None,
) -> None:
    """Update an existing workflow."""
    args = ctx_to_args(ctx, workflow_url=workflow_url, definition=definition)
    _update_workflow(args)


@_version_app.command("list")
def list_workflow_versions(
    ctx: typer.Context,
    workflow_url: Annotated[str, typer.Argument(help="Workflow URL or ID")],
) -> None:
    """List versions of a workflow."""
    args = ctx_to_args(ctx, workflow_url=workflow_url)
    _list_workflow_versions(args)


@workflow_app.command("fork")
def fork_workflow(
    ctx: typer.Context,
    workflow_url: Annotated[str, typer.Argument(help="Workflow URL or ID")],
) -> None:
    """Fork a workflow."""
    args = ctx_to_args(ctx, workflow_url=workflow_url)
    _fork_workflow(args)


@workflow_app.command("build", hidden=True)
def build_workflow(
    ctx: typer.Context,
    prompt: Annotated[str, typer.Argument(help="Natural language prompt describing the workflow")],
) -> None:
    """Build a workflow from a prompt."""
    args = ctx_to_args(ctx, prompt=prompt)
    _stub_build(args)


@workflow_app.command("run", hidden=True)
def run_workflow(
    ctx: typer.Context,
    workflow_url: Annotated[str, typer.Argument(help="Workflow URL or ID")],
    input: Annotated[Optional[str], typer.Option("--input", help="Input file or URL")] = None,
) -> None:
    """Run a workflow."""
    args = ctx_to_args(ctx, workflow_url=workflow_url, input=input)
    _stub_run(args)


@workflow_app.command("deploy", hidden=True)
def deploy_workflow(
    ctx: typer.Context,
    workflow_url: Annotated[str, typer.Argument(help="Workflow URL or ID")],
) -> None:
    """Deploy a workflow."""
    args = ctx_to_args(ctx, workflow_url=workflow_url)
    _stub_deploy(args)


@workflow_app.command("delete")
def delete_workflow(
    ctx: typer.Context,
    workflow_url: Annotated[str, typer.Argument(help="Workflow URL or ID")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation prompt.")] = False,
) -> None:
    """Move a workflow to Trash (30-day retention)."""
    args = ctx_to_args(ctx, workflow_url=workflow_url, yes=yes)
    _delete_workflow(args)


@workflow_app.command("restore")
def restore_workflow_cmd(
    ctx: typer.Context,
    workflow_url: Annotated[str, typer.Argument(help="Workflow URL or ID")],
) -> None:
    """Restore a workflow from Trash."""
    args = ctx_to_args(ctx, workflow_url=workflow_url)
    _restore_workflow(args)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_workspace_and_key(args):  # noqa: ANN001
    """Return (workspace_url, api_key) or call output_error and return None."""
    from roboflow.cli._resolver import resolve_ws_and_key

    return resolve_ws_and_key(args)


def _read_definition_file(args):  # noqa: ANN001
    """Read and parse a JSON definition file. Returns the parsed dict, or None if no file given.

    Calls output_error and returns False on failure.
    """
    import json
    import os

    from roboflow.cli._output import output_error

    if not args.definition:
        return None

    if not os.path.isfile(args.definition):
        output_error(args, f"File not found: {args.definition}", hint="Provide a valid JSON file path.")
        return False

    with open(args.definition) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as exc:
            output_error(args, f"Invalid JSON in {args.definition}: {exc}")
            return False


# ---------------------------------------------------------------------------
# Implemented commands
# ---------------------------------------------------------------------------


def _list_workflows(args) -> None:  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error
    from roboflow.cli._table import format_table

    resolved = _resolve_workspace_and_key(args)
    if resolved is None:
        return
    workspace_url, api_key = resolved

    try:
        data = rfapi.list_workflows(api_key, workspace_url)
    except rfapi.RoboflowError as exc:
        output_error(args, str(exc), exit_code=3)
        return

    workflows = data if isinstance(data, list) else data.get("workflows", [])

    table = format_table(
        workflows,
        columns=["name", "url", "status"],
        headers=["NAME", "URL", "STATUS"],
    )
    output(args, workflows, text=table)


def _get_workflow(args) -> None:  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error

    resolved = _resolve_workspace_and_key(args)
    if resolved is None:
        return
    workspace_url, api_key = resolved

    try:
        data = rfapi.get_workflow(api_key, workspace_url, args.workflow_url)
    except rfapi.RoboflowError as exc:
        output_error(args, str(exc), exit_code=3)
        return

    workflow = data.get("workflow", data) if isinstance(data, dict) else data

    lines = []
    if isinstance(workflow, dict):
        field_map = [
            ("Name", "name"),
            ("URL", "url"),
            ("Description", "description"),
            ("Blocks", "blockCount"),
        ]
        for label, key in field_map:
            if key in workflow:
                lines.append(f"  {label:14s} {workflow[key]}")
    text = "\n".join(lines) if lines else "(no workflow details)"

    output(args, data, text=text)


def _create_workflow(args) -> None:  # noqa: ANN001
    import json as _json

    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error

    resolved = _resolve_workspace_and_key(args)
    if resolved is None:
        return
    workspace_url, api_key = resolved

    definition = _read_definition_file(args)
    if definition is False:
        return

    # The API expects config/template as JSON strings.
    config = _json.dumps(definition) if definition is not None else "{}"
    template = "{}"

    try:
        data = rfapi.create_workflow(
            api_key,
            workspace_url,
            name=args.name,
            config=config,
            template=template,
        )
    except rfapi.RoboflowError as exc:
        output_error(args, str(exc))
        return

    text = f"Created workflow: {args.name}"
    output(args, data, text=text)


def _update_workflow(args) -> None:  # noqa: ANN001
    import json as _json

    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error

    resolved = _resolve_workspace_and_key(args)
    if resolved is None:
        return
    workspace_url, api_key = resolved

    definition = _read_definition_file(args)
    if definition is False:
        return

    # Fetch the existing workflow to get required id/name/url fields.
    try:
        existing = rfapi.get_workflow(api_key, workspace_url, args.workflow_url)
    except rfapi.RoboflowError as exc:
        output_error(args, str(exc), exit_code=3)
        return

    wf = existing.get("workflow", existing) if isinstance(existing, dict) else existing
    if not isinstance(wf, dict):
        output_error(args, "Unexpected response from API when fetching workflow.")
        return

    workflow_id = wf.get("id", "")
    workflow_name = wf.get("name", "")
    workflow_url_slug = wf.get("url", args.workflow_url)

    # Merge: use new definition as config if provided, otherwise keep existing.
    if definition is not None:
        config = _json.dumps(definition) if not isinstance(definition, str) else definition
    else:
        config = wf.get("config", "{}")
        if not isinstance(config, str):
            config = _json.dumps(config)

    try:
        data = rfapi.update_workflow(
            api_key,
            workspace_url,
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            workflow_url=workflow_url_slug,
            config=config,
        )
    except rfapi.RoboflowError as exc:
        output_error(args, str(exc))
        return

    text = f"Updated workflow: {args.workflow_url}"
    output(args, data, text=text)


def _list_workflow_versions(args) -> None:  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error
    from roboflow.cli._table import format_table

    resolved = _resolve_workspace_and_key(args)
    if resolved is None:
        return
    workspace_url, api_key = resolved

    try:
        data = rfapi.list_workflow_versions(api_key, workspace_url, args.workflow_url)
    except rfapi.RoboflowError as exc:
        output_error(args, str(exc), exit_code=3)
        return

    versions = data if isinstance(data, list) else data.get("versions", [])

    table = format_table(
        versions,
        columns=["version", "created"],
        headers=["VERSION", "CREATED"],
    )
    output(args, versions, text=table)


def _fork_workflow(args) -> None:  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error

    resolved = _resolve_workspace_and_key(args)
    if resolved is None:
        return
    workspace_url, api_key = resolved

    # Parse workflow_url: could be "workflow-slug" or "source-ws/workflow-slug".
    parts = args.workflow_url.strip("/").split("/")
    if len(parts) == 2:
        source_workspace = parts[0]
        source_workflow = parts[1]
    else:
        # Default: source workspace is the current workspace.
        source_workspace = workspace_url
        source_workflow = parts[0]

    try:
        data = rfapi.fork_workflow(
            api_key,
            workspace_url,
            source_workspace=source_workspace,
            source_workflow=source_workflow,
        )
    except rfapi.RoboflowError as exc:
        output_error(args, str(exc))
        return

    # Extract the forked workflow URL from potentially nested response
    new_url = ""
    if isinstance(data, dict):
        wf = data.get("workflow", data)
        if isinstance(wf, dict):
            new_url = str(wf.get("url", wf.get("workflow_url", "")))
        else:
            new_url = str(wf) if wf else ""
    result = {"status": "forked", "source": args.workflow_url, "new_url": new_url}
    text = f"Forked workflow: {args.workflow_url} -> {new_url}"
    output(args, result, text=text)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


def _stub_build(args) -> None:  # noqa: ANN001
    from roboflow.cli._output import output_error

    output_error(
        args,
        "This command is not yet implemented.",
        hint="Requires Roboflow Agent API. Coming in a future release.",
    )


def _stub_run(args) -> None:  # noqa: ANN001
    from roboflow.cli._output import output_error

    output_error(
        args,
        "This command is not yet implemented.",
        hint="Requires inference_sdk integration. Coming in a future release.",
    )


def _stub_deploy(args) -> None:  # noqa: ANN001
    from roboflow.cli._output import output_error

    output_error(
        args,
        "This command is not yet implemented.",
        hint="Coming in a future release.",
    )


def _delete_workflow(args) -> None:  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error

    resolved = _resolve_workspace_and_key(args)
    if resolved is None:
        return
    workspace_url, api_key = resolved

    if not getattr(args, "yes", False) and not getattr(args, "json", False):
        confirmed = typer.confirm(
            f"Move workflow '{workspace_url}/{args.workflow_url}' to Trash? "
            "(Retained for 30 days.)",
            default=False,
        )
        if not confirmed:
            output(args, {"cancelled": True}, text="Cancelled.")
            return

    try:
        data = rfapi.delete_workflow(api_key, workspace_url, args.workflow_url)
    except rfapi.RoboflowError as exc:
        output_error(
            args,
            str(exc),
            hint="Check your API key has 'workflow:update' scope on this workspace.",
            exit_code=3,
        )
        return

    output(
        args,
        data,
        text=f"Moved {workspace_url}/{args.workflow_url} to Trash (30-day retention).",
    )


def _restore_workflow(args) -> None:  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error

    resolved = _resolve_workspace_and_key(args)
    if resolved is None:
        return
    workspace_url, api_key = resolved

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

    workflows = trash.get("sections", {}).get("workflows", [])
    # Match on URL first, fall back to id for callers who pass a Firestore id.
    match = next(
        (w for w in workflows if w.get("url") == args.workflow_url or w.get("id") == args.workflow_url),
        None,
    )
    if not match:
        output_error(
            args,
            f"Workflow '{workspace_url}/{args.workflow_url}' is not in Trash.",
            hint="Run 'roboflow trash list' to see what can be restored.",
            exit_code=3,
        )
        return

    try:
        data = rfapi.restore_trash_item(api_key, workspace_url, "workflow", match["id"])
    except rfapi.RoboflowError as exc:
        output_error(
            args,
            str(exc),
            hint="Check your API key has 'project:update' scope on this workspace.",
            exit_code=3,
        )
        return

    output(args, data, text=f"Restored {workspace_url}/{args.workflow_url} from Trash.")
