"""Workflow management commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``workflow`` command group."""
    wf_parser = subparsers.add_parser("workflow", help="Manage workflows")
    wf_subs = wf_parser.add_subparsers(title="workflow commands", dest="workflow_command")

    # --- workflow list ---
    list_p = wf_subs.add_parser("list", help="List workflows in a workspace")
    list_p.set_defaults(func=_list_workflows)

    # --- workflow get ---
    get_p = wf_subs.add_parser("get", help="Show details for a workflow")
    get_p.add_argument("workflow_url", help="Workflow URL or ID")
    get_p.set_defaults(func=_get_workflow)

    # --- workflow create ---
    create_p = wf_subs.add_parser("create", help="Create a new workflow")
    create_p.add_argument("--name", required=True, help="Workflow name")
    create_p.add_argument("--definition", help="Path to JSON definition file")
    create_p.add_argument("--description", default=None, help="Workflow description")
    create_p.set_defaults(func=_create_workflow)

    # --- workflow update ---
    update_p = wf_subs.add_parser("update", help="Update an existing workflow")
    update_p.add_argument("workflow_url", help="Workflow URL or ID")
    update_p.add_argument("--definition", help="Path to JSON definition file")
    update_p.set_defaults(func=_update_workflow)

    # --- workflow version ---
    version_p = wf_subs.add_parser("version", help="Manage workflow versions")
    version_subs = version_p.add_subparsers(title="workflow version commands", dest="workflow_version_command")
    version_list_p = version_subs.add_parser("list", help="List versions of a workflow")
    version_list_p.add_argument("workflow_url", help="Workflow URL or ID")
    version_list_p.set_defaults(func=_list_workflow_versions)
    version_p.set_defaults(func=lambda args: version_p.print_help())

    # --- workflow fork ---
    fork_p = wf_subs.add_parser("fork", help="Fork a workflow")
    fork_p.add_argument("workflow_url", help="Workflow URL or ID")
    fork_p.set_defaults(func=_fork_workflow)

    # --- workflow build (stub) ---
    build_p = wf_subs.add_parser("build", help="Build a workflow from a prompt")
    build_p.add_argument("prompt", help="Natural language prompt describing the workflow")
    build_p.set_defaults(func=_stub_build)

    # --- workflow run (stub) ---
    run_p = wf_subs.add_parser("run", help="Run a workflow")
    run_p.add_argument("workflow_url", help="Workflow URL or ID")
    run_p.add_argument("--input", dest="input", help="Input file or URL")
    run_p.set_defaults(func=_stub_run)

    # --- workflow deploy (stub) ---
    deploy_p = wf_subs.add_parser("deploy", help="Deploy a workflow")
    deploy_p.add_argument("workflow_url", help="Workflow URL or ID")
    deploy_p.set_defaults(func=_stub_deploy)

    # Default
    wf_parser.set_defaults(func=lambda args: wf_parser.print_help())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_workspace_and_key(args: argparse.Namespace):
    """Return (workspace_url, api_key) or call output_error and return None."""
    from roboflow.cli._output import output_error
    from roboflow.cli._resolver import resolve_default_workspace
    from roboflow.config import load_roboflow_api_key

    workspace_url = args.workspace
    if not workspace_url:
        workspace_url = resolve_default_workspace(api_key=args.api_key)

    if not workspace_url:
        output_error(args, "No workspace specified.", hint="Use --workspace or run 'roboflow auth login'.")
        return None

    api_key = args.api_key or load_roboflow_api_key(workspace_url)
    if not api_key:
        output_error(args, "No API key found.", hint="Set ROBOFLOW_API_KEY or run 'roboflow auth login'.", exit_code=2)
        return None

    return workspace_url, api_key


def _read_definition_file(args: argparse.Namespace):
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


def _list_workflows(args: argparse.Namespace) -> None:
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


def _get_workflow(args: argparse.Namespace) -> None:
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


def _create_workflow(args: argparse.Namespace) -> None:
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error

    resolved = _resolve_workspace_and_key(args)
    if resolved is None:
        return
    workspace_url, api_key = resolved

    definition = _read_definition_file(args)
    if definition is False:
        return

    try:
        data = rfapi.create_workflow(
            api_key,
            workspace_url,
            name=args.name,
            definition=definition,
            description=args.description,
        )
    except rfapi.RoboflowError as exc:
        output_error(args, str(exc))
        return

    text = f"Created workflow: {args.name}"
    output(args, data, text=text)


def _update_workflow(args: argparse.Namespace) -> None:
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error

    resolved = _resolve_workspace_and_key(args)
    if resolved is None:
        return
    workspace_url, api_key = resolved

    definition = _read_definition_file(args)
    if definition is False:
        return

    try:
        data = rfapi.update_workflow(
            api_key,
            workspace_url,
            workflow_url=args.workflow_url,
            definition=definition,
        )
    except rfapi.RoboflowError as exc:
        output_error(args, str(exc))
        return

    text = f"Updated workflow: {args.workflow_url}"
    output(args, data, text=text)


def _list_workflow_versions(args: argparse.Namespace) -> None:
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


def _fork_workflow(args: argparse.Namespace) -> None:
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error

    resolved = _resolve_workspace_and_key(args)
    if resolved is None:
        return
    workspace_url, api_key = resolved

    try:
        data = rfapi.fork_workflow(api_key, workspace_url, args.workflow_url)
    except rfapi.RoboflowError as exc:
        output_error(args, str(exc))
        return

    new_url = data.get("url", data.get("workflow_url", "")) if isinstance(data, dict) else ""
    result = {"status": "forked", "source": args.workflow_url, "new_url": new_url}
    text = f"Forked workflow: {args.workflow_url} -> {new_url}"
    output(args, result, text=text)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


def _stub_build(args: argparse.Namespace) -> None:
    from roboflow.cli._output import output_error

    output_error(
        args,
        "This command is not yet implemented.",
        hint="Requires Roboflow Agent API. Coming in a future release.",
    )


def _stub_run(args: argparse.Namespace) -> None:
    from roboflow.cli._output import output_error

    output_error(
        args,
        "This command is not yet implemented.",
        hint="Requires inference_sdk integration. Coming in a future release.",
    )


def _stub_deploy(args: argparse.Namespace) -> None:
    from roboflow.cli._output import output_error

    output_error(
        args,
        "This command is not yet implemented.",
        hint="Coming in a future release.",
    )
