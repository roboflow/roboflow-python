"""Workflow management commands."""

from __future__ import annotations

from typing import Annotated, Any, Callable, Dict, List, Optional

import typer

from roboflow.cli._compat import SortedGroup, ctx_to_args

workflow_app = typer.Typer(cls=SortedGroup, help="Manage workflows", no_args_is_help=True)

# ---------------------------------------------------------------------------
# Sub-app for ``workflow version`` subcommands
# ---------------------------------------------------------------------------

_version_app = typer.Typer(cls=SortedGroup, help="Manage workflow versions", no_args_is_help=True)
workflow_app.add_typer(_version_app, name="version")

# ---------------------------------------------------------------------------
# Sub-apps for ``workflow deploy-gates`` subcommands
# ---------------------------------------------------------------------------

_deploy_gates_app = typer.Typer(
    cls=SortedGroup,
    help="Deploy gates: checks a saved version must pass before it publishes",
    no_args_is_help=True,
)
workflow_app.add_typer(_deploy_gates_app, name="deploy-gates")

_eval_gate_app = typer.Typer(cls=SortedGroup, help="Gate a workflow on a Workflow Eval", no_args_is_help=True)
_deploy_gates_app.add_typer(_eval_gate_app, name="eval")

_approval_gate_app = typer.Typer(
    cls=SortedGroup,
    help="Gate a workflow on manual approval (approvers decide in the Roboflow app)",
    no_args_is_help=True,
)
_deploy_gates_app.add_typer(_approval_gate_app, name="approval")

WorkflowArg = Annotated[
    str, typer.Argument(help="Workflow URL or ID (resolved with a GET, which needs the workflow:read scope)")
]
# Named to sort after workflow_url: SortedGroup orders positional arguments alphabetically.
VersionArg = Annotated[str, typer.Argument(help="Workflow version ID")]
ThresholdOpt = Annotated[
    Optional[List[str]],
    typer.Option("--threshold", help="CHECK_ID=RATE minimum pass rate, 0 < RATE <= 1 (repeatable)"),
]
ExpectedUpdatedAtOpt = Annotated[
    Optional[str],
    typer.Option(
        "--expected-updated-at",
        help="The gate's updatedAt as you read it ('null' if it had none); refuses the write if it changed since",
    ),
]
WaitOpt = Annotated[bool, typer.Option("--wait", help="Poll until no gate is pending; exit 1 if not publishable")]
TimeoutOpt = Annotated[float, typer.Option("--timeout", help="Seconds to wait before giving up")]
IntervalOpt = Annotated[float, typer.Option("--interval", help="Seconds between polls")]
IdempotencyOpt = Annotated[
    Optional[str], typer.Option("--idempotency-key", help="UUID v4 to reuse when retrying (default: generated)")
]


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


@_version_app.command("save")
def save_workflow_version(
    ctx: typer.Context,
    workflow_url: WorkflowArg,
    definition: Annotated[str, typer.Option("--definition", help="Path to JSON definition file")],
    name: Annotated[Optional[str], typer.Option("--name", help="Version name")] = None,
    idempotency_key: IdempotencyOpt = None,
) -> None:
    """Save a definition as a new version without publishing it."""
    args = ctx_to_args(
        ctx, workflow_url=workflow_url, definition=definition, name=name, idempotency_key=idempotency_key
    )
    _save_workflow_version(args)


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


@workflow_app.command("deploy")
def deploy_workflow(ctx: typer.Context, workflow_url: WorkflowArg, workflow_version: VersionArg) -> None:
    """Publish a saved version. A gated workflow publishes only a version that passed every gate."""
    args = ctx_to_args(ctx, workflow_url=workflow_url, version_id=workflow_version)
    _deploy_workflow_version(args)


@_deploy_gates_app.command("list")
def list_deploy_gates(
    ctx: typer.Context,
    workflow_url: Annotated[
        Optional[str], typer.Argument(help="Workflow URL or ID; omit to list the workspace's Eval gates")
    ] = None,
    gate_type: Annotated[
        Optional[str],
        typer.Option("--type", help="Only gates of this type (the workspace list has workflow_eval only)"),
    ] = None,
    eval_id: Annotated[Optional[str], typer.Option("--eval", help="Only gates backed by this Eval")] = None,
) -> None:
    """List deploy gates, of one workflow or of the whole workspace."""
    args = ctx_to_args(ctx, workflow_url=workflow_url, gate_type=gate_type, eval_id=eval_id)
    _list_deploy_gates(args)


@_deploy_gates_app.command("run")
def run_deploy_gates(
    ctx: typer.Context,
    workflow_url: WorkflowArg,
    workflow_version: VersionArg,
    wait: WaitOpt = False,
    timeout: TimeoutOpt = 3600,
    interval: IntervalOpt = 5,
    idempotency_key: IdempotencyOpt = None,
) -> None:
    """Start every deploy gate on a version. An Eval gate runs the Eval (may consume credits); approval waits."""
    args = ctx_to_args(
        ctx,
        workflow_url=workflow_url,
        version_id=workflow_version,
        wait=wait,
        timeout=timeout,
        interval=interval,
        idempotency_key=idempotency_key,
    )
    _run_deploy_gates(args)


@_deploy_gates_app.command("status")
def deploy_gates_status(
    ctx: typer.Context,
    workflow_url: WorkflowArg,
    workflow_version: VersionArg,
    wait: WaitOpt = False,
    timeout: TimeoutOpt = 3600,
    interval: IntervalOpt = 5,
) -> None:
    """Show every gate's verdict on a version and whether it is publishable."""
    args = ctx_to_args(
        ctx, workflow_url=workflow_url, version_id=workflow_version, wait=wait, timeout=timeout, interval=interval
    )
    _deploy_gates_status(args)


@_eval_gate_app.command("get")
def get_eval_gate(ctx: typer.Context, workflow_url: WorkflowArg) -> None:
    """Show the Eval gate with the Eval's checks and their thresholds."""
    args = ctx_to_args(ctx, workflow_url=workflow_url)
    _gate_call(args, lambda gates, wf: gates.get_eval_gate(wf), _eval_gate_text)


@_eval_gate_app.command("set")
def set_eval_gate(
    ctx: typer.Context,
    workflow_url: WorkflowArg,
    eval_id: Annotated[str, typer.Option("--eval", help="Eval ID")],
    threshold: ThresholdOpt = None,
    expected_updated_at: ExpectedUpdatedAtOpt = None,
) -> None:
    """Gate a workflow on an Eval; checks without --threshold need every applicable Case to pass."""
    args = ctx_to_args(ctx, workflow_url=workflow_url)
    thresholds = _parse_thresholds(args, threshold)
    expected = _expected_updated_at(expected_updated_at)
    _gate_call(
        args,
        lambda gates, wf: gates.set_eval_gate(wf, eval_id, check_thresholds=thresholds, expected_updated_at=expected),
        _eval_gate_text,
    )


@_eval_gate_app.command("update")
def update_eval_gate(
    ctx: typer.Context,
    workflow_url: WorkflowArg,
    threshold: Annotated[
        List[str], typer.Option("--threshold", help="CHECK_ID=RATE minimum pass rate, 0 < RATE <= 1 (repeatable)")
    ],
    expected_updated_at: ExpectedUpdatedAtOpt = None,
) -> None:
    """Change some checks' minimum pass rates; the others keep theirs."""
    args = ctx_to_args(ctx, workflow_url=workflow_url)
    thresholds = _parse_thresholds(args, threshold)
    expected = _expected_updated_at(expected_updated_at)
    _gate_call(
        args,
        lambda gates, wf: gates.update_eval_gate(wf, thresholds, expected_updated_at=expected),
        _eval_gate_text,
    )


@_eval_gate_app.command("new-check-thresholds")
def set_eval_gate_new_check_thresholds(
    ctx: typer.Context,
    workflow_url: WorkflowArg,
    threshold: Annotated[
        List[str], typer.Option("--threshold", help="CHECK_ID=RATE minimum pass rate, 0 < RATE <= 1 (repeatable)")
    ],
) -> None:
    """Set thresholds for checks added to the Eval after the gate was set."""
    args = ctx_to_args(ctx, workflow_url=workflow_url)
    thresholds = _parse_thresholds(args, threshold)
    _gate_call(args, lambda gates, wf: gates.set_eval_gate_new_check_thresholds(wf, thresholds), _eval_gate_text)


@_eval_gate_app.command("remove")
def remove_eval_gate(
    ctx: typer.Context,
    workflow_url: WorkflowArg,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation prompt.")] = False,
) -> None:
    """Remove the Eval gate, so the workflow publishes without passing it."""
    from roboflow.cli._output import confirm_destructive

    args = ctx_to_args(ctx, workflow_url=workflow_url, yes=yes)
    if not confirm_destructive(args, f"Remove the Eval deploy gate from '{workflow_url}'?"):
        return
    _gate_call(
        args,
        lambda gates, wf: gates.remove_eval_gate(wf),
        lambda _status: f"Removed the Eval deploy gate from {workflow_url}.",
    )


@_eval_gate_app.command("checks")
def list_eval_gate_checks(
    ctx: typer.Context,
    workflow_url: WorkflowArg,
    version_id: Annotated[Optional[str], typer.Option("--version", help="Only this version's check")] = None,
) -> None:
    """Show the Eval deploy checks of every version, or of one."""
    args = ctx_to_args(ctx, workflow_url=workflow_url)
    _gate_call(args, lambda gates, wf: gates.get_eval_checks(wf, version_id=version_id), _eval_checks_text)


@_approval_gate_app.command("get")
def get_approval_gate(ctx: typer.Context, workflow_url: WorkflowArg) -> None:
    """Show who may approve the workflow's versions."""
    args = ctx_to_args(ctx, workflow_url=workflow_url)
    _gate_call(args, lambda gates, wf: gates.get_approval_gate(wf), _approval_gate_text)


@_approval_gate_app.command("set")
def set_approval_gate(
    ctx: typer.Context,
    workflow_url: WorkflowArg,
    user: Annotated[
        Optional[List[str]], typer.Option("--user", help="Member user ID who may approve (repeatable)")
    ] = None,
    role: Annotated[
        Optional[List[str]], typer.Option("--role", help="Role ID whose members may approve (repeatable)")
    ] = None,
    allow_self_approval: Annotated[
        Optional[bool],
        typer.Option(
            "--self-approval/--no-self-approval",
            help="Whether a version's author may approve it (server default: no)",
        ),
    ] = None,
    expected_revision: Annotated[
        Optional[int],
        typer.Option("--expected-revision", help="The gate's revision as you read it; refuses the write if it changed"),
    ] = None,
) -> None:
    """Set who may approve the workflow's versions; one approval is enough. A plain API key is refused."""
    args = ctx_to_args(ctx, workflow_url=workflow_url)
    if not user and not role:
        from roboflow.cli._output import output_error

        output_error(args, "Name at least one approver.", hint="Pass --user USER_ID or --role ROLE_ID (repeatable).")
        return
    _gate_call(
        args,
        lambda gates, wf: gates.set_approval_gate(
            wf,
            user_ids=user or (),
            roles=role or (),
            allow_self_approval=allow_self_approval,
            expected_revision=expected_revision,
        ),
        _approval_gate_text,
    )


@_approval_gate_app.command("remove")
def remove_approval_gate(
    ctx: typer.Context,
    workflow_url: WorkflowArg,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation prompt.")] = False,
) -> None:
    """Remove the manual approval gate, so versions publish without approval."""
    from roboflow.cli._output import confirm_destructive

    args = ctx_to_args(ctx, workflow_url=workflow_url, yes=yes)
    if not confirm_destructive(args, f"Remove the manual approval deploy gate from '{workflow_url}'?"):
        return
    _gate_call(
        args,
        lambda gates, wf: gates.remove_approval_gate(wf),
        lambda _status: f"Removed the manual approval deploy gate from {workflow_url}.",
    )


@_approval_gate_app.command("list")
def list_approvals(ctx: typer.Context, workflow_url: WorkflowArg) -> None:
    """Show the approval status of every version someone decided on or was asked to review."""
    args = ctx_to_args(ctx, workflow_url=workflow_url)
    _gate_call(args, lambda gates, wf: gates.list_approvals(wf), _approvals_text)


@_approval_gate_app.command("status")
def version_approval_status(ctx: typer.Context, workflow_url: WorkflowArg, workflow_version: VersionArg) -> None:
    """Show who approved or rejected a version, and who may still approve it."""
    args = ctx_to_args(ctx, workflow_url=workflow_url, version_id=workflow_version)
    _gate_call(args, lambda gates, wf: gates.get_version_approval(wf, workflow_version), _version_approval_text)


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
    from roboflow.adapters.workflowdeploygatesapi import WorkflowDeployGateError
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
    except WorkflowDeployGateError as exc:
        _fail_gate(args, exc)
        return
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


def _delete_workflow(args) -> None:  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import confirm_destructive, output, output_api_error

    resolved = _resolve_workspace_and_key(args)
    if resolved is None:
        return
    workspace_url, api_key = resolved

    if not confirm_destructive(
        args,
        f"Move workflow '{workspace_url}/{args.workflow_url}' to Trash? (Retained for 30 days.)",
    ):
        return

    try:
        data = rfapi.delete_workflow(api_key, workspace_url, args.workflow_url)
    except rfapi.RoboflowError as exc:
        # Idempotent re-delete: a workflow already in Trash returns 404
        # because the public API filters trashed workflows out of the URL
        # match. Probe Trash and treat as a no-op success when found —
        # mirrors the project / version delete behavior.
        if getattr(exc, "status_code", None) == 404:
            try:
                trash = rfapi.list_trash(api_key, workspace_url)
            except rfapi.RoboflowError:
                trash = None
            if trash is not None:
                already = next(
                    (
                        w
                        for w in trash.get("sections", {}).get("workflows", [])
                        if w.get("url") == args.workflow_url or w.get("id") == args.workflow_url
                    ),
                    None,
                )
                if already is not None:
                    data = {
                        "deleted": True,
                        "type": "workflow",
                        "workspace": workspace_url,
                        "workflow": args.workflow_url,
                        "workflowId": already.get("id"),
                        "trash": True,
                        "alreadyInTrash": True,
                    }
                    output(
                        args,
                        data,
                        text=f"{workspace_url}/{args.workflow_url} is already in Trash (no-op).",
                    )
                    return
        output_api_error(
            args,
            exc,
            hint="Check your API key has 'workflow:update' scope on this workspace.",
        )
        return

    output(
        args,
        data,
        text=f"Moved {workspace_url}/{args.workflow_url} to Trash (30-day retention).",
    )


def _restore_workflow(args) -> None:  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_api_error, output_error

    resolved = _resolve_workspace_and_key(args)
    if resolved is None:
        return
    workspace_url, api_key = resolved

    try:
        trash = rfapi.list_trash(api_key, workspace_url)
    except rfapi.RoboflowError as exc:
        output_api_error(
            args,
            exc,
            hint="Check your API key has 'project:read' scope on this workspace.",
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
        output_api_error(
            args,
            exc,
            hint="Check your API key has 'workflow:update' scope on this workspace.",
        )
        return

    output(args, data, text=f"Restored {workspace_url}/{args.workflow_url} from Trash.")


# ---------------------------------------------------------------------------
# Deploy gates
# ---------------------------------------------------------------------------


def _resolve_workflow_id(args, api_key: str, workspace_url: str) -> Optional[str]:  # noqa: ANN001
    """Return the workflow document id for ``args.workflow_url`` (a URL slug or an id)."""
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output_api_error, output_error

    try:
        data = rfapi.get_workflow(api_key, workspace_url, args.workflow_url)
    except rfapi.RoboflowError as exc:
        output_api_error(
            args,
            exc,
            hint="Reading the workflow needs the workflow:read scope.",
            not_found_hint="Run 'roboflow workflow list' to see the workspace's workflows.",
        )
        return None
    workflow = data.get("workflow", data) if isinstance(data, dict) else None
    workflow_id = workflow.get("id") if isinstance(workflow, dict) else None
    if not workflow_id:
        output_error(args, "Unexpected response from API when fetching workflow.")
        return None
    return str(workflow_id)


def _gate_steps(workflow: str, version: str, gate_types: List[str]) -> List[str]:
    """The CLI steps that pass the refusing gates; no gate types means the Eval gate's."""
    run = f"roboflow workflow deploy-gates run {workflow} {version} --wait"
    approved = (
        "an approver approves it in the Roboflow app "
        f"(roboflow workflow deploy-gates approval status {workflow} {version} shows who may)"
    )
    if "manual_approval" not in gate_types:
        return [run]
    if "workflow_eval" not in gate_types:
        return [f"wait until {approved}"]
    return [f"{run}, which also waits until {approved}"]


def _gate_hint(args, exc) -> Optional[str]:  # noqa: ANN001
    workflow = getattr(args, "workflow_url", None) or "<workflow>"
    version = getattr(args, "version_id", None) or "<version-id>"
    definition = getattr(args, "definition", None) or "<file.json>"
    gate_types = getattr(exc, "gate_types", [])
    publish_steps = [
        f"roboflow workflow version save {workflow} --definition {definition}",
        *_gate_steps(workflow, "<version-id>", gate_types),
        f"roboflow workflow deploy {workflow} <version-id>",
    ]
    pass_steps = _gate_steps(workflow, version, gate_types)
    hints = {
        "deploy_check_required": (
            "This workflow has deploy gates, so it publishes only a saved version that passed them. Run: "
            + ", then ".join(publish_steps)
            + "."
        ),
        "deploy_check_not_passed": "Pass the version's deploy gates first: " + ", then ".join(pass_steps) + ".",
        "deploy_gate_feature_unavailable": "Deploy gates need a plan with workflow version history.",
        "deploy_gate_permission_required": "Changing deploy gates needs the Manage Workflow Deploy Gates permission.",
        "deploy_gate_invalid_eval": "The Eval cannot gate this workflow; see error.details.reason.",
        "deploy_check_local_inference_not_allowed": "Deploy checks run on the server. Set the Eval to run there.",
        "deploy_check_unavailable": "Deploy checks cannot start right now. Try again later.",
        "deploy_gate_invalid_rules": (
            "The approval rules cannot be set; see error.details.reason. Approvers must be workspace members "
            "whose role grants Manage Workflows."
        ),
        "deploy_approval_rules_user_required": (
            "Only a person can change who approves, and an API key is not one. Set the approval rules in the "
            "Roboflow app, or with a credential that acts for a user."
        ),
    }
    if exc.code in hints:
        return hints[exc.code]
    if exc.status_code == 401:
        return "Check your API key, or run 'roboflow auth login'."
    if exc.status_code == 403:
        return (
            "The API key needs workflow:read or workflow:update (run also needs workflow-evals:run and model:infer) "
            "and cannot be folder-scoped."
        )
    return None


def _fail_gate(args, exc) -> None:  # noqa: ANN001
    """Render a deploy gates API error as a structured CLI error and exit."""
    import json

    from roboflow.cli._output import output_error

    payload = exc.to_dict()
    payload.pop("hint", None)
    if exc.status_code in (401, 403):
        exit_code = 2
    elif exc.status_code == 404:
        exit_code = 3
    else:
        exit_code = 1
    output_error(args, json.dumps({"error": payload}), hint=_gate_hint(args, exc), exit_code=exit_code)


def _gate_call(
    args,  # noqa: ANN001
    call: Callable[[Any, Optional[str]], Any],
    text: Optional[Callable[[Any], str]] = None,
    *,
    needs_workflow: bool = True,
) -> Any:
    """Resolve the client (and workflow id), run *call*, and print its result. Returns the result."""
    from roboflow.adapters.workflowevalsapi import WorkflowEvalError
    from roboflow.cli._output import output, output_error, suppress_sdk_output
    from roboflow.core.workflow_deploy_gates import WorkflowDeployGates

    resolved = _resolve_workspace_and_key(args)
    if resolved is None:
        return None
    workspace_url, api_key = resolved
    workflow_id = _resolve_workflow_id(args, api_key, workspace_url) if needs_workflow else None
    if needs_workflow and workflow_id is None:
        return None
    try:
        with suppress_sdk_output(args):
            result = call(WorkflowDeployGates(api_key, workspace_url), workflow_id)
    except WorkflowEvalError as exc:
        _fail_gate(args, exc)
        return None
    except TimeoutError as exc:
        output_error(
            args,
            str(exc),
            hint="The checks keep running. Run 'roboflow workflow deploy-gates status ... --wait' to keep waiting.",
        )
        return None
    output(args, result, text=text(result) if text and not getattr(args, "json", False) else None)
    return result


def _parse_thresholds(args, values: Optional[List[str]]) -> Optional[Dict[str, float]]:  # noqa: ANN001
    from roboflow.cli._output import output_error

    if not values:
        return None
    thresholds: Dict[str, float] = {}
    for value in values:
        check_id, _, rate = value.rpartition("=")
        try:
            parsed: Optional[float] = float(rate)
        except ValueError:
            parsed = None
        if not check_id or parsed is None:
            output_error(
                args,
                f"Invalid --threshold '{value}'.",
                hint="Use CHECK_ID=RATE with 0 < RATE <= 1, e.g. --threshold code-reading=0.98.",
            )
            return None
        thresholds[check_id] = parsed
    return thresholds


def _expected_updated_at(value: Optional[str]) -> Any:
    from roboflow.core.workflow_deploy_gates import UNSET

    if value is None:
        return UNSET
    return None if value.lower() == "null" else value


def _wait_for(args, gates, workflow_id: Optional[str]) -> Dict[str, Any]:  # noqa: ANN001
    import sys

    if not args.json and not getattr(args, "quiet", False):
        print(f"Waiting for the deploy gates on version {args.version_id}...", file=sys.stderr)
    return gates.wait_for_version(workflow_id, args.version_id, timeout=args.timeout, interval=args.interval)


def _exit_unless_publishable(args, verdicts: Any) -> None:  # noqa: ANN001
    import sys

    if args.wait and isinstance(verdicts, dict) and not verdicts.get("publishable"):
        sys.exit(1)


def _verdicts_text(verdicts: Dict[str, Any]) -> str:
    from roboflow.cli._table import format_table

    standing = "publishable" if verdicts.get("publishable") else "not publishable"
    lines = [f"Version {verdicts.get('versionId', '')}: {standing}"]
    gates = verdicts.get("gates") or []
    if gates:
        rows = [
            {"type": g.get("type", ""), "status": g.get("status", ""), "reason": g.get("reason") or ""} for g in gates
        ]
        lines += ["", format_table(rows, columns=["type", "status", "reason"], headers=["GATE", "STATUS", "REASON"])]
    else:
        lines.append("The workflow has no deploy gates.")
    return "\n".join(lines)


def _rate(rate: Any) -> str:
    return f"{rate:.0%}" if isinstance(rate, (int, float)) else "not set (100%)"


def _eval_gate_text(status: Dict[str, Any]) -> str:
    from roboflow.cli._table import format_table

    if not status.get("gate"):
        return "No Eval deploy gate."
    evaluation = status.get("eval") or {}
    gate = status.get("gate") or {}
    lines = [
        f"Eval: {evaluation.get('name') or gate.get('evalId', '')} ({gate.get('evalId', '')})",
        f"Updated: {gate.get('updatedAt', '')}",
    ]
    checks = status.get("checks") or []
    if checks:
        rows = [
            {"id": c.get("id", ""), "name": c.get("name", ""), "rate": _rate(c.get("minimumPassRate"))} for c in checks
        ]
        lines += ["", format_table(rows, columns=["id", "name", "rate"], headers=["CHECK", "NAME", "MIN PASS RATE"])]
    return "\n".join(lines)


def _eval_checks_text(data: Dict[str, Any]) -> str:
    from roboflow.cli._table import format_table

    items = data.get("items") if "items" in data else [data]
    rows = [
        {
            "version": c.get("versionId", ""),
            "status": c.get("status", ""),
            "current": "yes" if c.get("current") else f"no ({c.get('staleReason') or 'stale'})",
            "started": c.get("startedAt", ""),
        }
        for c in items or []
    ]
    return format_table(
        rows, columns=["version", "status", "current", "started"], headers=["VERSION", "STATUS", "CURRENT", "STARTED"]
    )


def _workspace_gates_text(data: Dict[str, Any]) -> str:
    from roboflow.cli._table import format_table

    rows = [
        {
            "workflow": g.get("workflowUrl") or g.get("workflowId", ""),
            "type": g.get("type", ""),
            "eval": g.get("evalId", ""),
        }
        for g in data.get("items") or []
    ]
    return format_table(rows, columns=["workflow", "type", "eval"], headers=["WORKFLOW", "GATE", "EVAL"])


def _workflow_gates_text(data: Dict[str, Any]) -> str:
    from roboflow.cli._table import format_table

    rows = []
    for item in data.get("items") or []:
        status = item.get("status") or {}
        if item.get("type") == "manual_approval":
            details = "approvers: " + ", ".join(row["approver"] for row in _approver_rows(status))
        else:
            details = "eval: " + (
                (status.get("eval") or {}).get("name") or (status.get("gate") or {}).get("evalId", "")
            )
        rows.append({"type": item.get("type", ""), "details": details})
    if not rows:
        return "The workflow has no deploy gates."
    return format_table(rows, columns=["type", "details"], headers=["GATE", "DETAILS"])


def _approver_rows(status: Dict[str, Any]) -> List[Dict[str, str]]:
    """One row per approval rule, naming the member or role as the app shows it."""
    users = {u.get("uid"): u for u in status.get("users") or []}
    roles = {r.get("id"): r for r in status.get("roles") or []}
    rows = []
    for rule in (status.get("gate") or {}).get("rules") or []:
        if rule.get("kind") == "user":
            uid = rule.get("userId", "")
            member = users.get(uid) or {}
            rows.append({"kind": "user", "approver": member.get("email") or member.get("name") or uid, "id": uid})
        else:
            role_id = rule.get("role", "")
            rows.append({"kind": "role", "approver": (roles.get(role_id) or {}).get("name") or role_id, "id": role_id})
    return rows


def _approval_gate_text(status: Dict[str, Any]) -> str:
    from roboflow.cli._table import format_table

    gate = status.get("gate")
    if not gate:
        return "No manual approval gate."
    self_approval = "allowed" if gate.get("allowSelfApproval") else "not allowed"
    lines = [
        f"Revision: {gate.get('revision', '')} (updated {gate.get('updatedAt', '')})",
        f"Self-approval: {self_approval}",
        "One approval from anyone below approves a version; one rejection rejects it.",
        "",
        format_table(_approver_rows(status), columns=["kind", "approver", "id"], headers=["RULE", "APPROVER", "ID"]),
    ]
    return "\n".join(lines)


def _version_approval_text(approval: Dict[str, Any]) -> str:
    from roboflow.cli._table import format_table

    lines = [f"Version {approval.get('versionId', '')}: {approval.get('status', '')}"]
    if approval.get("reason"):
        lines.append(str(approval["reason"]))
    decisions = approval.get("decisions") or []
    if decisions:
        rows = [
            {
                "approver": (d.get("approver") or {}).get("email") or (d.get("approver") or {}).get("uid", ""),
                "decision": d.get("decision", ""),
                "counted": "yes" if d.get("counted") else f"no ({d.get('staleReason') or 'not eligible'})",
                "decided": d.get("decidedAt", ""),
                "comment": d.get("comment") or "",
            }
            for d in decisions
        ]
        lines += [
            "",
            format_table(
                rows,
                columns=["approver", "decision", "counted", "decided", "comment"],
                headers=["APPROVER", "DECISION", "COUNTED", "DECIDED", "COMMENT"],
            ),
        ]
    return "\n".join(lines)


def _approvals_text(data: Dict[str, Any]) -> str:
    from roboflow.cli._table import format_table

    rows = [
        {"version": a.get("versionId", ""), "status": a.get("status", ""), "reason": a.get("reason") or ""}
        for a in data.get("items") or []
    ]
    if not rows:
        return "No version has an approval decision or review request."
    return format_table(rows, columns=["version", "status", "reason"], headers=["VERSION", "STATUS", "REASON"])


def _save_workflow_version(args) -> None:  # noqa: ANN001
    definition = _read_definition_file(args)
    if definition is False:
        return

    def text(saved: Dict[str, Any]) -> str:
        version = saved.get("version") or {}
        version_id = version.get("id", "")
        if saved.get("replayed"):
            head = f"Version {version_id} was already saved with this idempotency key. It is not published."
        else:
            head = f"Saved version {version_id} of {args.workflow_url}. It is not published."
        return (
            f"{head}\nRun its deploy checks: roboflow workflow deploy-gates run {args.workflow_url} {version_id} --wait"
        )

    _gate_call(
        args,
        lambda gates, wf: gates.save_version(
            wf, definition, version_name=args.name, idempotency_key=args.idempotency_key
        ),
        text,
    )


def _deploy_workflow_version(args) -> None:  # noqa: ANN001
    def text(deployed: Dict[str, Any]) -> str:
        version_id = deployed.get("versionId", args.version_id)
        return f"Published version {version_id} of {deployed.get('workflowUrl', args.workflow_url)}."

    _gate_call(args, lambda gates, wf: gates.deploy(wf, args.version_id), text)


def _list_deploy_gates(args) -> None:  # noqa: ANN001
    if args.workflow_url:
        _gate_call(args, lambda gates, wf: gates.list_for_workflow(wf), _workflow_gates_text)
    else:
        _gate_call(
            args,
            lambda gates, _wf: gates.list(gate_type=args.gate_type, eval_id=args.eval_id),
            _workspace_gates_text,
            needs_workflow=False,
        )


def _run_deploy_gates(args) -> None:  # noqa: ANN001
    def call(gates, workflow_id: Optional[str]) -> Any:  # noqa: ANN001
        started = gates.run(workflow_id, args.version_id, idempotency_key=args.idempotency_key)
        return _wait_for(args, gates, workflow_id) if args.wait else started

    _exit_unless_publishable(args, _gate_call(args, call, _verdicts_text))


def _deploy_gates_status(args) -> None:  # noqa: ANN001
    def call(gates, workflow_id: Optional[str]) -> Any:  # noqa: ANN001
        return _wait_for(args, gates, workflow_id) if args.wait else gates.get_version(workflow_id, args.version_id)

    _exit_unless_publishable(args, _gate_call(args, call, _verdicts_text))
