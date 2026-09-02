"""Batch Processing commands backed by Roboflow's durable workspace jobs."""

from __future__ import annotations

import re
import uuid
from enum import Enum
from typing import Annotated, Optional

import typer

from roboflow.cli._compat import SortedGroup, ctx_to_args

batch_app = typer.Typer(cls=SortedGroup, help="Run and manage Batch Processing jobs", no_args_is_help=True)


class BatchMachine(str, Enum):
    """Execution pools exposed by the Asset Library Batch Processing surface."""

    CPU = "cpu"
    GPU = "gpu"


@batch_app.command("create")
def create(
    ctx: typer.Context,
    workflow: Annotated[str, typer.Option(help="Published Workflow ID to run")],
    image_ids: Annotated[
        Optional[str],
        typer.Option("--image-ids", help="Comma-separated exact Asset Library image IDs"),
    ] = None,
    query: Annotated[
        Optional[str],
        typer.Option(help="RoboQL filter selecting all current matches"),
    ] = None,
    all_images: Annotated[
        bool,
        typer.Option("--all", help="Explicitly run on the entire Asset Library"),
    ] = False,
    machine: Annotated[
        BatchMachine,
        typer.Option(help="Execution machine; defaults to the product UI default"),
    ] = BatchMachine.CPU,
    name: Annotated[Optional[str], typer.Option(help="Optional user-facing job name")] = None,
    request_id: Annotated[
        Optional[str],
        typer.Option(help="Stable idempotency key; reuse only when retrying this same launch"),
    ] = None,
) -> None:
    """Queue a Workflow over one exact or reviewed Asset Library selection."""
    args = ctx_to_args(
        ctx,
        workflow=workflow,
        image_ids=image_ids,
        query=query,
        all_images=all_images,
        machine=machine.value,
        name=name,
        request_id=request_id,
    )
    _create(args)


@batch_app.command("status")
def status(
    ctx: typer.Context,
    job_id: Annotated[str, typer.Argument(help="Batch job ID")],
) -> None:
    """Show current durable job status and configuration."""
    _status(ctx_to_args(ctx, job_id=job_id))


@batch_app.command("list")
def list_jobs(
    ctx: typer.Context,
    page_size: Annotated[int, typer.Option(min=1, max=100, help="Jobs per page")] = 10,
    next_page_token: Annotated[Optional[str], typer.Option(help="Pagination token")] = None,
    search: Annotated[Optional[str], typer.Option(help="Search names, Workflows, and status text")] = None,
) -> None:
    """List Batch Processing jobs."""
    _list(ctx_to_args(ctx, page_size=page_size, next_page_token=next_page_token, search=search))


@batch_app.command("abort")
def abort(
    ctx: typer.Context,
    job_id: Annotated[str, typer.Argument(help="Batch job ID")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Confirm without prompting")] = False,
) -> None:
    """Abort a Batch Processing job."""
    _abort(ctx_to_args(ctx, job_id=job_id, yes=yes))


@batch_app.command("restart")
def restart(
    ctx: typer.Context,
    job_id: Annotated[str, typer.Argument(help="Batch job ID")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Confirm credit-spending restart")] = False,
) -> None:
    """Restart a Batch Processing job with its existing configuration."""
    _restart(ctx_to_args(ctx, job_id=job_id, yes=yes))


def _resolve_ws_and_key(args):  # noqa: ANN001
    from roboflow.cli._resolver import resolve_ws_and_key

    return resolve_ws_and_key(args)


def _parse_image_ids(raw: Optional[str]) -> list[str]:
    if raw is None:
        return []
    return list(dict.fromkeys(part.strip() for part in raw.split(",") if part.strip()))


def _validate_job_id(args, job_id: str, *, label: str = "job ID") -> None:  # noqa: ANN001
    from roboflow.cli._output import output_error

    if not re.fullmatch(r"[a-z0-9-]{1,20}", job_id):
        output_error(
            args,
            f"{label} must be 1-20 lowercase letters, numbers, or hyphens.",
        )


def _create(args) -> None:  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_api_error, output_error

    ids = _parse_image_ids(args.image_ids)
    selection_modes = int(bool(ids)) + int(args.query is not None) + int(args.all_images)
    if selection_modes != 1:
        output_error(
            args,
            "Choose exactly one selection: --image-ids, --query, or --all.",
            hint="The CLI never broadens an omitted or ambiguous selection to the whole Asset Library.",
        )
        return
    if args.query is not None and not args.query.strip():
        output_error(
            args,
            "--query cannot be empty.",
            hint="Use --all to explicitly select the entire Asset Library.",
        )
        return
    if len(ids) > 2048:
        output_error(args, "At most 2048 explicit image IDs can be queued in one request.")
        return

    request_id = args.request_id or str(uuid.uuid4())
    if not 8 <= len(request_id) <= 128 or not re.fullmatch(r"[A-Za-z0-9_-]+", request_id):
        output_error(
            args,
            "--request-id must be 8-128 letters, numbers, underscores, or hyphens.",
        )
        return
    resolved = _resolve_ws_and_key(args)
    if not resolved:
        return
    workspace, api_key = resolved
    query = "" if args.all_images else args.query

    try:
        result = rfapi.create_asset_library_batch_job(
            api_key,
            workspace,
            workflow_id=args.workflow,
            idempotency_key=request_id,
            image_ids=ids or None,
            query=query,
            machine_type=args.machine,
            display_name=args.name,
        )
    except rfapi.RoboflowError as exc:
        output_api_error(
            args,
            exc,
            hint=f"Retry the same launch with --request-id {request_id}; use a new ID only for a new job intent.",
            auth_hint="Check the API key has 'batch-processing:trigger' scope and access to the selected resources.",
        )
        return

    result = {**result, "requestId": request_id}
    text = (
        f"Queued {result['displayName']}\n"
        f"jobId={result['jobId']}\n"
        f"taskId={result['taskId']}\n"
        f"requestId={request_id}\n"
        f"Next: roboflow asynctasks wait {result['taskId']}"
    )
    output(args, result, text=text)


def _status(args) -> None:  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_api_error

    _validate_job_id(args, args.job_id)
    resolved = _resolve_ws_and_key(args)
    if not resolved:
        return
    workspace, api_key = resolved
    try:
        result = rfapi.get_batch_processing_job(api_key, workspace, args.job_id)
    except rfapi.RoboflowError as exc:
        output_api_error(
            args,
            exc,
            auth_hint="Check the API key has 'batch-processing:read' scope.",
            not_found_hint=(
                "If the job was just queued, wait using the task ID returned by 'batch create'; "
                "otherwise check the job ID and workspace."
            ),
        )
        return

    job = result["job"]
    state = job.get("currentStage") or ("terminal" if job["isTerminal"] else "queued")
    output(
        args,
        result,
        text=(f"jobId={job['jobId']} state={state} terminal={job['isTerminal']} error={job['error']}"),
    )


def _list(args) -> None:  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_api_error, output_error
    from roboflow.cli._table import format_table

    if args.next_page_token:
        _validate_job_id(args, args.next_page_token, label="--next-page-token")
    if args.search is not None and len(args.search) > 160:
        output_error(args, "--search must be at most 160 characters.")
        return
    resolved = _resolve_ws_and_key(args)
    if not resolved:
        return
    workspace, api_key = resolved
    try:
        result = rfapi.list_batch_processing_jobs(
            api_key,
            workspace,
            page_size=args.page_size,
            next_page_token=args.next_page_token,
            search=args.search,
        )
    except rfapi.RoboflowError as exc:
        output_api_error(args, exc, auth_hint="Check the API key has 'batch-processing:read' scope.")
        return

    rows = [
        {
            "jobId": job["jobId"],
            "name": job["name"],
            "stage": job.get("currentStage") or ("terminal" if job["isTerminal"] else "queued"),
            "error": job["error"],
            "updated": job["lastUpdate"],
        }
        for job in result["jobs"]
    ]
    table = format_table(rows, columns=["jobId", "name", "stage", "error", "updated"])
    if result.get("nextPageToken"):
        table += f"\nNext page: --next-page-token {result['nextPageToken']}"
    output(args, result, text=table)


def _abort(args) -> None:  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import confirm_destructive

    _validate_job_id(args, args.job_id)
    if not confirm_destructive(args, f"Abort Batch Processing job '{args.job_id}'?"):
        return
    _run_control_action(args, rfapi.abort_batch_processing_job, "Aborted")


def _restart(args) -> None:  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import confirm_destructive

    _validate_job_id(args, args.job_id)
    if not confirm_destructive(
        args,
        f"Restart Batch Processing job '{args.job_id}'? This can consume credits.",
    ):
        return
    _run_control_action(args, rfapi.restart_batch_processing_job, "Restarted")


def _run_control_action(args, action, verb: str) -> None:  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_api_error

    resolved = _resolve_ws_and_key(args)
    if not resolved:
        return
    workspace, api_key = resolved
    try:
        result = action(api_key, workspace, args.job_id)
    except rfapi.RoboflowError as exc:
        output_api_error(
            args,
            exc,
            auth_hint="Check the API key has 'batch-processing:trigger' scope.",
            not_found_hint="Check the job ID and workspace.",
        )
        return
    output(args, result, text=f"{verb} Batch Processing job {args.job_id}.")
