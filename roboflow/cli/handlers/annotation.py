"""Annotation management commands: batch and job operations."""

from __future__ import annotations

from typing import Annotated

import typer

from roboflow.cli._compat import SortedGroup, ctx_to_args

annotation_app = typer.Typer(cls=SortedGroup, help="Annotation management commands", no_args_is_help=True)
batch_app = typer.Typer(cls=SortedGroup, help="Annotation batch commands", no_args_is_help=True)
job_app = typer.Typer(cls=SortedGroup, help="Annotation job commands", no_args_is_help=True)

annotation_app.add_typer(batch_app, name="batch")
annotation_app.add_typer(job_app, name="job")


# ---------------------------------------------------------------------------
# batch commands
# ---------------------------------------------------------------------------


@batch_app.command("list")
def batch_list(
    ctx: typer.Context,
    project: Annotated[str, typer.Option("-p", "--project", help="Project ID")],
) -> None:
    """List annotation batches."""
    args = ctx_to_args(ctx, project=project)
    _batch_list(args)


@batch_app.command("get")
def batch_get(
    ctx: typer.Context,
    batch_id: Annotated[str, typer.Argument(help="Batch ID")],
    project: Annotated[str, typer.Option("-p", "--project", help="Project ID")],
) -> None:
    """Get annotation batch details."""
    args = ctx_to_args(ctx, batch_id=batch_id, project=project)
    _batch_get(args)


# ---------------------------------------------------------------------------
# job commands
# ---------------------------------------------------------------------------


@job_app.command("list")
def job_list(
    ctx: typer.Context,
    project: Annotated[str, typer.Option("-p", "--project", help="Project ID")],
) -> None:
    """List annotation jobs."""
    args = ctx_to_args(ctx, project=project)
    _job_list(args)


@job_app.command("get")
def job_get(
    ctx: typer.Context,
    job_id: Annotated[str, typer.Argument(help="Job ID")],
    project: Annotated[str, typer.Option("-p", "--project", help="Project ID")],
) -> None:
    """Get annotation job details."""
    args = ctx_to_args(ctx, job_id=job_id, project=project)
    _job_get(args)


@job_app.command("create")
def job_create(
    ctx: typer.Context,
    project: Annotated[str, typer.Option("-p", "--project", help="Project ID")],
    name: Annotated[str, typer.Option(help="Job name")],
    batch: Annotated[str, typer.Option(help="Batch ID")],
    num_images: Annotated[int, typer.Option("--num-images", help="Number of images")],
    labeler: Annotated[str, typer.Option(help="Labeler email")],
    reviewer: Annotated[str, typer.Option(help="Reviewer email")],
) -> None:
    """Create an annotation job."""
    args = ctx_to_args(
        ctx,
        project=project,
        name=name,
        batch=batch,
        num_images=num_images,
        labeler=labeler,
        reviewer=reviewer,
    )
    _job_create(args)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _normalize_timestamps(obj):  # noqa: ANN001
    """Recursively convert Firestore timestamp dicts ({"_seconds": N, "_nanoseconds": N}) to ISO 8601 strings."""
    from datetime import datetime, timezone

    if isinstance(obj, dict):
        if "_seconds" in obj and "_nanoseconds" in obj and len(obj) == 2:
            return datetime.fromtimestamp(obj["_seconds"], tz=timezone.utc).isoformat()
        return {k: _normalize_timestamps(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_timestamps(item) for item in obj]
    return obj


def _resolve_project_context(args):  # noqa: ANN001
    """Resolve workspace/project from -p flag and return (api_key, ws, proj) or call output_error."""
    from roboflow.cli._output import output_error
    from roboflow.cli._resolver import resolve_resource
    from roboflow.config import load_roboflow_api_key

    try:
        workspace_url, project_slug, _version = resolve_resource(args.project, workspace_override=args.workspace)
    except ValueError as exc:
        output_error(args, str(exc))
        return None

    api_key = args.api_key or load_roboflow_api_key(workspace_url)
    if not api_key:
        output_error(args, "No API key found.", hint="Set ROBOFLOW_API_KEY or run 'roboflow auth login'.", exit_code=2)
        return None

    return api_key, workspace_url, project_slug


# ---------------------------------------------------------------------------
# handler implementations
# ---------------------------------------------------------------------------


def _batch_list(args):  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error
    from roboflow.cli._table import format_table

    ctx = _resolve_project_context(args)
    if ctx is None:
        return
    api_key, workspace_url, project_slug = ctx

    try:
        data = rfapi.list_batches(api_key, workspace_url, project_slug)
    except rfapi.RoboflowError as exc:
        output_error(args, str(exc), exit_code=3)
        return

    batches = data if isinstance(data, list) else data.get("batches", data)
    batches = _normalize_timestamps(batches)

    table = format_table(
        batches if isinstance(batches, list) else [],
        columns=["name", "id", "status", "images"],
        headers=["NAME", "ID", "STATUS", "IMAGE_COUNT"],
    )
    output(args, batches, text=table)


def _batch_get(args):  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error

    ctx = _resolve_project_context(args)
    if ctx is None:
        return
    api_key, workspace_url, project_slug = ctx

    try:
        data = rfapi.get_batch(api_key, workspace_url, project_slug, args.batch_id)
    except rfapi.RoboflowError as exc:
        output_error(args, str(exc), exit_code=3)
        return

    data = _normalize_timestamps(data)
    batch = data.get("batch", data) if isinstance(data, dict) else data

    lines = []
    if isinstance(batch, dict):
        for key, val in batch.items():
            lines.append(f"  {key:16s} {val}")
    text = "\n".join(lines) if lines else "(no batch details)"

    output(args, data, text=text)


def _job_list(args):  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error
    from roboflow.cli._table import format_table

    ctx = _resolve_project_context(args)
    if ctx is None:
        return
    api_key, workspace_url, project_slug = ctx

    try:
        data = rfapi.list_annotation_jobs(api_key, workspace_url, project_slug)
    except rfapi.RoboflowError as exc:
        output_error(args, str(exc), exit_code=3)
        return

    jobs = data if isinstance(data, list) else data.get("jobs", data)
    jobs = _normalize_timestamps(jobs)

    table = format_table(
        jobs if isinstance(jobs, list) else [],
        columns=["name", "id", "status", "assigned_to"],
        headers=["NAME", "ID", "STATUS", "ASSIGNED_TO"],
    )
    output(args, jobs, text=table)


def _job_get(args):  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error

    ctx = _resolve_project_context(args)
    if ctx is None:
        return
    api_key, workspace_url, project_slug = ctx

    try:
        data = rfapi.get_annotation_job(api_key, workspace_url, project_slug, args.job_id)
    except rfapi.RoboflowError as exc:
        output_error(args, str(exc), exit_code=3)
        return

    data = _normalize_timestamps(data)
    job = data.get("job", data) if isinstance(data, dict) else data

    lines = []
    if isinstance(job, dict):
        for key, val in job.items():
            lines.append(f"  {key:16s} {val}")
    text = "\n".join(lines) if lines else "(no job details)"

    output(args, data, text=text)


def _job_create(args):  # noqa: ANN001
    import roboflow
    from roboflow.cli._output import output, output_error, suppress_sdk_output

    ctx = _resolve_project_context(args)
    if ctx is None:
        return
    _api_key, workspace_url, project_slug = ctx

    with suppress_sdk_output(args):
        try:
            rf = roboflow.Roboflow(api_key=_api_key)
            workspace = rf.workspace(workspace_url)
            project = workspace.project(project_slug)
        except Exception as exc:
            output_error(args, str(exc))
            return

    try:
        result = project.create_annotation_job(
            name=args.name,
            batch_id=args.batch,
            num_images=args.num_images,
            labeler_email=args.labeler,
            reviewer_email=args.reviewer,
        )
    except Exception as exc:
        output_error(args, str(exc))
        return

    output(args, result, text=f"Created annotation job: {args.name}")
