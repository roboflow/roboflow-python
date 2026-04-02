"""Annotation management commands: batch and job operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``annotation`` command group."""
    ann_parser = subparsers.add_parser("annotation", help="Annotation management commands")
    ann_sub = ann_parser.add_subparsers(title="annotation commands", dest="annotation_command")

    _add_batch(ann_sub)
    _add_job(ann_sub)

    ann_parser.set_defaults(func=lambda args: ann_parser.print_help())


# ---------------------------------------------------------------------------
# batch
# ---------------------------------------------------------------------------


def _add_batch(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    batch_parser = sub.add_parser("batch", help="Annotation batch commands")
    batch_sub = batch_parser.add_subparsers(title="batch commands", dest="batch_command")

    # batch list
    p = batch_sub.add_parser("list", help="List annotation batches")
    p.add_argument("-p", "--project", required=True, help="Project ID")
    p.set_defaults(func=_batch_list)

    # batch get
    p = batch_sub.add_parser("get", help="Get annotation batch details")
    p.add_argument("batch_id", help="Batch ID")
    p.add_argument("-p", "--project", required=True, help="Project ID")
    p.set_defaults(func=_batch_get)

    batch_parser.set_defaults(func=lambda args: batch_parser.print_help())


# ---------------------------------------------------------------------------
# job
# ---------------------------------------------------------------------------


def _add_job(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    job_parser = sub.add_parser("job", help="Annotation job commands")
    job_sub = job_parser.add_subparsers(title="job commands", dest="job_command")

    # job list
    p = job_sub.add_parser("list", help="List annotation jobs")
    p.add_argument("-p", "--project", required=True, help="Project ID")
    p.set_defaults(func=_job_list)

    # job get
    p = job_sub.add_parser("get", help="Get annotation job details")
    p.add_argument("job_id", help="Job ID")
    p.add_argument("-p", "--project", required=True, help="Project ID")
    p.set_defaults(func=_job_get)

    # job create
    p = job_sub.add_parser("create", help="Create an annotation job")
    p.add_argument("-p", "--project", required=True, help="Project ID")
    p.add_argument("--name", required=True, help="Job name")
    p.add_argument("--batch", required=True, help="Batch ID")
    p.add_argument("--num-images", required=True, type=int, help="Number of images")
    p.add_argument("--labeler", required=True, help="Labeler email")
    p.add_argument("--reviewer", required=True, help="Reviewer email")
    p.set_defaults(func=_job_create)

    job_parser.set_defaults(func=lambda args: job_parser.print_help())


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _normalize_timestamps(obj):
    """Recursively convert Firestore timestamp dicts ({"_seconds": N, "_nanoseconds": N}) to ISO 8601 strings."""
    from datetime import datetime, timezone

    if isinstance(obj, dict):
        if "_seconds" in obj and "_nanoseconds" in obj and len(obj) == 2:
            return datetime.fromtimestamp(obj["_seconds"], tz=timezone.utc).isoformat()
        return {k: _normalize_timestamps(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_timestamps(item) for item in obj]
    return obj


# ---------------------------------------------------------------------------
# handlers
# ---------------------------------------------------------------------------


def _resolve_project_context(args: argparse.Namespace):  # type: ignore[return]
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


def _batch_list(args: argparse.Namespace) -> None:
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


def _batch_get(args: argparse.Namespace) -> None:
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


def _job_list(args: argparse.Namespace) -> None:
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


def _job_get(args: argparse.Namespace) -> None:
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


def _job_create(args: argparse.Namespace) -> None:
    import roboflow
    from roboflow.cli._output import output, output_error, suppress_sdk_output

    ctx = _resolve_project_context(args)
    if ctx is None:
        return
    _api_key, workspace_url, project_slug = ctx

    with suppress_sdk_output(args):
        try:
            rf = roboflow.Roboflow()
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
