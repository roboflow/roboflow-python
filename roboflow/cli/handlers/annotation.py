"""Annotation management commands: batch and job operations."""

from __future__ import annotations

import json
from typing import Annotated, Any, Callable, Optional

import typer

from roboflow.cli._compat import SortedGroup, ctx_to_args

annotation_app = typer.Typer(cls=SortedGroup, help="Annotation management commands", no_args_is_help=True)
batch_app = typer.Typer(cls=SortedGroup, help="Annotation batch commands", no_args_is_help=True)
job_app = typer.Typer(cls=SortedGroup, help="Annotation job commands", no_args_is_help=True)
annotation_app.add_typer(batch_app, name="batch")
annotation_app.add_typer(job_app, name="job")


@batch_app.command("list")
def batch_list(
    ctx: typer.Context,
    project: Annotated[str, typer.Option("-p", "--project", help="Project ID")],
) -> None:
    """List established upload batches."""
    _batch_list(ctx_to_args(ctx, project=project))


@batch_app.command("get")
def batch_get(
    ctx: typer.Context,
    batch_id: Annotated[str, typer.Argument(help="Batch ID")],
    project: Annotated[str, typer.Option("-p", "--project", help="Project ID")],
) -> None:
    """Get an established upload batch."""
    _simple_command(ctx_to_args(ctx, batch_id=batch_id, project=project), "get_batch", batch_id)


@batch_app.command("admin-list")
def batch_admin_list(
    ctx: typer.Context,
    project: Annotated[str, typer.Option("-p", "--project", help="Project ID")],
    limit: Annotated[int, typer.Option(help="Maximum batches to return, from 1 to 200")] = 50,
    after: Annotated[Optional[str], typer.Option(help="Continuation token from the previous page")] = None,
    show_empty: Annotated[bool, typer.Option("--show-empty", help="Include batches with no images")] = False,
) -> None:
    """List annotation-board batches with cursor pagination."""
    _simple_command(
        ctx_to_args(ctx, project=project),
        "list_annotation_batches",
        limit=limit,
        after=after,
        show_empty=show_empty,
    )


@batch_app.command("admin-get")
def batch_admin_get(
    ctx: typer.Context,
    batch_id: Annotated[str, typer.Argument(help="Batch ID")],
    project: Annotated[str, typer.Option("-p", "--project", help="Project ID")],
) -> None:
    """Get an annotation-board batch."""
    _simple_command(ctx_to_args(ctx, project=project), "get_annotation_batch", batch_id)


@batch_app.command("images")
def batch_images(
    ctx: typer.Context,
    batch_id: Annotated[str, typer.Argument(help="Batch ID")],
    project: Annotated[str, typer.Option("-p", "--project", help="Project ID")],
    limit: Annotated[int, typer.Option(help="Maximum image IDs to return, from 1 to 200")] = 50,
    after: Annotated[Optional[str], typer.Option(help="Continuation token from the previous page")] = None,
) -> None:
    """List image IDs in an annotation batch."""
    _simple_command(
        ctx_to_args(ctx, project=project),
        "list_annotation_batch_images",
        batch_id,
        limit=limit,
        after=after,
    )


@batch_app.command("create")
def batch_create(
    ctx: typer.Context,
    project: Annotated[str, typer.Option("-p", "--project", help="Project ID")],
    source_batch_id: Annotated[str, typer.Option("--source-batch-id", help="Source batch ID")],
    image_ids: Annotated[list[str], typer.Option("--image-id", help="Image ID to move; repeat for multiple images")],
    name: Annotated[Optional[str], typer.Option(help="Optional new batch name")] = None,
) -> None:
    """Move selected images into a new annotation batch."""
    _simple_command(
        ctx_to_args(ctx, project=project),
        "create_annotation_batch",
        source_batch_id=source_batch_id,
        image_ids=image_ids,
        name=name,
        success="Created annotation batch.",
    )


@batch_app.command("merge")
def batch_merge(
    ctx: typer.Context,
    project: Annotated[str, typer.Option("-p", "--project", help="Project ID")],
    source_batch_ids: Annotated[
        list[str], typer.Option("--source-batch-id", help="Source batch ID; repeat for multiple batches")
    ],
    target_batch_id: Annotated[str, typer.Option("--target-batch-id", help="Target batch ID")],
    yes: Annotated[bool, typer.Option("-y", "--yes", help="Confirm the merge")] = False,
) -> None:
    """Merge source batches into a target batch."""
    args = ctx_to_args(ctx, project=project, yes=yes)
    if _confirm(args, "Merge the source batches and remove the emptied batches?"):
        _simple_command(
            args,
            "merge_annotation_batches",
            source_batch_ids=source_batch_ids,
            target_batch_id=target_batch_id,
            success="Merged annotation batches.",
        )


@batch_app.command("delete")
def batch_delete(
    ctx: typer.Context,
    batch_id: Annotated[str, typer.Argument(help="Batch ID")],
    project: Annotated[str, typer.Option("-p", "--project", help="Project ID")],
    permanent: Annotated[bool, typer.Option(help="Also delete the batch's image sources")] = False,
    yes: Annotated[bool, typer.Option("-y", "--yes", help="Confirm deletion")] = False,
) -> None:
    """Delete an annotation batch."""
    args = ctx_to_args(ctx, project=project, yes=yes)
    if _confirm(args, "Delete this annotation batch?"):
        _simple_command(
            args,
            "delete_annotation_batch",
            batch_id,
            permanent=permanent,
            success="Deleted annotation batch.",
        )


@job_app.command("list")
def job_list(
    ctx: typer.Context,
    project: Annotated[str, typer.Option("-p", "--project", help="Project ID")],
) -> None:
    """List annotation jobs with legacy list-only output."""
    _job_list(ctx_to_args(ctx, project=project))


@job_app.command("admin-list")
def job_admin_list(
    ctx: typer.Context,
    project: Annotated[str, typer.Option("-p", "--project", help="Project ID")],
    limit: Annotated[int, typer.Option(help="Maximum jobs to return, from 1 to 200")] = 50,
    after: Annotated[Optional[str], typer.Option(help="Continuation token from the previous page")] = None,
    show_empty: Annotated[bool, typer.Option("--show-empty", help="Include jobs with no images")] = False,
) -> None:
    """List annotation jobs with the full paginated response."""
    _simple_command(
        ctx_to_args(ctx, project=project),
        "list_annotation_jobs_admin",
        limit=limit,
        after=after,
        show_empty=show_empty,
    )


@job_app.command("get")
def job_get(
    ctx: typer.Context,
    job_id: Annotated[str, typer.Argument(help="Annotation job ID")],
    project: Annotated[str, typer.Option("-p", "--project", help="Project ID")],
) -> None:
    """Get annotation job details."""
    _simple_command(ctx_to_args(ctx, project=project), "get_annotation_job", job_id)


@job_app.command("admin-get")
def job_admin_get(
    ctx: typer.Context,
    job_id: Annotated[str, typer.Argument(help="Annotation job ID")],
    project: Annotated[str, typer.Option("-p", "--project", help="Project ID")],
) -> None:
    """Get an annotation job through the administration endpoint."""
    _simple_command(ctx_to_args(ctx, project=project), "get_annotation_job_admin", job_id)


@job_app.command("images")
def job_images(
    ctx: typer.Context,
    job_id: Annotated[str, typer.Argument(help="Annotation job ID")],
    project: Annotated[str, typer.Option("-p", "--project", help="Project ID")],
    limit: Annotated[int, typer.Option(help="Maximum image IDs to return, from 1 to 200")] = 50,
    after: Annotated[Optional[str], typer.Option(help="Continuation token from the previous page")] = None,
) -> None:
    """List image IDs assigned to a job."""
    _simple_command(
        ctx_to_args(ctx, project=project),
        "list_annotation_job_images",
        job_id,
        limit=limit,
        after=after,
    )


@job_app.command("create")
def job_create(
    ctx: typer.Context,
    project: Annotated[str, typer.Option("-p", "--project", help="Project ID")],
    batch: Annotated[str, typer.Option(help="Source batch ID")],
    labeler: Annotated[str, typer.Option(help="Labeler email")],
    reviewer: Annotated[str, typer.Option(help="Reviewer email")],
    name: Annotated[Optional[str], typer.Option(help="Optional job name")] = None,
    num_images: Annotated[Optional[int], typer.Option("--num-images", help="Number of images")] = None,
    instructions: Annotated[Optional[str], typer.Option(help="Labeling instructions")] = None,
) -> None:
    """Create an annotation job from a batch."""
    _simple_command(
        ctx_to_args(ctx, project=project),
        "create_annotation_job_from_batch",
        batch_id=batch,
        labeler_email=labeler,
        reviewer_email=reviewer,
        name=name,
        num_images=num_images,
        instructions=instructions,
        success=f"Created annotation job{f': {name}' if name else '.'}",
    )


@job_app.command("admin-create")
def job_admin_create(
    ctx: typer.Context,
    project: Annotated[str, typer.Option("-p", "--project", help="Project ID")],
    batch: Annotated[str, typer.Option(help="Source batch ID")],
    labeler: Annotated[str, typer.Option(help="Labeler email")],
    reviewer: Annotated[str, typer.Option(help="Reviewer email")],
    name: Annotated[Optional[str], typer.Option(help="Optional job name")] = None,
    num_images: Annotated[Optional[int], typer.Option("--num-images", help="Number of images")] = None,
    instructions: Annotated[Optional[str], typer.Option(help="Labeling instructions")] = None,
) -> None:
    """Create an annotation job through the administration endpoint."""
    _simple_command(
        ctx_to_args(ctx, project=project),
        "create_annotation_job_admin",
        batch_id=batch,
        labeler_email=labeler,
        reviewer_email=reviewer,
        name=name,
        num_images=num_images,
        instructions=instructions,
        success=f"Created annotation job{f': {name}' if name else '.'}",
    )


@job_app.command("reassign-images")
def job_reassign_images(
    ctx: typer.Context,
    project: Annotated[str, typer.Option("-p", "--project", help="Project ID")],
    image_ids: Annotated[
        list[str], typer.Option("--image-id", help="Image ID to reassign; repeat for multiple images")
    ],
    labeler: Annotated[str, typer.Option(help="Labeler email")],
    reviewer: Annotated[Optional[str], typer.Option(help="Reviewer email")] = None,
    instructions: Annotated[Optional[str], typer.Option(help="Labeling instructions")] = None,
    name: Annotated[Optional[str], typer.Option(help="Optional job name")] = None,
    yes: Annotated[bool, typer.Option("-y", "--yes", help="Confirm reassignment of the images")] = False,
) -> None:
    """Create a job by explicitly reassigning images."""
    args = ctx_to_args(ctx, project=project, yes=yes)
    if _confirm(args, "Remove these images from their current assignments and reassign them to a new job?"):
        _simple_command(
            args,
            "reassign_annotation_job_images",
            image_ids=image_ids,
            labeler_email=labeler,
            reviewer_email=reviewer,
            instructions=instructions,
            name=name,
            success="Reassigned images to a new annotation job.",
        )


@job_app.command("add-images")
def job_add_images(
    ctx: typer.Context,
    job_id: Annotated[str, typer.Argument(help="Target annotation job ID")],
    project: Annotated[str, typer.Option("-p", "--project", help="Project ID")],
    image_ids: Annotated[list[str], typer.Option("--image-id", help="Image ID to add; repeat for multiple images")],
    yes: Annotated[bool, typer.Option("-y", "--yes", help="Confirm reassignment of the images")] = False,
) -> None:
    """Move selected images into an existing annotation job."""
    args = ctx_to_args(ctx, project=project, yes=yes)
    if _confirm(args, "Move these images out of their current assignments and into this job?"):
        _simple_command(
            args,
            "add_images_to_annotation_job",
            job_id,
            image_ids=image_ids,
            success="Added images to the annotation job.",
        )


@job_app.command("update")
def job_update(
    ctx: typer.Context,
    job_id: Annotated[str, typer.Argument(help="Annotation job ID")],
    project: Annotated[str, typer.Option("-p", "--project", help="Project ID")],
    labeler: Annotated[Optional[str], typer.Option(help="New labeler email")] = None,
    reviewer: Annotated[Optional[str], typer.Option(help="New reviewer email")] = None,
    instructions: Annotated[Optional[str], typer.Option(help="Replacement instructions")] = None,
) -> None:
    """Update exactly one assignment field on a job."""
    _simple_command(
        ctx_to_args(ctx, project=project),
        "update_annotation_job",
        job_id,
        labeler_email=labeler,
        reviewer_email=reviewer,
        instructions=instructions,
        success="Updated annotation job.",
    )


@job_app.command("submit-review")
def job_submit_review(
    ctx: typer.Context,
    job_id: Annotated[str, typer.Argument(help="Annotation job ID")],
    project: Annotated[str, typer.Option("-p", "--project", help="Project ID")],
) -> None:
    """Advance a labeling job into review."""
    _simple_command(
        ctx_to_args(ctx, project=project),
        "submit_annotation_job_for_review",
        job_id,
        success="Submitted annotation job for review.",
    )


@job_app.command("return-edits")
def job_return_edits(
    ctx: typer.Context,
    job_id: Annotated[str, typer.Argument(help="Annotation job ID")],
    project: Annotated[str, typer.Option("-p", "--project", help="Project ID")],
    new_labeler: Annotated[Optional[str], typer.Option("--new-labeler", help="Replacement labeler")] = None,
) -> None:
    """Move a review job back to labeling."""
    _simple_command(
        ctx_to_args(ctx, project=project),
        "return_annotation_job_for_edits",
        job_id,
        new_labeler_email=new_labeler,
        success="Returned annotation job for edits.",
    )


@job_app.command("review-image")
def job_review_image(
    ctx: typer.Context,
    job_id: Annotated[str, typer.Argument(help="Annotation job ID")],
    image_id: Annotated[str, typer.Argument(help="Image ID in the job")],
    project: Annotated[str, typer.Option("-p", "--project", help="Project ID")],
    status: Annotated[str, typer.Option(help="approved, rejected, annotated, or unannotated")],
) -> None:
    """Set the review status for one image."""
    _simple_command(
        ctx_to_args(ctx, project=project),
        "review_annotation_job_image",
        job_id,
        image_id,
        status=status,
        success=f"Set image review status to {status}.",
    )


@job_app.command("review-images")
def job_review_images(
    ctx: typer.Context,
    job_id: Annotated[str, typer.Argument(help="Annotation job ID")],
    project: Annotated[str, typer.Option("-p", "--project", help="Project ID")],
    current_status: Annotated[str, typer.Option("--current-status", help="Only images in this status")],
    status: Annotated[str, typer.Option(help="New status")],
) -> None:
    """Set a status for every matching image in a job."""
    _simple_command(
        ctx_to_args(ctx, project=project),
        "review_annotation_job_images",
        job_id,
        status=status,
        current_status=current_status,
        success=f"Set matching image review statuses to {status}.",
    )


@job_app.command("accept")
def job_accept(
    ctx: typer.Context,
    job_id: Annotated[str, typer.Argument(help="Annotation job ID")],
    project: Annotated[str, typer.Option("-p", "--project", help="Project ID")],
    split_method: Annotated[str, typer.Option("--split-method", help="preset, split, train, valid, or test")],
    statuses: Annotated[list[str], typer.Option("--status", help="Status to accept; repeat as needed")],
    train_count: Annotated[int, typer.Option("--train-count", help="Images assigned to train")],
    valid_count: Annotated[int, typer.Option("--valid-count", help="Images assigned to validation")],
    test_count: Annotated[int, typer.Option("--test-count", help="Images assigned to test")],
    image_ids: Annotated[
        Optional[list[str]], typer.Option("--image-id", help="Optional image subset; repeat as needed")
    ] = None,
    yes: Annotated[bool, typer.Option("-y", "--yes", help="Confirm Dataset acceptance")] = False,
) -> None:
    """Accept job images into Dataset and assign their splits."""
    args = ctx_to_args(ctx, project=project, yes=yes)
    if _confirm(args, "Accept these annotation job images into Dataset?"):
        _simple_command(
            args,
            "accept_annotation_job_images",
            job_id,
            split_method=split_method,
            statuses_to_include=statuses,
            train_count=train_count,
            valid_count=valid_count,
            test_count=test_count,
            image_ids=image_ids,
            success="Accepted annotation job images into Dataset.",
        )


@job_app.command("move-to-unassigned")
def job_move_to_unassigned(
    ctx: typer.Context,
    job_id: Annotated[str, typer.Argument(help="Annotation job ID")],
    project: Annotated[str, typer.Option("-p", "--project", help="Project ID")],
    yes: Annotated[bool, typer.Option("-y", "--yes", help="Confirm removal of the job")] = False,
) -> None:
    """Remove a job and retain its images as unassigned."""
    args = ctx_to_args(ctx, project=project, yes=yes)
    if _confirm(args, "Remove this job and move its images to unassigned?"):
        _simple_command(
            args,
            "move_annotation_job_to_unassigned",
            job_id,
            success="Moved annotation job images to unassigned.",
        )


@job_app.command("delete-annotations")
def job_delete_annotations(
    ctx: typer.Context,
    job_id: Annotated[str, typer.Argument(help="Annotation job ID")],
    project: Annotated[str, typer.Option("-p", "--project", help="Project ID")],
    yes: Annotated[bool, typer.Option("-y", "--yes", help="Confirm annotation deletion")] = False,
) -> None:
    """Delete project annotations from every image assigned to a job."""
    args = ctx_to_args(ctx, project=project, yes=yes)
    if _confirm(args, "Delete every project annotation assigned to this job?"):
        _simple_command(
            args,
            "delete_annotation_job_annotations",
            job_id,
            success="Deleted annotation job annotations.",
        )


def _resolve_project_context(args: Any) -> Optional[tuple[str, str, str]]:
    from roboflow.cli._output import output_error
    from roboflow.cli._resolver import resolve_resource
    from roboflow.config import load_roboflow_api_key

    try:
        workspace, project, _version = resolve_resource(args.project, workspace_override=args.workspace)
    except ValueError as exc:
        output_error(args, str(exc))
        return None
    api_key = args.api_key or load_roboflow_api_key(workspace)
    if not api_key:
        output_error(args, "No API key found.", hint="Set ROBOFLOW_API_KEY or run 'roboflow auth login'.", exit_code=2)
        return None
    return api_key, workspace, project


def _call(args: Any, operation: Callable[[str, str, str], Any]) -> Any:
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output_api_error, output_error

    context = _resolve_project_context(args)
    if context is None:
        return None
    try:
        return _normalize_timestamps(operation(*context))
    except rfapi.RoboflowError as exc:
        output_api_error(args, exc)
    except ValueError as exc:
        output_error(args, str(exc))
    return None


def _simple_command(args: Any, method: str, *positional: Any, success: Optional[str] = None, **kwargs: Any) -> None:
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output

    operation = getattr(rfapi, method)
    data = _call(args, lambda key, workspace, project: operation(key, workspace, project, *positional, **kwargs))
    if data is not None:
        output(args, data, text=success or json.dumps(data, indent=2, default=str))


def _batch_list(args: Any) -> None:
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output
    from roboflow.cli._table import format_table

    data = _call(args, lambda key, workspace, project: rfapi.list_batches(key, workspace, project))
    if data is None:
        return
    batches = data if isinstance(data, list) else data.get("batches", data)
    table = format_table(
        batches if isinstance(batches, list) else [],
        columns=["name", "id", "status", "images"],
        headers=["NAME", "ID", "STATUS", "IMAGE_COUNT"],
    )
    output(args, batches, text=table)


def _job_list(args: Any) -> None:
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output
    from roboflow.cli._table import format_table

    data = _call(args, lambda key, workspace, project: rfapi.list_annotation_jobs(key, workspace, project))
    if data is None:
        return
    jobs = data if isinstance(data, list) else data.get("jobs", data)
    table = format_table(
        jobs if isinstance(jobs, list) else [],
        columns=["name", "id", "status", "assigned_to"],
        headers=["NAME", "ID", "STATUS", "ASSIGNED_TO"],
    )
    output(args, jobs, text=table)


def _confirm(args: Any, prompt: str) -> bool:
    from roboflow.cli._output import confirm_destructive

    return confirm_destructive(args, prompt=prompt)


def _normalize_timestamps(obj: Any) -> Any:
    from datetime import datetime, timezone

    if isinstance(obj, dict):
        if "_seconds" in obj and "_nanoseconds" in obj and len(obj) == 2:
            return datetime.fromtimestamp(obj["_seconds"], tz=timezone.utc).isoformat()
        return {key: _normalize_timestamps(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_normalize_timestamps(item) for item in obj]
    return obj
