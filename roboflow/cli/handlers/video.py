"""Video commands: native video Source ingestion and legacy video inference."""

from __future__ import annotations

from typing import Annotated, Optional

import typer

from roboflow.cli._compat import SortedGroup, ctx_to_args

video_app = typer.Typer(
    cls=SortedGroup,
    help="Native video upload and video inference operations",
    no_args_is_help=True,
)


@video_app.command("infer")
def infer(
    ctx: typer.Context,
    project: Annotated[str, typer.Option("-p", "--project", help="Project ID")],
    version_number: Annotated[int, typer.Option("-v", "--version", help="Model version number")],
    video_file: Annotated[str, typer.Option("-f", "--file", help="Path to video file")],
    fps: Annotated[int, typer.Option("--fps", help="Frames per second")] = 5,
) -> None:
    """Run video inference."""
    args = ctx_to_args(ctx, project=project, version_number=version_number, video_file=video_file, fps=fps)
    _video_infer(args)


@video_app.command("status")
def status(
    ctx: typer.Context,
    job_id: Annotated[str, typer.Argument(help="Job ID to check")],
) -> None:
    """Check a legacy video inference job's status (not native upload ingestion)."""
    args = ctx_to_args(ctx, job_id=job_id)
    _video_status(args)


@video_app.command("upload")
def upload(
    ctx: typer.Context,
    project: Annotated[str, typer.Option("-p", "--project", help="Project ID, or workspace/project")],
    video_file: Annotated[str, typer.Option("-f", "--file", help="Path to an original .mp4 or .mov file")],
    batch: Annotated[Optional[str], typer.Option("-b", "--batch", help="Annotation batch name")] = None,
    metadata: Annotated[
        Optional[str], typer.Option("--metadata", help='JSON object of metadata, e.g. \'{"camera": "one"}\'')
    ] = None,
    poll_interval: Annotated[
        float, typer.Option("--poll-interval", min=0.1, help="Seconds between status polls while waiting")
    ] = 2.0,
    poll_timeout: Annotated[
        float, typer.Option("--poll-timeout", min=0, help="Seconds to wait for a terminal state")
    ] = 300.0,
    split: Annotated[Optional[str], typer.Option("-s", "--split", help="Dataset split: train, valid or test")] = None,
    tag: Annotated[Optional[str], typer.Option("-t", "--tag", help="Comma-separated tag names")] = None,
    wait: Annotated[
        bool, typer.Option("--wait/--no-wait", help="Poll until the upload reaches a terminal state")
    ] = True,
) -> None:
    """Upload original video bytes as a native video Source.

    Streams the file unchanged and reports the canonical video ID.
    """
    args = ctx_to_args(
        ctx,
        project=project,
        video_file=video_file,
        batch=batch,
        metadata=metadata,
        poll_interval=poll_interval,
        poll_timeout=poll_timeout,
        split=split,
        tag=tag,
        wait=wait,
    )
    _video_upload(args)


@video_app.command("upload-status")
def upload_status(
    ctx: typer.Context,
    video_id: Annotated[str, typer.Argument(help="Video ID reported by 'roboflow video upload'")],
    project: Annotated[str, typer.Option("-p", "--project", help="Project ID, or workspace/project")],
    poll_interval: Annotated[
        float, typer.Option("--poll-interval", min=0.1, help="Seconds between status polls while waiting")
    ] = 2.0,
    poll_timeout: Annotated[
        float, typer.Option("--poll-timeout", min=0, help="Seconds to wait for a terminal state")
    ] = 300.0,
    wait: Annotated[
        bool, typer.Option("--wait/--no-wait", help="Poll until a terminal state instead of reading once")
    ] = False,
) -> None:
    """Check a native video upload's ingestion status and canonical video ID."""
    args = ctx_to_args(
        ctx,
        video_id=video_id,
        project=project,
        poll_interval=poll_interval,
        poll_timeout=poll_timeout,
        wait=wait,
    )
    _video_upload_status(args)


# ---------------------------------------------------------------------------
# Business logic (unchanged from argparse version)
# ---------------------------------------------------------------------------


def _video_infer(args) -> None:  # noqa: ANN001
    import roboflow
    from roboflow.cli._output import output, output_error
    from roboflow.config import load_roboflow_api_key

    api_key = args.api_key or load_roboflow_api_key(None)
    if not api_key:
        output_error(args, "No API key found.", hint="Set ROBOFLOW_API_KEY or run 'roboflow auth login'.", exit_code=2)
        return

    try:
        from roboflow.cli._output import suppress_sdk_output

        with suppress_sdk_output():
            rf = roboflow.Roboflow(api_key)
            project = rf.workspace().project(args.project)
            version = project.version(args.version_number)
            model = getattr(version, "_model", None)
            if model is None:
                output_error(
                    args,
                    f"No model found for project '{args.project}' version {args.version_number}.",
                    hint="Train or deploy a model for this version before running video inference.",
                    exit_code=3,
                )
                return

            job_id, _signed_url, _expire_time = model.predict_video(
                args.video_file,
                args.fps,
                prediction_type="batch-video",
            )
    except Exception as exc:
        output_error(args, str(exc))
        return

    data = {"job_id": job_id, "status": "submitted"}
    output(args, data, text=f"Video inference submitted. Job ID: {job_id}")


def _video_status(args) -> None:  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error
    from roboflow.config import load_roboflow_api_key

    api_key = args.api_key or load_roboflow_api_key(None)
    if not api_key:
        output_error(args, "No API key found.", hint="Set ROBOFLOW_API_KEY or run 'roboflow auth login'.", exit_code=2)
        return

    try:
        data = rfapi.get_video_job_status(api_key, args.job_id)
    except rfapi.RoboflowError as exc:
        msg = str(exc)
        if "NOT FOUND" in msg.upper():
            output_error(
                args,
                f"Video job '{args.job_id}' not found.",
                hint="Check the job ID. You can get job IDs from 'roboflow video infer'.",
                exit_code=3,
            )
        else:
            output_error(args, msg, exit_code=3)
        return

    status = data.get("status", "unknown")
    progress = data.get("progress", "")
    text_lines = [
        f"Job ID:   {args.job_id}",
        f"Status:   {status}",
    ]
    if progress:
        text_lines.append(f"Progress: {progress}")
    output(args, data, text="\n".join(text_lines))


# ---------------------------------------------------------------------------
# Native video Source business logic
# ---------------------------------------------------------------------------

_TERMINAL_UPLOAD_STATES = frozenset({"uploaded", "failed"})


def _load_project(args):  # noqa: ANN001
    """Load the project for a native video command, honoring CLI credential precedence."""
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output_api_error
    from roboflow.cli._resolver import resolve_project_context

    resolved = resolve_project_context(args)
    if resolved is None:
        return None
    api_key, workspace, project_slug = resolved

    try:
        data = rfapi.get_project(api_key, workspace, project_slug)
    except rfapi.RoboflowError as exc:
        output_api_error(
            args,
            exc,
            hint=f"Check that project '{workspace}/{project_slug}' exists and your API key can read it.",
        )
        return None

    from roboflow.core.project import Project

    return Project(api_key, data["project"])


def _unknown_video_hint(args) -> str:  # noqa: ANN001
    return f"Check the video ID reported by 'roboflow video upload -p {args.project}'."


def _emit_upload_status(args, status) -> None:  # noqa: ANN001
    """Render an ingestion status, exiting nonzero when the upload failed."""
    from roboflow.cli._output import output, output_error

    state = status.get("status", "unknown")
    video_id = status.get("videoId", "")

    if state == "failed":
        output_error(
            args,
            f"Native video upload {video_id} failed during processing.",
            hint=status.get("error") or "Re-upload the original file and check that it is a valid MP4/MOV.",
        )
        return

    lines = [f"Video ID: {video_id}", f"Status:   {state}"]
    if status.get("duplicate"):
        lines.append("Duplicate: yes (deduplicated onto an existing Source)")
    resolved_batch = status.get("resolvedBatch")
    if isinstance(resolved_batch, dict):
        lines.append(f"Batch:    {resolved_batch.get('name', '')} ({resolved_batch.get('id', '')})")
    elif resolved_batch:
        lines.append(f"Batch:    {resolved_batch}")
    if state not in _TERMINAL_UPLOAD_STATES:
        lines.append(f"Still processing. Re-check with 'roboflow video upload-status {video_id} -p {args.project}'.")

    # The server status document is the JSON payload, so --json stays a stable
    # passthrough of videoId/status/duplicate/resolvedBatch.
    output(args, status, text="\n".join(lines))


def _wait_for_upload(args, project, video_id):  # noqa: ANN001
    """Bounded wait, reporting the video ID so a timeout stays actionable."""
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output_api_error

    try:
        return project.wait_for_video_upload(
            video_id,
            poll_interval=args.poll_interval,
            poll_timeout=args.poll_timeout,
        )
    except rfapi.RoboflowError as exc:
        output_api_error(
            args,
            exc,
            hint=f"Re-check with 'roboflow video upload-status {video_id} -p {args.project}'.",
            not_found_hint=_unknown_video_hint(args),
        )
        return None


def _video_upload(args) -> None:  # noqa: ANN001
    import json as json_mod
    import os

    from roboflow.adapters import rfapi
    from roboflow.cli._output import output_api_error, output_error

    if not os.path.isfile(args.video_file):
        output_error(args, f"Video file not found: {args.video_file}", hint="Check the path to the video file.")
        return

    metadata = None
    if args.metadata:
        try:
            metadata = json_mod.loads(args.metadata)
            if not isinstance(metadata, dict):
                raise ValueError("not a JSON object")
        except ValueError as exc:
            output_error(args, f"Invalid --metadata: {exc}", hint='Example: \'{"camera": "one"}\'')
            return

    tags = [t.strip() for t in args.tag.split(",") if t.strip()] if args.tag else None

    project = _load_project(args)
    if project is None:
        return

    try:
        # Upload without waiting so the reserved video ID is known even if a
        # later bounded wait times out; `wait_for_video_upload` continues it.
        status = project.upload_video(
            args.video_file,
            batch_name=args.batch,
            tag_names=tags,
            metadata=metadata,
            split=args.split,
            wait=False,
        )
    except ValueError as exc:
        output_error(args, str(exc), hint="Native video upload accepts original .mp4 and .mov files.")
        return
    except rfapi.RoboflowError as exc:
        # The failing call may come after the bytes were stored; a re-upload then
        # deduplicates onto that Source instead of creating a second one.
        output_api_error(
            args,
            exc,
            hint="Check the project type and plan limits. If the bytes were already stored, "
            "re-uploading the same file reuses that Source.",
        )
        return

    video_id = status.get("videoId")
    if args.wait and video_id and status.get("status") not in _TERMINAL_UPLOAD_STATES:
        status = _wait_for_upload(args, project, video_id)
        if status is None:
            return

    _emit_upload_status(args, status)


def _video_upload_status(args) -> None:  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output_api_error

    project = _load_project(args)
    if project is None:
        return

    if args.wait:
        status = _wait_for_upload(args, project, args.video_id)
        if status is None:
            return
    else:
        try:
            status = project.get_video_upload_status(args.video_id)
        except rfapi.RoboflowError as exc:
            output_api_error(args, exc, not_found_hint=_unknown_video_hint(args))
            return

    _emit_upload_status(args, status)
