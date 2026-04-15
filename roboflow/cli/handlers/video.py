"""Video inference commands."""

from __future__ import annotations

from typing import Annotated

import typer

from roboflow.cli._compat import SortedGroup, ctx_to_args

video_app = typer.Typer(cls=SortedGroup, help="Video inference operations", no_args_is_help=True)


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
    """Check video inference job status."""
    args = ctx_to_args(ctx, job_id=job_id)
    _video_status(args)


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
            model = version.model

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
