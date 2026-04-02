"""Video inference commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``video`` command group."""
    video_parser = subparsers.add_parser("video", help="Video inference operations")
    video_subs = video_parser.add_subparsers(title="video commands", dest="video_command")

    # --- video infer ---
    infer_p = video_subs.add_parser("infer", help="Run video inference")
    infer_p.add_argument("-p", "--project", dest="project", required=True, help="Project ID")
    infer_p.add_argument("-v", "--version", dest="version_number", type=int, required=True, help="Model version number")
    infer_p.add_argument("-f", "--file", dest="video_file", required=True, help="Path to video file")
    infer_p.add_argument("--fps", dest="fps", type=int, default=5, help="Frames per second (default: 5)")
    infer_p.set_defaults(func=_video_infer)

    # --- video status ---
    status_p = video_subs.add_parser("status", help="Check video inference job status")
    status_p.add_argument("job_id", help="Job ID to check")
    status_p.set_defaults(func=_video_status)

    # Default
    video_parser.set_defaults(func=lambda args: video_parser.print_help())


def _video_infer(args: argparse.Namespace) -> None:
    import roboflow
    from roboflow.cli._output import output, output_error
    from roboflow.config import load_roboflow_api_key

    api_key = args.api_key or load_roboflow_api_key(None)
    if not api_key:
        output_error(args, "No API key found.", hint="Set ROBOFLOW_API_KEY or run 'roboflow auth login'.", exit_code=2)
        return

    try:
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


def _video_status(args: argparse.Namespace) -> None:
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
