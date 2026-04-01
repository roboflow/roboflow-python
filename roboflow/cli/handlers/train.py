"""Train commands: start training for a dataset version."""

from __future__ import annotations

import json as _json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse


def _extract_error_message(raw: str) -> str:
    """Extract a human-readable message from a potentially JSON-encoded error string."""
    # Strip status code prefix like "404: {...}"
    text = raw.strip()
    colon_idx = text.find(": {")
    if colon_idx > 0 and colon_idx < 5:
        text = text[colon_idx + 2 :]

    try:
        parsed = _json.loads(text)
        if isinstance(parsed, dict):
            err = parsed.get("error", parsed)
            if isinstance(err, dict):
                return str(err.get("message") or err.get("hint") or err)
            return str(err)
    except (ValueError, TypeError):
        pass
    return raw


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register ``train`` subcommand and its verbs."""
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_subs = train_parser.add_subparsers(title="train commands", dest="train_command")

    # --- train start ---
    start_parser = train_subs.add_parser("start", help="Start training for a dataset version")
    _add_start_args(start_parser, required=True)
    start_parser.set_defaults(func=_start)

    # Default: `train` without subcommand behaves like `train start`
    _add_start_args(train_parser, required=False)
    train_parser.set_defaults(func=_start)


def _add_start_args(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    """Add shared arguments for the train start command."""
    parser.add_argument(
        "-p",
        "--project",
        dest="project",
        required=required,
        help="Project ID to train",
    )
    parser.add_argument(
        "-v",
        "--version",
        dest="version_number",
        type=int,
        required=required,
        help="Version number to train",
    )
    parser.add_argument(
        "-t",
        "--type",
        dest="model_type",
        default=None,
        help="Model type (e.g. rfdetr-nano, yolov8n)",
    )
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint",
        default=None,
        help="Checkpoint to resume training from",
    )
    parser.add_argument(
        "--speed",
        dest="speed",
        default=None,
        help="Training speed preset",
    )
    parser.add_argument(
        "--epochs",
        dest="epochs",
        type=int,
        default=None,
        help="Number of training epochs",
    )


def _start(args: argparse.Namespace) -> None:
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error
    from roboflow.cli._resolver import resolve_resource
    from roboflow.config import load_roboflow_api_key

    if not getattr(args, "project", None):
        output_error(args, "Project is required.", hint="Use -p/--project.")
        return
    if getattr(args, "version_number", None) is None:
        output_error(args, "Version is required.", hint="Use -v/--version.")
        return

    try:
        workspace_url, project_slug, _version = resolve_resource(args.project, workspace_override=args.workspace)
    except ValueError as exc:
        output_error(args, str(exc))
        return

    api_key = args.api_key or load_roboflow_api_key(workspace_url)
    if not api_key:
        output_error(args, "No API key found.", hint="Set ROBOFLOW_API_KEY or run 'roboflow auth login'.", exit_code=2)
        return

    try:
        rfapi.start_version_training(
            api_key,
            workspace_url,
            project_slug,
            str(args.version_number),
            speed=args.speed,
            checkpoint=args.checkpoint,
            model_type=args.model_type,
            epochs=args.epochs,
        )
    except rfapi.RoboflowError as exc:
        output_error(args, _extract_error_message(str(exc)))
        return

    data = {
        "status": "training_started",
        "project": project_slug,
        "version": args.version_number,
    }
    output(args, data, text=f"Training started for {project_slug} version {args.version_number}.")
