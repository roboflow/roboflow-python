"""Train commands: start training for a dataset version."""

from __future__ import annotations

from typing import Annotated, Optional

import typer

from roboflow.cli._compat import ctx_to_args

train_app = typer.Typer(help="Train a model", invoke_without_command=True)


@train_app.callback(invoke_without_command=True)
def _train_callback(
    ctx: typer.Context,
    project: Annotated[Optional[str], typer.Option("-p", "--project", help="Project ID to train")] = None,
    version_number: Annotated[Optional[int], typer.Option("-v", "--version", help="Version number to train")] = None,
    model_type: Annotated[
        Optional[str], typer.Option("-t", "--type", help="Model type (e.g. rfdetr-nano, yolov8n)")
    ] = None,
    checkpoint: Annotated[Optional[str], typer.Option(help="Checkpoint to resume training from")] = None,
    speed: Annotated[Optional[str], typer.Option(help="Training speed preset")] = None,
    epochs: Annotated[Optional[int], typer.Option(help="Number of training epochs")] = None,
) -> None:
    """Train a model. When invoked without a subcommand, behaves like ``train start``."""
    if ctx.invoked_subcommand is not None:
        return
    # No subcommand — behave like `train start`
    if not project:
        from roboflow.cli._output import output_error

        args = ctx_to_args(ctx)
        output_error(args, "Project is required.", hint="Use -p/--project.")
        return
    if version_number is None:
        from roboflow.cli._output import output_error

        args = ctx_to_args(ctx)
        output_error(args, "Version is required.", hint="Use -v/--version.")
        return
    args = ctx_to_args(
        ctx,
        project=project,
        version_number=version_number,
        model_type=model_type,
        checkpoint=checkpoint,
        speed=speed,
        epochs=epochs,
    )
    _start(args)


@train_app.command("start")
def start_training(
    ctx: typer.Context,
    project: Annotated[str, typer.Option("-p", "--project", help="Project ID to train")],
    version_number: Annotated[int, typer.Option("-v", "--version", help="Version number to train")],
    model_type: Annotated[
        Optional[str], typer.Option("-t", "--type", help="Model type (e.g. rfdetr-nano, yolov8n)")
    ] = None,
    checkpoint: Annotated[Optional[str], typer.Option(help="Checkpoint to resume training from")] = None,
    speed: Annotated[Optional[str], typer.Option(help="Training speed preset")] = None,
    epochs: Annotated[Optional[int], typer.Option(help="Number of training epochs")] = None,
) -> None:
    """Start training for a dataset version."""
    args = ctx_to_args(
        ctx,
        project=project,
        version_number=version_number,
        model_type=model_type,
        checkpoint=checkpoint,
        speed=speed,
        epochs=epochs,
    )
    _start(args)


# ---------------------------------------------------------------------------
# Business logic (unchanged from argparse version)
# ---------------------------------------------------------------------------


def _start(args):  # noqa: ANN001
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

    # Ensure the version has the required export format before training
    if args.model_type:
        _ensure_export(args, api_key, workspace_url, project_slug, str(args.version_number), args.model_type)

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
        err_str = str(exc)
        if "Unknown error" in err_str:
            output_error(
                args,
                "Training failed. The server returned an unexpected error.",
                hint="Ensure the version is fully generated and exported. "
                "Run 'roboflow version export -p <project> <version> -f coco' first.",
            )
        else:
            output_error(args, err_str)
        return

    data = {
        "status": "training_started",
        "project": project_slug,
        "version": args.version_number,
    }
    output(args, data, text=f"Training started for {project_slug} version {args.version_number}.")


def _ensure_export(args, api_key, workspace_url, project_slug, version_str, model_type):
    """Check if the version has the required export format; trigger and poll if not."""
    import sys
    import time

    from roboflow.adapters import rfapi
    from roboflow.util.versions import get_model_format

    required_format = get_model_format(model_type)

    try:
        version_data = rfapi.get_version(api_key, workspace_url, project_slug, version_str)
    except rfapi.RoboflowError:
        return  # Can't check; let the train call handle errors

    version_info = version_data.get("version", {})

    # Check if still generating
    if version_info.get("generating"):
        if not getattr(args, "quiet", False):
            print(f"Version is still generating ({version_info.get('progress', 0):.0%})... waiting.", file=sys.stderr)
        while True:
            time.sleep(5)
            try:
                version_data = rfapi.get_version(api_key, workspace_url, project_slug, version_str, nocache=True)
                version_info = version_data.get("version", {})
                if not version_info.get("generating"):
                    break
                if not getattr(args, "quiet", False):
                    print(
                        f"  Generating... {version_info.get('progress', 0):.0%}",
                        file=sys.stderr,
                    )
            except rfapi.RoboflowError:
                break

    # Check if export exists
    exports = version_info.get("exports", [])
    if required_format not in exports:
        if not getattr(args, "quiet", False):
            print(
                f"Exporting version in {required_format} format (required for {model_type})...",
                file=sys.stderr,
            )
        try:
            rfapi.get_version_export(api_key, workspace_url, project_slug, version_str, required_format)
        except rfapi.RoboflowError:
            pass  # Export may have been triggered; poll below

        # Poll until export is ready
        for _ in range(120):  # Up to 10 minutes
            time.sleep(5)
            try:
                version_data = rfapi.get_version(api_key, workspace_url, project_slug, version_str, nocache=True)
                current_exports = version_data.get("version", {}).get("exports", [])
                if required_format in current_exports:
                    if not getattr(args, "quiet", False):
                        print("  Export complete.", file=sys.stderr)
                    return
            except rfapi.RoboflowError:
                pass
