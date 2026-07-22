"""Train commands: start training for a dataset version."""

from __future__ import annotations

from typing import Annotated, Optional

import typer

from roboflow.cli._compat import SortedGroup, ctx_to_args

train_app = typer.Typer(cls=SortedGroup, help="Train a model", invoke_without_command=True)


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


@train_app.command("cancel")
def cancel_training(
    ctx: typer.Context,
    target: Annotated[
        str,
        typer.Argument(
            help="Training to cancel as 'project/version' (e.g. 'my-project/3' or 'workspace/my-project/3')"
        ),
    ],
    continue_if_no_refund: Annotated[
        bool,
        typer.Option(
            "--continue-if-no-refund",
            help=(
                "Cancel even if the run is past the refund window. "
                "Default: false (server replies refund:false without cancelling)."
            ),
        ),
    ] = False,
) -> None:
    """Cancel an in-flight training run.

    Works for any architecture, including NAS sweeps in the mining or
    training phase. Server-side gate: only valid while the run is in-flight;
    a finished/failed run returns 409 CANNOT_CANCEL.
    """
    args = ctx_to_args(ctx, target=target, continue_if_no_refund=continue_if_no_refund)
    _cancel(args)


@train_app.command("stop")
def stop_training(
    ctx: typer.Context,
    target: Annotated[
        str,
        typer.Argument(help="Training to stop as 'project/version'"),
    ],
) -> None:
    """Request a graceful early-stop on an in-flight training run.

    Distinct from cancel: the run finishes the current phase (mining or
    training) instead of terminating immediately. Idempotent — calling
    stop on an already-stopped run is a no-op.
    """
    args = ctx_to_args(ctx, target=target)
    _stop(args)


@train_app.command("delete")
def delete_training(
    ctx: typer.Context,
    target: Annotated[
        str,
        typer.Argument(help="Training to delete as 'project/version'"),
    ],
    training_id: Annotated[
        Optional[str],
        typer.Option(
            "--training-id",
            help=(
                "Training id of the run to delete (versions can own several). Omit to target the version's sole run."
            ),
        ),
    ] = None,
) -> None:
    """Move a terminal training run to the workspace Trash (soft delete).

    The run and every model it produced disappear from listings but stay
    restorable for 30 days ('roboflow train restore' or the web Trash view),
    after which they are permanently deleted. In-flight runs are refused —
    stop or cancel first — as is the run backing the version's registered
    model. Permanent deletion is only available in the web UI's Trash view.
    """
    args = ctx_to_args(ctx, target=target, training_id=training_id)
    _delete(args)


@train_app.command("restore")
def restore_training(
    ctx: typer.Context,
    target: Annotated[
        str,
        typer.Argument(help="Version the trashed training belongs to, as 'project/version'"),
    ],
    training_id: Annotated[
        str,
        typer.Option(
            "--training-id",
            help="Training id of the trashed run to restore (required).",
        ),
    ],
) -> None:
    """Restore a trashed training run (and its models) back into listings.

    Fails while the parent project or version is itself in Trash — restore
    those first ('roboflow trash list' shows what is trashed).
    """
    args = ctx_to_args(ctx, target=target, training_id=training_id)
    _restore(args)


@train_app.command("results")
def training_results(
    ctx: typer.Context,
    target: Annotated[
        str,
        typer.Argument(help="Training to inspect as 'project/version'"),
    ],
) -> None:
    """Run-level training results bundle.

    For NAS sweeps returns { trainingId, status, modelGroup, modelCount,
    recommendedByHardware, mining?, models: [...] }. For non-NAS trainings
    returns a minimal bundle with the produced model.

    Pass the returned `modelGroup` to `roboflow model list --group ...` to
    list every NAS model from that run with full metadata.
    """
    args = ctx_to_args(ctx, target=target)
    _results(args)


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


def _resolve_train_target(args):
    """Parse '<project>/<version>' (or full 'workspace/<project>/<version>') and resolve api key.

    Returns (api_key, workspace_url, project_slug, version_str) or None if validation fails.
    """
    from roboflow.cli._output import output_error
    from roboflow.cli._resolver import resolve_resource
    from roboflow.config import load_roboflow_api_key

    try:
        workspace_url, project_slug, version = resolve_resource(args.target, workspace_override=args.workspace)
    except ValueError as exc:
        output_error(args, str(exc))
        return None
    if version is None:
        output_error(
            args,
            "Version is required.",
            hint="Pass it as 'project/version' or 'workspace/project/version'.",
        )
        return None
    api_key = args.api_key or load_roboflow_api_key(workspace_url)
    if not api_key:
        output_error(
            args,
            "No API key found.",
            hint="Set ROBOFLOW_API_KEY or run 'roboflow auth login'.",
            exit_code=2,
        )
        return None
    return api_key, workspace_url, project_slug, str(version)


def _cancel(args):  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error

    resolved = _resolve_train_target(args)
    if resolved is None:
        return
    api_key, workspace_url, project_slug, version_str = resolved

    try:
        result = rfapi.cancel_version_training(
            api_key,
            workspace_url,
            project_slug,
            version_str,
            continue_if_no_refund=getattr(args, "continue_if_no_refund", False),
        )
    except rfapi.RoboflowError as exc:
        msg = str(exc)
        # 409 from server lands here as a RoboflowError carrying the JSON
        # body; surface it with code "CANNOT_CANCEL" if present.
        hint = None
        if "non-running" in msg or "Cannot cancel" in msg:
            hint = (
                "Cancel only applies to in-flight runs. Check status with 'roboflow train results <project>/<version>'."
            )
        output_error(args, msg, hint=hint, exit_code=3)
        return

    output(
        args,
        {"status": "cancelled", "project": project_slug, "version": version_str, **(result or {})},
        text=f"Training cancelled for {project_slug} version {version_str}.",
    )


def _stop(args):  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error

    resolved = _resolve_train_target(args)
    if resolved is None:
        return
    api_key, workspace_url, project_slug, version_str = resolved

    try:
        result = rfapi.stop_version_training(api_key, workspace_url, project_slug, version_str)
    except rfapi.RoboflowError as exc:
        output_error(args, str(exc), exit_code=3)
        return

    output(
        args,
        {"status": "stop_requested", "project": project_slug, "version": version_str, **(result or {})},
        text=f"Early-stop requested for {project_slug} version {version_str}.",
    )


def _delete(args):  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error

    resolved = _resolve_train_target(args)
    if resolved is None:
        return
    api_key, workspace_url, project_slug, version_str = resolved

    try:
        result = rfapi.delete_version_training(
            api_key,
            workspace_url,
            project_slug,
            version_str,
            training_id=getattr(args, "training_id", None),
        )
    except rfapi.RoboflowError as exc:
        msg = str(exc)
        hint = None
        if "in progress" in msg:
            hint = "Stop or cancel the run first: 'roboflow train stop <project>/<version>'."
        elif "registered model" in msg:
            hint = "This run backs the version's registered model. Register another model first."
        elif "MULTIPLE_TRAININGS" in msg:
            hint = "This version owns several runs. Pass --training-id (see 'roboflow train results')."
        output_error(args, msg, hint=hint, exit_code=3)
        return

    output(
        args,
        {"status": "in_trash", "project": project_slug, "version": version_str, **(result or {})},
        text=(
            f"Training moved to Trash for {project_slug} version {version_str}. "
            "Restorable for 30 days via 'roboflow train restore'."
        ),
    )


def _restore(args):  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error

    resolved = _resolve_train_target(args)
    if resolved is None:
        return
    api_key, workspace_url, project_slug, version_str = resolved

    try:
        result = rfapi.restore_version_training(
            api_key,
            workspace_url,
            project_slug,
            version_str,
            training_id=args.training_id,
        )
    except rfapi.RoboflowError as exc:
        msg = str(exc)
        hint = None
        if "not in trash" in msg.lower():
            hint = "Only trashed runs can be restored. 'roboflow trash list' shows what is trashed."
        elif "in trash" in msg.lower():
            hint = "Restore the parent project/version first ('roboflow trash list')."
        output_error(args, msg, hint=hint, exit_code=3)
        return

    output(
        args,
        {"status": "restored", "project": project_slug, "version": version_str, **(result or {})},
        text=f"Training restored for {project_slug} version {version_str}.",
    )


def _results(args):  # noqa: ANN001

    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error

    resolved = _resolve_train_target(args)
    if resolved is None:
        return
    api_key, workspace_url, project_slug, version_str = resolved

    try:
        result = rfapi.get_training_results(api_key, workspace_url, project_slug, version_str)
    except rfapi.RoboflowError as exc:
        output_error(args, str(exc), exit_code=3)
        return

    job_type = result.get("jobType", "unknown")
    model_count = result.get("modelCount", 0)
    model_group = result.get("modelGroup")
    text_summary = (
        f"{job_type} run for {project_slug} v{version_str}: status={result.get('status')}, models={model_count}"
    )
    if model_group:
        text_summary += f", group={model_group}"
    output(args, result, text=text_summary)
