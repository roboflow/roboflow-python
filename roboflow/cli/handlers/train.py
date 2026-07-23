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
    train_recipe: Annotated[
        Optional[str],
        typer.Option(
            "--train-recipe",
            help=(
                "Full trainRecipe as inline JSON or @path/to/file.json (see 'roboflow train "
                "recipe'); --epochs is folded into its hyperparameters unless the recipe "
                "already sets epochs"
            ),
        ),
    ] = None,
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
        train_recipe=train_recipe,
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
    train_recipe: Annotated[
        Optional[str],
        typer.Option(
            "--train-recipe",
            help=(
                "Full trainRecipe as inline JSON or @path/to/file.json (see 'roboflow train "
                "recipe'); --epochs is folded into its hyperparameters unless the recipe "
                "already sets epochs"
            ),
        ),
    ] = None,
) -> None:
    """Start training for a dataset version.

    With --train-recipe, the training is created via the v2 trainings API
    and the new trainingId is printed. Start from the ``template`` field of
    ``roboflow train recipe`` output, edit it (hyperparameters, online
    augmentation), and pass it inline or as ``@path/to/file.json``; --epochs is folded into its
    hyperparameters unless the recipe already sets epochs.
    """
    args = ctx_to_args(
        ctx,
        project=project,
        version_number=version_number,
        model_type=model_type,
        checkpoint=checkpoint,
        speed=speed,
        epochs=epochs,
        train_recipe=train_recipe,
    )
    _start(args)


@train_app.command("recipe")
def describe_train_recipe(
    ctx: typer.Context,
    project: Annotated[str, typer.Option("-p", "--project", help="Project ID")],
    version_number: Annotated[int, typer.Option("-v", "--version", help="Version number")],
    model_type: Annotated[
        str,
        typer.Option("-m", "--model-type", "-t", "--type", help="Model type to describe (e.g. rfdetr-medium)"),
    ],
) -> None:
    """Show the training recipe schema and template for a model type.

    Prints the tunable hyperparameter schema, the allowed online
    augmentation/preprocessing steps, and a ready-to-submit ``template``
    that can be edited and passed to ``roboflow train start --train-recipe``.
    """
    args = ctx_to_args(ctx, project=project, version_number=version_number, model_type=model_type)
    _recipe(args)


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

    # Custom recipes go through the v2 trainings API. Presence, not
    # truthiness: an explicitly supplied empty value (e.g. an unset shell
    # variable) must fail JSON validation, not fall through and start a
    # legacy training.
    if getattr(args, "train_recipe", None) is not None:
        _start_v2(args, api_key, workspace_url, project_slug)
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


def _parse_json_flag(args, raw, flag):
    """Parse a JSON-object CLI flag value; exits with a clean error on invalid input.

    Accepts inline JSON, or ``@path/to/file.json`` to read the JSON from a
    file (curl-style; unambiguous because ``@`` can never start valid JSON).
    """
    import json
    import os

    from roboflow.cli._output import output_error

    source = "string"
    if raw.startswith("@"):
        path = os.path.expanduser(raw[1:])
        try:
            with open(path, encoding="utf-8") as f:
                raw = f.read()
        except OSError as exc:
            output_error(
                args,
                f"Cannot read {flag} file {path}: {exc.strerror or exc}",
                hint="Pass inline JSON, or @<path> pointing to a readable JSON file.",
            )
            return None  # unreachable: output_error sys.exits
        source = "file"

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        output_error(args, f"Invalid JSON in {flag} {source}: {exc}", hint="Pass a valid JSON string.")
        return None  # unreachable: output_error sys.exits
    if not isinstance(parsed, dict):
        output_error(
            args,
            f"{flag} must be a JSON object, got {type(parsed).__name__}",
            hint="Pass a JSON object string, e.g. '{\"lr\": 0.0002}'.",
        )
        return None  # unreachable: output_error sys.exits
    return parsed


def _start_v2(args, api_key, workspace_url, project_slug):
    """Create a training via the v2 trainings API with a custom trainRecipe."""
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error
    from roboflow.util.train_recipe import fold_epochs_into_recipe

    version_str = str(args.version_number)
    if not args.model_type:
        output_error(
            args,
            "--train-recipe requires a model type.",
            hint=(
                "Recipes are minted per model type; without -t/--type the platform "
                "would train the project's default architecture. Pass the model type "
                "the recipe was described for (e.g. -t rfdetr-medium)."
            ),
        )
        return
    train_recipe = _parse_json_flag(args, args.train_recipe, "--train-recipe")
    if args.epochs is not None:
        # Fold --epochs into the recipe: the server dense-fills recipe
        # hyperparameters (including a default epochs) and resolves them
        # ahead of the body's top-level value, which would otherwise be
        # silently ignored. An epochs set in the recipe wins.
        train_recipe = fold_epochs_into_recipe(train_recipe, args.epochs)

    # Ensure the version has the required export format before training
    if args.model_type:
        _ensure_export(args, api_key, workspace_url, project_slug, version_str, args.model_type)

    try:
        result = rfapi.create_training_v2(
            api_key,
            workspace_url,
            project_slug,
            version_str,
            model_type=args.model_type,
            speed=args.speed,
            checkpoint=args.checkpoint,
            epochs=args.epochs,
            train_recipe=train_recipe,
        )
    except rfapi.RoboflowError as exc:
        output_error(args, str(exc))
        return

    data = {
        "status": "training_created",
        "project": project_slug,
        "version": args.version_number,
        **result,
    }
    training_id = result.get("trainingId")
    output(
        args,
        data,
        text=f"Training created for {project_slug} version {args.version_number}. trainingId: {training_id}",
    )


def _recipe(args):  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error
    from roboflow.cli._resolver import resolve_resource
    from roboflow.config import load_roboflow_api_key

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
        result = rfapi.get_train_recipe(
            api_key, workspace_url, project_slug, str(args.version_number), model_type=args.model_type
        )
    except rfapi.RoboflowError as exc:
        output_error(args, str(exc))
        return

    # No text form — the recipe is structured data; print JSON in both modes.
    output(args, result)


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
