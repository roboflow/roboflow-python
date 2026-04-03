"""Model management commands: list, get, upload."""

from __future__ import annotations

from typing import Annotated, Optional

import typer

from roboflow.cli._compat import ctx_to_args

model_app = typer.Typer(help="Manage trained models", no_args_is_help=True)


@model_app.command("list")
def list_models(
    ctx: typer.Context,
    project: Annotated[str, typer.Option("-p", "--project", help="Project ID or shorthand (e.g. my-ws/my-project)")],
) -> None:
    """List trained models for a project."""
    args = ctx_to_args(ctx, project=project)
    _list_models(args)


@model_app.command("get")
def get_model(
    ctx: typer.Context,
    model_url: Annotated[str, typer.Argument(help="Model URL (e.g. workspace/model-name)")],
) -> None:
    """Show details for a trained model."""
    args = ctx_to_args(ctx, model_url=model_url)
    _get_model(args)


@model_app.command("infer")
def model_infer(
    ctx: typer.Context,
    file: Annotated[str, typer.Argument(help="Path to an image file")],
    model: Annotated[str, typer.Option("-m", "--model", help="Model ID (project/version, e.g. my-project/3)")],
    confidence: Annotated[float, typer.Option("-c", "--confidence", help="Confidence threshold 0.0-1.0")] = 0.5,
    overlap: Annotated[float, typer.Option("-o", "--overlap", help="Overlap/NMS threshold 0.0-1.0")] = 0.5,
    type: Annotated[
        Optional[str],
        typer.Option("-t", "--type", help="Model type (auto-detected if not specified)"),
    ] = None,
) -> None:
    """Run inference on an image using a trained model."""
    from roboflow.cli.handlers.infer import _infer

    args = ctx_to_args(ctx, file=file, model=model, confidence=confidence, overlap=overlap, type=type)
    _infer(args)


@model_app.command("upload")
def upload_model(
    ctx: typer.Context,
    model_type: Annotated[str, typer.Option("-t", "--type", help="Model type (e.g. yolov8, yolov5)")],
    model_path: Annotated[str, typer.Option("-m", "--model-path", help="Path to the trained model file")],
    project: Annotated[
        Optional[list[str]], typer.Option("-p", "--project", help="Project ID (repeatable for multi-project deploy)")
    ] = None,
    version_number: Annotated[
        Optional[int], typer.Option("-v", "--version", help="Version number to deploy to (single-version deploy)")
    ] = None,
    filename: Annotated[str, typer.Option("-f", "--filename", help="Model file name")] = "weights/best.pt",
    model_name: Annotated[
        Optional[str], typer.Option("-n", "--model-name", help="Name for the model (multi-project deploy)")
    ] = None,
) -> None:
    """Upload a trained model."""
    args = ctx_to_args(
        ctx,
        project=project,
        version_number=version_number,
        model_type=model_type,
        model_path=model_path,
        filename=filename,
        model_name=model_name,
    )
    _upload_model(args)


# ---------------------------------------------------------------------------
# Business logic (unchanged from argparse version)
# ---------------------------------------------------------------------------


def _list_models(args):  # noqa: ANN001
    import roboflow
    from roboflow.cli._output import output, output_error, suppress_sdk_output
    from roboflow.cli._resolver import resolve_resource
    from roboflow.cli._table import format_table

    try:
        workspace_url, project_slug, _version = resolve_resource(args.project, workspace_override=args.workspace)
    except ValueError as exc:
        output_error(args, str(exc))
        return

    api_key = args.api_key or None

    try:
        with suppress_sdk_output(args):
            rf = roboflow.Roboflow(api_key=api_key)
            workspace = rf.workspace(workspace_url)
            project = workspace.project(project_slug)
            versions = project.versions()
    except Exception as exc:
        output_error(args, str(exc), exit_code=3)
        return

    models = []
    for v in versions:
        if v.model:
            models.append(
                {
                    "version": v.version,
                    "id": v.id,
                    "model": getattr(v, "model_format", ""),
                    "map": getattr(v, "model", {}).get("map", "")
                    if isinstance(getattr(v, "model", None), dict)
                    else "",
                }
            )

    table = format_table(
        models,
        columns=["version", "id", "model", "map"],
        headers=["VERSION", "ID", "MODEL", "MAP"],
    )
    output(args, models, text=table)


def _get_model(args):  # noqa: ANN001
    import json

    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error
    from roboflow.cli._resolver import resolve_resource
    from roboflow.config import load_roboflow_api_key

    try:
        workspace_url, project_slug, version = resolve_resource(args.model_url, workspace_override=args.workspace)
    except ValueError as exc:
        output_error(args, str(exc))
        return

    api_key = args.api_key or load_roboflow_api_key(workspace_url)
    if not api_key:
        output_error(args, "No API key found.", hint="Set ROBOFLOW_API_KEY or run 'roboflow auth login'.", exit_code=2)
        return

    try:
        if version is not None:
            data = rfapi.get_version(api_key, workspace_url, project_slug, str(version))
        else:
            data = rfapi.get_project(api_key, workspace_url, project_slug)
    except rfapi.RoboflowError as exc:
        output_error(args, str(exc), exit_code=3)
        return

    output(args, data, text=json.dumps(data, indent=2, default=str))


def _upload_model(args):  # noqa: ANN001
    import roboflow
    from roboflow.cli._output import output, output_error

    api_key = args.api_key or None
    rf = roboflow.Roboflow(api_key=api_key)
    workspace = rf.workspace(args.workspace)

    if args.version_number is not None:
        # Deploy to a specific version
        project_id = args.project[0] if isinstance(args.project, list) else args.project
        if not project_id:
            output_error(args, "Project is required for model upload.", hint="Use -p/--project.")
            return

        try:
            project = workspace.project(project_id)
            version = project.version(args.version_number)
            version.deploy(str(args.model_type), str(args.model_path), str(args.filename))
        except Exception as exc:
            output_error(args, str(exc))
            return
    else:
        # Deploy to multiple projects
        if not args.project:
            output_error(args, "At least one project is required.", hint="Use -p/--project.")
            return

        try:
            workspace.deploy_model(
                model_type=str(args.model_type),
                model_path=str(args.model_path),
                project_ids=args.project,
                model_name=str(args.model_name) if args.model_name else "",
                filename=str(args.filename),
            )
        except Exception as exc:
            output_error(args, str(exc))
            return

    output(args, {"status": "uploaded"}, text="Model uploaded successfully.")
