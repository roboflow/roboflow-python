"""Model management commands: list, get, upload."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register ``model`` subcommand and its verbs."""
    model_parser = subparsers.add_parser("model", help="Manage trained models")
    model_subs = model_parser.add_subparsers(title="model commands", dest="model_command")

    # --- model list ---
    list_parser = model_subs.add_parser("list", help="List trained models for a project")
    list_parser.add_argument(
        "-p",
        "--project",
        dest="project",
        required=True,
        help="Project ID or shorthand (e.g. my-ws/my-project)",
    )
    list_parser.set_defaults(func=_list_models)

    # --- model get ---
    get_parser = model_subs.add_parser("get", help="Show details for a trained model")
    get_parser.add_argument(
        "model_url",
        help="Model URL (e.g. workspace/model-name)",
    )
    get_parser.set_defaults(func=_get_model)

    # --- model upload ---
    upload_parser = model_subs.add_parser("upload", help="Upload a trained model")
    upload_parser.add_argument(
        "-p",
        "--project",
        dest="project",
        action="append",
        help="Project ID (can be specified multiple times for multi-project deploy)",
    )
    upload_parser.add_argument(
        "-v",
        "--version",
        dest="version_number",
        type=int,
        default=None,
        help="Version number to deploy to (for single-version deploy)",
    )
    upload_parser.add_argument(
        "-t",
        "--type",
        dest="model_type",
        required=True,
        help="Model type (e.g. yolov8, yolov5)",
    )
    upload_parser.add_argument(
        "-m",
        "--model-path",
        dest="model_path",
        required=True,
        help="Path to the trained model file",
    )
    upload_parser.add_argument(
        "-f",
        "--filename",
        dest="filename",
        default="weights/best.pt",
        help="Name of the model file (default: weights/best.pt)",
    )
    upload_parser.add_argument(
        "-n",
        "--model-name",
        dest="model_name",
        default=None,
        help="Name for the model (used in multi-project deploy)",
    )
    upload_parser.set_defaults(func=_upload_model)

    # Default when no verb is given
    model_parser.set_defaults(func=lambda args: model_parser.print_help())


def _list_models(args: argparse.Namespace) -> None:
    import roboflow
    from roboflow.cli._output import output, output_error
    from roboflow.cli._resolver import resolve_resource
    from roboflow.cli._table import format_table

    try:
        workspace_url, project_slug, _version = resolve_resource(args.project, workspace_override=args.workspace)
    except ValueError as exc:
        output_error(args, str(exc))
        return

    api_key = args.api_key or None
    rf = roboflow.Roboflow(api_key=api_key)
    workspace = rf.workspace(workspace_url)
    project = workspace.project(project_slug)

    versions = project.versions()
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


def _get_model(args: argparse.Namespace) -> None:
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


def _upload_model(args: argparse.Namespace) -> None:
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
