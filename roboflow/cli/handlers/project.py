"""Project management commands: list, get, create."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register ``project`` subcommand and its verbs."""
    project_parser = subparsers.add_parser("project", help="Manage projects")
    project_subs = project_parser.add_subparsers(title="project commands", dest="project_command")

    # --- project list ---
    list_parser = project_subs.add_parser("list", help="List projects in a workspace")
    list_parser.add_argument("--type", dest="type", default=None, help="Filter by project type")
    list_parser.set_defaults(func=_list_projects)

    # --- project get ---
    get_parser = project_subs.add_parser("get", help="Show detailed info for a project")
    get_parser.add_argument("project_id", help="Project ID or shorthand (e.g. my-ws/my-project)")
    get_parser.set_defaults(func=_get_project)

    # --- project create ---
    create_parser = project_subs.add_parser("create", help="Create a new project")
    create_parser.add_argument("name", help="Project name")
    create_parser.add_argument(
        "--type",
        dest="type",
        required=True,
        choices=[
            "object-detection",
            "single-label-classification",
            "multi-label-classification",
            "instance-segmentation",
            "semantic-segmentation",
            "keypoint-detection",
        ],
        help="Project type",
    )
    create_parser.add_argument("--license", dest="license", default="Private", help="Project license")
    create_parser.add_argument("--annotation", dest="annotation", default="", help="Annotation group name")
    create_parser.set_defaults(func=_create_project)

    # Default when no verb is given
    project_parser.set_defaults(func=lambda args: project_parser.print_help())


def _list_projects(args: argparse.Namespace) -> None:
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error
    from roboflow.cli._table import format_table
    from roboflow.config import load_roboflow_api_key

    workspace_url = args.workspace
    if not workspace_url:
        from roboflow.cli._resolver import resolve_default_workspace

        workspace_url = resolve_default_workspace(api_key=args.api_key)

    if not workspace_url:
        output_error(args, "No workspace specified.", hint="Use --workspace or run 'roboflow auth login'.")
        return

    api_key = args.api_key or load_roboflow_api_key(workspace_url)
    if not api_key:
        output_error(args, "No API key found.", hint="Set ROBOFLOW_API_KEY or run 'roboflow auth login'.", exit_code=2)
        return

    try:
        data = rfapi.get_workspace(api_key, workspace_url)
    except rfapi.RoboflowError as exc:
        output_error(args, str(exc), exit_code=3)
        return

    projects = data.get("workspace", {}).get("projects", [])

    if args.type:
        projects = [p for p in projects if p.get("type") == args.type]

    table = format_table(
        projects,
        columns=["name", "id", "type", "versions", "images"],
        headers=["NAME", "ID", "TYPE", "VERSIONS", "IMAGES"],
    )
    output(args, projects, text=table)


def _get_project(args: argparse.Namespace) -> None:
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error
    from roboflow.cli._resolver import resolve_resource
    from roboflow.config import load_roboflow_api_key

    try:
        workspace_url, project_slug, _version = resolve_resource(args.project_id, workspace_override=args.workspace)
    except ValueError as exc:
        output_error(args, str(exc))
        return

    api_key = args.api_key or load_roboflow_api_key(workspace_url)
    if not api_key:
        output_error(args, "No API key found.", hint="Set ROBOFLOW_API_KEY or run 'roboflow auth login'.", exit_code=2)
        return

    try:
        data = rfapi.get_project(api_key, workspace_url, project_slug)
    except rfapi.RoboflowError as exc:
        output_error(args, str(exc), exit_code=3)
        return

    project = data.get("project", data)
    lines = []
    field_map = [
        ("Name", "name"),
        ("ID", "id"),
        ("Type", "type"),
        ("License", "license"),
        ("Annotation", "annotation"),
        ("Classes", "classes"),
        ("Images", "images"),
        ("Versions", "versions"),
        ("Created", "created"),
        ("Updated", "updated"),
        ("Public", "public"),
    ]
    epoch_keys = {"created", "updated"}
    for label, key in field_map:
        if key in project:
            val = project[key]
            if key in epoch_keys and isinstance(val, (int, float)):
                import datetime

                val = datetime.datetime.fromtimestamp(val).strftime("%Y-%m-%d %H:%M:%S")
            elif isinstance(val, dict):
                val = ", ".join(f"{k}: {v}" for k, v in val.items())
            lines.append(f"  {label:12s} {val}")
    text = "\n".join(lines) if lines else "(no project details)"

    output(args, data, text=text)


def _create_project(args: argparse.Namespace) -> None:
    import roboflow
    from roboflow.cli._output import output, output_error, suppress_sdk_output

    annotation = args.annotation if args.annotation else args.name

    with suppress_sdk_output(args):
        try:
            rf = roboflow.Roboflow()
            workspace = rf.workspace(args.workspace)
        except Exception as exc:
            output_error(args, str(exc))
            return

    try:
        project = workspace.create_project(
            project_name=args.name,
            project_type=args.type,
            project_license=args.license,
            annotation=annotation,
        )
    except Exception as exc:
        msg = str(exc)
        hint = None
        # Try to extract a useful message from HTTP 422 responses
        if hasattr(exc, "response"):
            try:
                body = exc.response.json()  # type: ignore[union-attr]
                if "error" in body:
                    hint = body["error"].get("message", None) if isinstance(body["error"], dict) else str(body["error"])
                elif "message" in body:
                    hint = str(body["message"])
            except Exception:
                pass
        output_error(args, msg, hint=hint)
        return

    data = {
        "id": project.id,
        "name": project.name,
        "type": project.type,
    }
    output(args, data, text=f"Created project: {project.name} ({project.id})")
