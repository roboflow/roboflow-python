"""Project management commands: list, get, create, health."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Optional

import typer

from roboflow.cli._compat import SortedGroup, ctx_to_args


class ProjectType(str, Enum):
    """Supported project types."""

    object_detection = "object-detection"
    single_label_classification = "single-label-classification"
    multi_label_classification = "multi-label-classification"
    instance_segmentation = "instance-segmentation"
    semantic_segmentation = "semantic-segmentation"
    keypoint_detection = "keypoint-detection"


project_app = typer.Typer(cls=SortedGroup, help="Manage projects", no_args_is_help=True)


@project_app.command("list")
def list_projects(
    ctx: typer.Context,
    type: Annotated[Optional[str], typer.Option(help="Filter by project type")] = None,
) -> None:
    """List projects in a workspace."""
    args = ctx_to_args(ctx, type=type)
    _list_projects(args)


@project_app.command("get")
def get_project(
    ctx: typer.Context,
    project_id: Annotated[str, typer.Argument(help="Project ID or shorthand (e.g. my-ws/my-project)")],
) -> None:
    """Show detailed info for a project."""
    args = ctx_to_args(ctx, project_id=project_id)
    _get_project(args)


@project_app.command("create")
def create_project(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Project name")],
    type: Annotated[ProjectType, typer.Option("--type", help="Project type")],
    license: Annotated[str, typer.Option(help="Project license")] = "Private",
    annotation: Annotated[str, typer.Option(help="Annotation group name")] = "",
) -> None:
    """Create a new project."""
    args = ctx_to_args(ctx, name=name, type=type.value, license=license, annotation=annotation)
    _create_project(args)


@project_app.command("delete")
def delete_project(
    ctx: typer.Context,
    project_id: Annotated[str, typer.Argument(help="Project ID or shorthand (e.g. my-ws/my-project)")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation prompt.")] = False,
) -> None:
    """Move a project to Trash (30-day retention; cancels in-flight trainings)."""
    args = ctx_to_args(ctx, project_id=project_id, yes=yes)
    _delete_project(args)


@project_app.command("restore")
def restore_project(
    ctx: typer.Context,
    project_id: Annotated[str, typer.Argument(help="Project ID or shorthand (e.g. my-ws/my-project)")],
) -> None:
    """Restore a project from Trash."""
    args = ctx_to_args(ctx, project_id=project_id)
    _restore_project(args)


@project_app.command("fork")
def fork_project(
    ctx: typer.Context,
    source: Annotated[
        str,
        typer.Argument(help="Source project: Universe URL or '<workspace>/<project>' shorthand."),
    ],
    no_wait: Annotated[
        bool,
        typer.Option("--no-wait", help="Return immediately with the taskId instead of waiting."),
    ] = False,
    timeout: Annotated[
        int,
        typer.Option("--timeout", help="Seconds to wait for completion (0 = no timeout)."),
    ] = 1800,
) -> None:
    """Fork a public Universe project into a workspace."""
    args = ctx_to_args(ctx, source=source, no_wait=no_wait, timeout=timeout)
    _fork_project(args)
@project_app.command("health")
def health_project(
    ctx: typer.Context,
    project_id: Annotated[str, typer.Argument(help="Project ID or shorthand (e.g. my-ws/my-project)")],
    regenerate: Annotated[
        bool, typer.Option("--regenerate", "-r", help="Force regeneration of health check data.")
    ] = False,
) -> None:
    """Show dataset health check for a project (class balance, dimensions, splits)."""
    args = ctx_to_args(ctx, project_id=project_id, regenerate=regenerate)
    _health_project(args)


# ---------------------------------------------------------------------------
# Business logic (unchanged from argparse version)
# ---------------------------------------------------------------------------


def _list_projects(args):  # noqa: ANN001
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


def _get_project(args):  # noqa: ANN001
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


def _create_project(args):  # noqa: ANN001
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


def _delete_project(args):  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error
    from roboflow.cli._resolver import resolve_resource
    from roboflow.config import load_roboflow_api_key

    try:
        workspace_url, project_slug, _version = resolve_resource(args.project_id, workspace_override=args.workspace)
    except ValueError as exc:
        output_error(
            args,
            str(exc),
            hint="Use 'my-workspace/my-project' or set --workspace and pass 'my-project'.",
        )
        return

    api_key = args.api_key or load_roboflow_api_key(workspace_url)
    if not api_key:
        output_error(
            args,
            "No API key found.",
            hint="Set ROBOFLOW_API_KEY or run 'roboflow auth login'.",
            exit_code=2,
        )
        return

    if not getattr(args, "yes", False) and not getattr(args, "json", False):
        import typer

        confirmed = typer.confirm(
            f"Move '{workspace_url}/{project_slug}' to Trash? "
            "(Retained for 30 days. Any in-flight trainings will be cancelled.)",
            default=False,
        )
        if not confirmed:
            output(args, {"cancelled": True}, text="Cancelled.")
            return

    try:
        data = rfapi.delete_project(api_key, workspace_url, project_slug)
    except rfapi.RoboflowError as exc:
        output_error(
            args,
            str(exc),
            hint="Check your API key has 'project:update' scope on this workspace.",
            exit_code=3,
        )
        return

    output(
        args,
        data,
        text=f"Moved {workspace_url}/{project_slug} to Trash (30-day retention).",
    )


def _restore_project(args):  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error
    from roboflow.cli._resolver import resolve_resource
    from roboflow.config import load_roboflow_api_key

    try:
        workspace_url, project_slug, _version = resolve_resource(args.project_id, workspace_override=args.workspace)
    except ValueError as exc:
        output_error(
            args,
            str(exc),
            hint="Use 'my-workspace/my-project' or set --workspace and pass 'my-project'.",
        )
        return

    api_key = args.api_key or load_roboflow_api_key(workspace_url)
    if not api_key:
        output_error(
            args,
            "No API key found.",
            hint="Set ROBOFLOW_API_KEY or run 'roboflow auth login'.",
            exit_code=2,
        )
        return

    try:
        trash = rfapi.list_trash(api_key, workspace_url)
    except rfapi.RoboflowError as exc:
        output_error(
            args,
            str(exc),
            hint="Check your API key has 'project:read' scope on this workspace.",
            exit_code=3,
        )
        return

    projects = trash.get("sections", {}).get("projects", [])
    match = next((p for p in projects if p.get("url") == project_slug), None)
    if not match:
        output_error(
            args,
            f"Project '{workspace_url}/{project_slug}' is not in Trash.",
            hint="Run 'roboflow trash list' to see what can be restored.",
            exit_code=3,
        )
        return

    try:
        data = rfapi.restore_trash_item(api_key, workspace_url, "project", match["id"])
    except rfapi.RoboflowError as exc:
        output_error(
            args,
            str(exc),
            hint="Check your API key has 'project:update' scope on this workspace.",
            exit_code=3,
        )
        return

    output(args, data, text=f"Restored {workspace_url}/{project_slug} from Trash.")


def _fork_project(args):  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error
    from roboflow.cli._resolver import resolve_default_workspace
    from roboflow.config import load_roboflow_api_key
    from roboflow.core.async_tasks import poll_until_terminal

    # The server accepts the full URL (or `<ws>/<proj>` shorthand) as `url`
    # and parses it itself — forward verbatim so the CLI doesn't duplicate
    # that logic.
    source = (args.source or "").strip()
    if not source:
        output_error(
            args,
            "Source is required.",
            hint="Use '<workspace>/<project>' or a Universe URL.",
        )
        return

    dest_workspace = args.workspace or resolve_default_workspace(api_key=args.api_key)
    if not dest_workspace:
        output_error(
            args,
            "No workspace specified.",
            hint="Use --workspace or run 'roboflow auth login'.",
            exit_code=2,
        )
        return

    api_key = args.api_key or load_roboflow_api_key(dest_workspace)
    if not api_key:
        output_error(
            args,
            "No API key found.",
            hint="Set ROBOFLOW_API_KEY or run 'roboflow auth login'.",
            exit_code=2,
        )
        return

    try:
        enqueued = rfapi.fork_project(api_key, dest_workspace, url=source)
    except rfapi.RoboflowError as exc:
        output_error(args, str(exc))
        return

    task_id = enqueued["taskId"]

    if args.no_wait:
        output(args, enqueued, text=f"Fork enqueued: taskId={task_id}")
        return

    def _print_progress(status):  # noqa: ANN001
        if args.json:
            return
        progress = status.get("progress")
        if not isinstance(progress, dict):
            return
        # Don't use `or` here: `current == 0` is a legitimate value.
        current = progress["current"] if "current" in progress else progress.get("completed")
        total = progress.get("total")
        if current is not None and total is not None:
            print(f"Task progress: {current}/{total}", flush=True)

    try:
        final = poll_until_terminal(
            api_key,
            dest_workspace,
            task_id,
            timeout=args.timeout,
            on_update=_print_progress,
            polling_url=enqueued.get("url"),
        )
    except rfapi.RoboflowError as exc:
        output_error(args, str(exc))
        return
    except TimeoutError as exc:
        output_error(args, str(exc))
        return

    if final.get("status") == "failed":
        output_error(args, final.get("error") or "Fork task failed.")
        return

    project_url = (final.get("result") or {}).get("url", "")
    text = f"Forked.\nDestination URL: {project_url}" if project_url else "Forked."
    output(args, final, text=text)
def _health_project(args):  # noqa: ANN001
    import json

    import roboflow
    from roboflow.cli._output import output, output_error, suppress_sdk_output

    with suppress_sdk_output(args):
        try:
            rf = roboflow.Roboflow(api_key=args.api_key)
            project = rf.workspace(args.workspace).project(args.project_id)
        except Exception as exc:
            output_error(args, str(exc))
            return

    try:
        data = project.health(regenerate=args.regenerate)
    except Exception as exc:
        output_error(args, str(exc), exit_code=3)
        return

    output(args, data, text=json.dumps(data, indent=2))
