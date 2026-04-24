"""Version management commands: list, get, download, export, create."""

from __future__ import annotations

import re
from typing import Annotated, Optional

import typer

from roboflow.cli._compat import SortedGroup, ctx_to_args

version_app = typer.Typer(cls=SortedGroup, help="Manage dataset versions", no_args_is_help=True)


@version_app.command("list")
def list_versions(
    ctx: typer.Context,
    project: Annotated[str, typer.Option("-p", "--project", help="Project ID")] = ...,  # type: ignore[assignment]
) -> None:
    """List versions for a project."""
    args = ctx_to_args(ctx, project=project)
    _list_versions(args)


@version_app.command("get")
def get_version(
    ctx: typer.Context,
    version_num: Annotated[str, typer.Argument(help="Version number or shorthand (e.g. my-project/3)")],
    project: Annotated[Optional[str], typer.Option("-p", "--project", help="Project ID")] = None,
) -> None:
    """Show detailed info for a version."""
    args = ctx_to_args(ctx, version_num=version_num, project=project)
    _get_version(args)


@version_app.command("download")
def download(
    ctx: typer.Context,
    url_or_id: Annotated[str, typer.Argument(help="Dataset URL or shorthand (e.g. ws/project/3)")],
    format: Annotated[str, typer.Option("-f", "--format", help="Export format (default: voc)")] = "voc",
    location: Annotated[Optional[str], typer.Option("-l", "--location", help="Download location")] = None,
) -> None:
    """Download a dataset version."""
    args = ctx_to_args(ctx, url_or_id=url_or_id, format=format, location=location)
    _download(args)


@version_app.command("export")
def export(
    ctx: typer.Context,
    version_num: Annotated[str, typer.Argument(help="Version number")],
    project: Annotated[str, typer.Option("-p", "--project", help="Project ID")] = ...,  # type: ignore[assignment]
    format: Annotated[str, typer.Option("-f", "--format", help="Export format (default: voc)")] = "voc",
) -> None:
    """Trigger an async export."""
    args = ctx_to_args(ctx, version_num=version_num, project=project, format=format)
    _export(args)


@version_app.command("create")
def create(
    ctx: typer.Context,
    project: Annotated[str, typer.Option("-p", "--project", help="Project ID")] = ...,  # type: ignore[assignment]
    settings: Annotated[str, typer.Option(help="Path to JSON file with augmentation/preprocessing config")] = ...,  # type: ignore[assignment]
) -> None:
    """Create a new dataset version.

    Settings JSON example::

        {"augmentation": {"flip": {"horizontal": true, "vertical": false},
          "rotate": {"degrees": 15}, "brightness": {"percent": 25}},
         "preprocessing": {"auto-orient": true, "resize": {"width": 640,
          "height": 640, "format": "Stretch to"}}}

    See https://docs.roboflow.com/datasets/create-a-dataset-version for all options.
    """
    args = ctx_to_args(ctx, project=project, settings=settings)
    _create(args)


@version_app.command("delete")
def delete_version(
    ctx: typer.Context,
    version_ref: Annotated[
        str,
        typer.Argument(help="Version shorthand (e.g. ws/project/3 or project/3)"),
    ],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation prompt.")] = False,
) -> None:
    """Move a version to Trash (30-day retention; cancels its in-flight training)."""
    args = ctx_to_args(ctx, version_ref=version_ref, yes=yes)
    _delete_version(args)


@version_app.command("restore")
def restore_version_cmd(
    ctx: typer.Context,
    version_ref: Annotated[
        str,
        typer.Argument(help="Version shorthand (e.g. ws/project/3 or project/3)"),
    ],
) -> None:
    """Restore a version from Trash (parent project must be active)."""
    args = ctx_to_args(ctx, version_ref=version_ref)
    _restore_version(args)


# ---------------------------------------------------------------------------
# Business logic (unchanged from argparse version)
# ---------------------------------------------------------------------------


def _list_versions(args):  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error
    from roboflow.cli._resolver import resolve_resource
    from roboflow.cli._table import format_table
    from roboflow.config import load_roboflow_api_key

    try:
        workspace_url, project_slug, _ver = resolve_resource(args.project, workspace_override=args.workspace)
    except ValueError as exc:
        output_error(args, str(exc))
        return

    api_key = args.api_key or load_roboflow_api_key(workspace_url)
    if not api_key:
        output_error(args, "No API key found.", hint="Set ROBOFLOW_API_KEY or run 'roboflow auth login'.", exit_code=2)
        return

    try:
        project_data = rfapi.get_project(api_key, workspace_url, project_slug)
    except rfapi.RoboflowError as exc:
        output_error(args, str(exc), exit_code=3)
        return

    versions = project_data.get("versions", [])
    rows = []
    for v in versions:
        rows.append(
            {
                "id": v.get("id", ""),
                "name": v.get("name", ""),
                "images": v.get("images", 0),
                "splits": _format_splits(v.get("splits", {})),
                "created": v.get("created", ""),
            }
        )

    table = format_table(
        rows,
        columns=["id", "name", "images", "splits", "created"],
        headers=["ID", "NAME", "IMAGES", "SPLITS", "CREATED"],
    )
    output(args, versions, text=table)


def _format_splits(splits: dict) -> str:
    if not splits:
        return ""
    parts = []
    for key in ("train", "valid", "test"):
        count = splits.get(key, 0)
        if count:
            parts.append(f"{key}:{count}")
    return " ".join(parts)


def _get_version(args):  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error
    from roboflow.cli._resolver import resolve_resource
    from roboflow.config import load_roboflow_api_key

    # Build shorthand: if --project is given, combine with version_num
    shorthand = args.version_num
    if args.project:
        shorthand = f"{args.project}/{args.version_num}"

    try:
        workspace_url, project_slug, version_num = resolve_resource(shorthand, workspace_override=args.workspace)
    except ValueError as exc:
        output_error(args, str(exc))
        return

    if version_num is None:
        output_error(args, "Version number is required.", hint="Use e.g. 'version get 3 -p my-project'.")
        return

    api_key = args.api_key or load_roboflow_api_key(workspace_url)
    if not api_key:
        output_error(args, "No API key found.", hint="Set ROBOFLOW_API_KEY or run 'roboflow auth login'.", exit_code=2)
        return

    try:
        data = rfapi.get_version(api_key, workspace_url, project_slug, str(version_num))
    except rfapi.RoboflowError as exc:
        output_error(args, str(exc), exit_code=3)
        return

    import json

    output(args, data, text=json.dumps(data, indent=2, default=str))


def _parse_url(url: str) -> tuple:
    """Parse a Roboflow URL or shorthand into (workspace, project, version).

    Supports:
    - Full URLs: https://universe.roboflow.com/ws/proj/3
    - Three segments: ws/proj/3
    - Two segments: ws/proj  OR  proj/3 (numeric = version, uses default ws)
    - One segment: proj (uses default ws, no version)
    """
    # Try full URL first
    url_regex = r"(?:https?://)?(?:universe|app)\.roboflow\.(?:com|one)/([^/]+)/([^/]+)(?:/dataset)?(?:/(\d+))?"
    match = re.match(url_regex, url)
    if match:
        return match.group(1), match.group(2), match.group(3)

    # Non-URL shorthand: use resolve_resource for proper disambiguation
    from roboflow.cli._resolver import resolve_resource

    try:
        ws, proj, ver = resolve_resource(url, workspace_override=None)
        return ws, proj, str(ver) if ver is not None else None
    except ValueError:
        return None, None, None


def _download(args):  # noqa: ANN001
    import roboflow
    from roboflow.cli._output import output, output_error, suppress_sdk_output

    w, p, v = _parse_url(args.url_or_id)

    if not w or not p:
        output_error(args, f"Could not parse URL or shorthand: {args.url_or_id}")
        return

    # Always suppress SDK "loading..." noise during workspace/project init
    with suppress_sdk_output():
        try:
            rf = roboflow.Roboflow()
            project = rf.workspace(w).project(p)
        except Exception as exc:
            output_error(args, str(exc), exit_code=3)
            return

    try:
        if not v:
            versions = project.versions()
            if not versions:
                output_error(args, f"Project {p} does not have any versions.")
                return
            version_obj = versions[-1]
        else:
            version_obj = project.version(int(v))

        version_obj.download(args.format, location=args.location, overwrite=True)
    except SystemExit:
        raise
    except Exception as exc:
        output_error(args, str(exc), exit_code=3)
        return

    data = {
        "workspace": w,
        "project": p,
        "version": int(v) if v else version_obj.version,
        "format": args.format,
        "location": args.location or "",
    }
    output(args, data, text=f"Downloaded {w}/{p}/{data['version']} in {args.format} format")


def _export(args):  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error
    from roboflow.cli._resolver import resolve_resource
    from roboflow.config import load_roboflow_api_key

    shorthand = f"{args.project}/{args.version_num}"
    try:
        workspace_url, project_slug, version_num = resolve_resource(shorthand, workspace_override=args.workspace)
    except ValueError as exc:
        output_error(args, str(exc))
        return

    if version_num is None:
        output_error(args, "Version number is required.")
        return

    api_key = args.api_key or load_roboflow_api_key(workspace_url)
    if not api_key:
        output_error(args, "No API key found.", hint="Set ROBOFLOW_API_KEY or run 'roboflow auth login'.", exit_code=2)
        return

    try:
        data = rfapi.get_version_export(api_key, workspace_url, project_slug, str(version_num), args.format)
    except rfapi.RoboflowError as exc:
        output_error(args, str(exc), exit_code=3)
        return

    if data.get("ready") is False:
        progress = data.get("progress", 0)
        output(args, data, text=f"Export in progress ({progress:.0%})...")
    else:
        output(args, data, text=f"Export ready for {project_slug}/{version_num} in {args.format} format")


def _create(args):  # noqa: ANN001
    import json

    import roboflow
    from roboflow.cli._output import output, output_error, suppress_sdk_output
    from roboflow.cli._resolver import resolve_resource
    from roboflow.config import load_roboflow_api_key

    try:
        workspace_url, project_slug, _ver = resolve_resource(args.project, workspace_override=args.workspace)
    except ValueError as exc:
        output_error(args, str(exc))
        return

    api_key = args.api_key or load_roboflow_api_key(workspace_url)
    if not api_key:
        output_error(args, "No API key found.", hint="Set ROBOFLOW_API_KEY or run 'roboflow auth login'.", exit_code=2)
        return

    try:
        with open(args.settings) as f:
            settings = json.load(f)
    except FileNotFoundError:
        output_error(args, f"Settings file not found: {args.settings}")
        return
    except json.JSONDecodeError as exc:
        output_error(args, f"Invalid JSON in settings file: {exc}")
        return

    with suppress_sdk_output():
        try:
            rf = roboflow.Roboflow(api_key)
            project = rf.workspace(workspace_url).project(project_slug)
            version_id = project.generate_version(settings)
        except Exception as exc:
            output_error(args, str(exc))
            return

    # generate_version returns the version number/ID directly
    version_num = version_id if version_id else "unknown"

    data = {"status": "created", "project": project_slug, "version": version_num}
    output(args, data, text=f"Created version {version_num} for project {project_slug}")


def _delete_version(args):  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error
    from roboflow.cli._resolver import resolve_resource
    from roboflow.config import load_roboflow_api_key

    try:
        workspace_url, project_slug, version_num = resolve_resource(args.version_ref, workspace_override=args.workspace)
    except ValueError as exc:
        output_error(args, str(exc))
        return

    if version_num is None:
        output_error(args, "Version number is required (e.g. project/3 or ws/project/3).")
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
            f"Move version '{workspace_url}/{project_slug}/{version_num}' to Trash? "
            "(Retained for 30 days. Any in-flight training will be cancelled.)",
            default=False,
        )
        if not confirmed:
            output(args, {"cancelled": True}, text="Cancelled.")
            return

    try:
        data = rfapi.delete_version(api_key, workspace_url, project_slug, version_num)
    except rfapi.RoboflowError as exc:
        output_error(args, str(exc), exit_code=3)
        return

    output(
        args,
        data,
        text=f"Moved {workspace_url}/{project_slug}/{version_num} to Trash.",
    )


def _restore_version(args):  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error
    from roboflow.cli._resolver import resolve_resource
    from roboflow.config import load_roboflow_api_key

    try:
        workspace_url, project_slug, version_num = resolve_resource(args.version_ref, workspace_override=args.workspace)
    except ValueError as exc:
        output_error(args, str(exc))
        return

    if version_num is None:
        output_error(args, "Version number is required (e.g. project/3 or ws/project/3).")
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
        output_error(args, str(exc), exit_code=3)
        return

    versions = trash.get("sections", {}).get("versions", [])
    target = str(version_num)
    match = next(
        (v for v in versions if str(v.get("id")) == target and v.get("parentUrl") == project_slug),
        None,
    )
    if not match:
        output_error(
            args,
            f"Version '{workspace_url}/{project_slug}/{version_num}' is not in Trash.",
            hint="Use 'roboflow trash list' to see what can be restored.",
            exit_code=3,
        )
        return

    try:
        data = rfapi.restore_trash_item(
            api_key,
            workspace_url,
            "version",
            match["id"],
            parent_id=match.get("parentId"),
        )
    except rfapi.RoboflowError as exc:
        output_error(args, str(exc), exit_code=3)
        return

    output(
        args,
        data,
        text=f"Restored {workspace_url}/{project_slug}/{version_num} from Trash.",
    )
