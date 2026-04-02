"""Version management commands: list, get, download, export, create."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register ``version`` subcommand and its verbs."""
    version_parser = subparsers.add_parser("version", help="Manage dataset versions")
    version_subs = version_parser.add_subparsers(title="version commands", dest="version_command")

    # --- version list ---
    list_parser = version_subs.add_parser("list", help="List versions for a project")
    list_parser.add_argument("-p", "--project", dest="project", required=True, help="Project ID")
    list_parser.set_defaults(func=_list_versions)

    # --- version get ---
    get_parser = version_subs.add_parser("get", help="Show detailed info for a version")
    get_parser.add_argument("version_num", help="Version number or shorthand (e.g. my-project/3)")
    get_parser.add_argument("-p", "--project", dest="project", default=None, help="Project ID")
    get_parser.set_defaults(func=_get_version)

    # --- version download ---
    dl_parser = version_subs.add_parser("download", help="Download a dataset version")
    dl_parser.add_argument("url_or_id", help="Dataset URL or shorthand (e.g. ws/project/3)")
    dl_parser.add_argument("-f", "--format", dest="format", default="voc", help="Export format (default: voc)")
    dl_parser.add_argument("-l", "--location", dest="location", default=None, help="Download location")
    dl_parser.set_defaults(func=_download)

    # --- version export ---
    export_parser = version_subs.add_parser("export", help="Trigger an async export")
    export_parser.add_argument("version_num", help="Version number")
    export_parser.add_argument("-p", "--project", dest="project", required=True, help="Project ID")
    export_parser.add_argument("-f", "--format", dest="format", default="voc", help="Export format (default: voc)")
    export_parser.set_defaults(func=_export)

    # --- version create (stub) ---
    create_parser = version_subs.add_parser("create", help="Create a new version (coming soon)")
    create_parser.add_argument("-p", "--project", dest="project", required=True, help="Project ID")
    create_parser.add_argument("--settings", dest="settings", default=None, help="Version settings as JSON string")
    create_parser.set_defaults(func=_create)

    # Default when no verb is given
    version_parser.set_defaults(func=lambda args: version_parser.print_help())


def _list_versions(args: argparse.Namespace) -> None:
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


def _get_version(args: argparse.Namespace) -> None:
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
    """Parse a Roboflow URL or shorthand into (workspace, project, version)."""
    regex = (
        r"(?:https?://)?(?:universe|app)\.roboflow\.(?:com|one)/([^/]+)/([^/]+)"
        r"(?:/dataset)?(?:/(\d+))?"
        r"|([^/]+)/([^/]+)(?:/(\d+))?"
    )
    match = re.match(regex, url)
    if match:
        organization = match.group(1) or match.group(4)
        dataset = match.group(2) or match.group(5)
        version = match.group(3) or match.group(6)
        return organization, dataset, version
    return None, None, None


def _download(args: argparse.Namespace) -> None:
    import roboflow
    from roboflow.cli._output import output, output_error, suppress_sdk_output

    w, p, v = _parse_url(args.url_or_id)

    if not w or not p:
        output_error(args, f"Could not parse URL or shorthand: {args.url_or_id}")
        return

    with suppress_sdk_output(args):
        try:
            rf = roboflow.Roboflow()
            project = rf.workspace(w).project(p)

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


def _export(args: argparse.Namespace) -> None:
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


def _create(args: argparse.Namespace) -> None:
    print("version create is not yet implemented")
