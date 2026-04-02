"""Top-level backwards-compatibility aliases.

Registers convenience commands at the root level (``roboflow login``,
``roboflow upload``, etc.) that delegate to the canonical noun-verb handlers.

This module is loaded *after* all other handlers by ``build_parser()`` so
that it can import their handler functions.
"""

from __future__ import annotations

import argparse

# Use SUPPRESS to hide legacy aliases from --help output
_HIDDEN = argparse.SUPPRESS


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register top-level aliases for common commands."""

    # --- roboflow login (visible alias for auth login) ---
    from roboflow.cli.handlers.auth import _login

    login_p = subparsers.add_parser("login", help="Log in to Roboflow (alias for 'auth login')")
    login_p.add_argument("--api-key", dest="login_api_key", default=None, help="API key (skip interactive login)")
    login_p.add_argument("--force", "-f", action="store_true", help="Force re-login")
    login_p.set_defaults(func=_login)

    # --- roboflow whoami (visible alias for auth status) ---
    from roboflow.cli.handlers.auth import _status

    whoami_p = subparsers.add_parser("whoami", help="Show current user (alias for 'auth status')")
    whoami_p.set_defaults(func=_status)

    # --- roboflow upload (visible alias for image upload) ---
    from roboflow.cli.handlers.image import _handle_upload

    upload_p = subparsers.add_parser("upload", help="Upload images to a project (alias for 'image upload')")
    upload_p.add_argument("path", help="Path to image file or directory")
    upload_p.add_argument("-p", "--project", dest="project", help="Project ID (required)", required=True)
    upload_p.add_argument("-a", "--annotation", dest="annotation", help="Path to annotation file")
    upload_p.add_argument("-m", "--labelmap", dest="labelmap", help="Path to labelmap file")
    upload_p.add_argument("-s", "--split", dest="split", default="train", help="Split (train/valid/test)")
    upload_p.add_argument("-r", "--retries", dest="num_retries", type=int, default=0, help="Retry count")
    upload_p.add_argument("-b", "--batch", dest="batch", help="Batch name")
    upload_p.add_argument("-t", "--tag", dest="tag_names", help="Comma-separated tag names")
    upload_p.add_argument("--metadata", dest="metadata", help="JSON metadata string")
    upload_p.add_argument("-c", "--concurrency", dest="concurrency", type=int, default=10, help="Upload concurrency")
    upload_p.add_argument("--is-prediction", dest="is_prediction", action="store_true", help="Mark as prediction")
    upload_p.set_defaults(func=_handle_upload)

    # --- roboflow import (hidden alias for image upload with directory) ---
    from roboflow.cli.handlers.image import _handle_upload as _handle_import

    import_p = subparsers.add_parser("import", help="Import dataset from folder (alias for 'image upload')")
    import_p.add_argument("path", metavar="folder", help="Path to dataset folder")
    import_p.add_argument("-p", "--project", dest="project", help="Project ID (required)", required=True)
    import_p.add_argument("-c", "--concurrency", dest="concurrency", type=int, default=10, help="Upload concurrency")
    import_p.add_argument("-n", "--batch-name", dest="batch", help="Batch name")
    import_p.add_argument("-r", "--retries", dest="num_retries", type=int, default=0, help="Retry count")
    import_p.set_defaults(func=_handle_import)

    # --- roboflow download (visible alias for version download) ---
    from roboflow.cli.handlers.version import _download

    download_p = subparsers.add_parser("download", help="Download a dataset version (alias for 'version download')")
    download_p.add_argument("url_or_id", metavar="datasetUrl", help="Dataset URL (e.g. workspace/project/version)")
    download_p.add_argument("-f", "--format", dest="format", default="voc", help="Export format")
    download_p.add_argument("-l", "--location", dest="location", help="Download location")
    download_p.set_defaults(func=_download)

    # --- roboflow search-export (hidden alias for search --export) ---
    from roboflow.cli.handlers.search import _search as _search_handler

    search_export_p = subparsers.add_parser("search-export", help=_HIDDEN)
    search_export_p.add_argument("query", help="Search query (e.g. 'tag:annotate' or '*')")
    search_export_p.add_argument("-f", dest="format", default="coco", help="Annotation format")
    search_export_p.add_argument("-l", dest="location", help="Local directory for export")
    search_export_p.add_argument("-d", dest="dataset", help="Limit to specific dataset")
    search_export_p.add_argument("-g", dest="annotation_group", help="Limit to annotation group")
    search_export_p.add_argument("-n", dest="name", help="Export name")
    search_export_p.add_argument(
        "--no-extract", dest="no_extract", action="store_true", help="Keep zip, skip extraction"
    )
    search_export_p.set_defaults(func=_search_handler, export=True)  # Force --export mode

    # --- roboflow upload_model (hidden alias for model upload) ---
    from roboflow.cli.handlers.model import _upload_model

    upload_model_p = subparsers.add_parser("upload_model", help=_HIDDEN)
    upload_model_p.add_argument("-a", dest="api_key", help="API key")
    upload_model_p.add_argument("-p", dest="project", action="append", help="Project ID")
    upload_model_p.add_argument("-v", dest="version_number", type=int, default=None, help="Version number")
    upload_model_p.add_argument("-t", dest="model_type", help="Model type")
    upload_model_p.add_argument("-m", dest="model_path", help="Model file path")
    upload_model_p.add_argument("-f", dest="filename", default="weights/best.pt", help="Model filename")
    upload_model_p.add_argument("-n", dest="model_name", help="Model name")
    upload_model_p.set_defaults(func=_upload_model)

    # --- roboflow get_workspace_info (hidden alias, preserved) ---
    get_ws_info_p = subparsers.add_parser("get_workspace_info", help=_HIDDEN)
    get_ws_info_p.add_argument("-a", dest="api_key", help="API key")
    get_ws_info_p.add_argument("-p", dest="project", help="Project ID")
    get_ws_info_p.add_argument("-v", dest="version_number", type=int, help="Version number")
    get_ws_info_p.set_defaults(func=_get_workspace_info_compat)

    # --- roboflow run_video_inference_api (hidden alias for video infer) ---
    from roboflow.cli.handlers.video import _video_infer

    video_api_p = subparsers.add_parser("run_video_inference_api", help=_HIDDEN)
    video_api_p.add_argument("-a", dest="api_key", help="API key")
    video_api_p.add_argument("-p", dest="project", help="Project ID")
    video_api_p.add_argument("-v", dest="version_number", type=int, help="Version number")
    video_api_p.add_argument("-f", dest="video_file", help="Video file path")
    video_api_p.add_argument("-fps", dest="fps", type=int, default=5, help="FPS")
    video_api_p.set_defaults(func=_video_infer)


def _get_workspace_info_compat(args: argparse.Namespace) -> None:
    """Backwards-compat handler for the old get_workspace_info command."""
    import roboflow

    rf = roboflow.Roboflow(args.api_key)
    workspace = rf.workspace()
    print("workspace", workspace)
    project = workspace.project(args.project)
    print("project", project)
    version = project.version(args.version_number)
    print("version", version)
