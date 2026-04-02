"""Image management commands: upload, get, search, tag, delete, annotate."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

from roboflow.adapters import rfapi
from roboflow.cli._output import output, output_error
from roboflow.config import API_URL, load_roboflow_api_key

if TYPE_CHECKING:
    import argparse


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``image`` command group."""
    image_parser = subparsers.add_parser("image", help="Image management commands")
    image_sub = image_parser.add_subparsers(title="image commands", dest="image_command")

    _add_upload(image_sub)
    _add_get(image_sub)
    _add_search(image_sub)
    _add_tag(image_sub)
    _add_delete(image_sub)
    _add_annotate(image_sub)

    image_parser.set_defaults(func=lambda args: image_parser.print_help())


# ---------------------------------------------------------------------------
# upload
# ---------------------------------------------------------------------------


def _add_upload(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser("upload", help="Upload an image file or import a directory")
    p.add_argument("path", help="Path to image file or directory (auto-detects single file vs. directory bulk import)")
    p.add_argument("-p", "--project", required=True, help="Project ID")
    p.add_argument("-a", "--annotation", default=None, help="Path to annotation file (single upload)")
    p.add_argument("-s", "--split", default="train", help="Dataset split (default: train)")
    p.add_argument("-b", "--batch", default=None, help="Batch name")
    p.add_argument("-t", "--tag", default=None, help="Comma-separated tag names")
    p.add_argument("--metadata", default=None, help="JSON string of key-value metadata")
    p.add_argument("-c", "--concurrency", type=int, default=10, help="Concurrency for directory import (default: 10)")
    p.add_argument("-r", "--retries", type=int, default=0, help="Retry failed uploads N times (default: 0)")
    p.add_argument("--labelmap", default=None, help="Path to labelmap file")
    p.add_argument("--is-prediction", action="store_true", default=False, help="Mark upload as prediction")
    p.set_defaults(func=_handle_upload)


def _handle_upload(args: argparse.Namespace) -> None:
    api_key = args.api_key or load_roboflow_api_key(args.workspace)
    if not api_key:
        output_error(args, "No API key found", hint="Set ROBOFLOW_API_KEY or run 'roboflow auth login'", exit_code=2)
        return

    path = args.path
    if os.path.isdir(path):
        _handle_upload_directory(args, api_key, path)
    elif os.path.isfile(path):
        _handle_upload_single(args, api_key, path)
    else:
        output_error(args, f"Path not found: {path}", hint="Provide a valid file or directory path")
        return


def _handle_upload_single(args: argparse.Namespace, api_key: str, path: str) -> None:
    import roboflow
    from roboflow.cli._output import suppress_sdk_output

    metadata_raw = getattr(args, "metadata", None)
    metadata = json.loads(metadata_raw) if metadata_raw else None
    tag_raw = getattr(args, "tag", None) or getattr(args, "tag_names", None)
    tag_names = tag_raw.split(",") if tag_raw else []
    retries = getattr(args, "retries", None) or getattr(args, "num_retries", 0) or 0

    # Always suppress SDK "loading..." noise during workspace/project init
    with suppress_sdk_output():
        try:
            rf = roboflow.Roboflow(api_key)
            workspace = rf.workspace(args.workspace)
            project = workspace.project(args.project)
        except Exception as exc:
            output_error(args, str(exc), exit_code=3)
            return

    try:
        project.single_upload(
            image_path=path,
            annotation_path=args.annotation,
            annotation_labelmap=getattr(args, "labelmap", None),
            split=args.split,
            num_retry_uploads=retries,
            batch_name=args.batch,
            tag_names=tag_names,
            is_prediction=getattr(args, "is_prediction", False),
            metadata=metadata,
        )
    except Exception as exc:
        msg = str(exc)
        hint = None
        if "cannot identify image file" in msg:
            hint = "Supported formats: JPEG, PNG, BMP, GIF, TIFF, WebP."
        output_error(args, msg, hint=hint)
        return

    data = {"status": "uploaded", "path": path, "project": args.project}
    output(args, data, text=f"Uploaded {path} to {args.project}")


def _handle_upload_directory(args: argparse.Namespace, api_key: str, path: str) -> None:
    import roboflow
    from roboflow.cli._output import suppress_sdk_output

    # Always suppress SDK "loading..." noise during workspace init
    with suppress_sdk_output():
        try:
            rf = roboflow.Roboflow(api_key)
            workspace = rf.workspace(args.workspace)
        except Exception as exc:
            output_error(args, str(exc), exit_code=3)
            return

    retries = getattr(args, "retries", None) or getattr(args, "num_retries", 0) or 0

    try:
        workspace.upload_dataset(
            dataset_path=path,
            project_name=args.project,
            num_workers=args.concurrency,
            batch_name=getattr(args, "batch", None),
            num_retries=retries,
        )
    except Exception as exc:
        output_error(args, str(exc))
        return

    # Count files uploaded (approximate via image extensions)
    count = 0
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
    for root, _dirs, files in os.walk(path):
        for f in files:
            if os.path.splitext(f)[1].lower() in image_exts:
                count += 1

    data = {"status": "imported", "path": path, "count": count}
    output(args, data, text=f"Imported {count} images from {path} to {args.project}")


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------


def _add_get(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser("get", help="Get image details")
    p.add_argument("image_id", help="Image ID")
    p.add_argument("-p", "--project", required=True, help="Project ID")
    p.set_defaults(func=_handle_get)


def _handle_get(args: argparse.Namespace) -> None:
    import requests

    api_key = args.api_key or load_roboflow_api_key(args.workspace)
    if not api_key:
        output_error(args, "No API key found", hint="Set ROBOFLOW_API_KEY or run 'roboflow auth login'", exit_code=2)
        return

    workspace_url = args.workspace or _default_workspace()
    if not workspace_url:
        output_error(args, "No workspace specified", hint="Use --workspace or run 'roboflow auth login'")
        return

    url = f"{API_URL}/{workspace_url}/{args.project}/images/{args.image_id}?api_key={api_key}"
    response = requests.get(url)
    if response.status_code != 200:
        output_error(args, f"Failed to get image: {response.text}", exit_code=3)
        return

    data = response.json()
    output(args, data, text=json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def _add_search(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser("search", help="Search images in workspace")
    p.add_argument("query", help="RoboQL search query")
    p.add_argument("-p", "--project", required=True, help="Project ID (used in query filter)")
    p.add_argument("--limit", type=int, default=50, help="Number of results (default: 50)")
    p.add_argument("--cursor", default=None, help="Continuation token for pagination")
    p.set_defaults(func=_handle_search)


def _handle_search(args: argparse.Namespace) -> None:
    api_key = args.api_key or load_roboflow_api_key(args.workspace)
    if not api_key:
        output_error(args, "No API key found", hint="Set ROBOFLOW_API_KEY or run 'roboflow auth login'", exit_code=2)
        return

    workspace_url: str = args.workspace or _default_workspace() or ""
    if not workspace_url:
        output_error(args, "No workspace specified", hint="Use --workspace or run 'roboflow auth login'")
        return

    result = rfapi.workspace_search(
        api_key=api_key,
        workspace_url=workspace_url,
        query=args.query,
        page_size=args.limit,
        continuation_token=args.cursor,
    )
    output(args, result, text=json.dumps(result, indent=2))


# ---------------------------------------------------------------------------
# tag
# ---------------------------------------------------------------------------


def _add_tag(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser("tag", help="Add or remove tags on an image")
    p.add_argument("image_id", help="Image ID")
    p.add_argument("-p", "--project", required=True, help="Project ID")
    p.add_argument("--add", default=None, dest="add_tags", help="Comma-separated tags to add")
    p.add_argument("--remove", default=None, dest="remove_tags", help="Comma-separated tags to remove")
    p.set_defaults(func=_handle_tag)


def _handle_tag(args: argparse.Namespace) -> None:
    import requests

    if not args.add_tags and not args.remove_tags:
        output_error(args, "Nothing to do", hint="Specify --add and/or --remove with comma-separated tags")
        return

    api_key = args.api_key or load_roboflow_api_key(args.workspace)
    if not api_key:
        output_error(args, "No API key found", hint="Set ROBOFLOW_API_KEY or run 'roboflow auth login'", exit_code=2)
        return

    workspace_url = args.workspace or _default_workspace()
    if not workspace_url:
        output_error(args, "No workspace specified", hint="Use --workspace or run 'roboflow auth login'")
        return

    base = f"{API_URL}/{workspace_url}/{args.project}/images/{args.image_id}/tags"
    added = []
    removed = []

    if args.add_tags:
        for tag in args.add_tags.split(","):
            tag = tag.strip()
            if not tag:
                continue
            resp = requests.post(f"{base}?api_key={api_key}", json={"tag": tag})
            if resp.status_code == 200:
                added.append(tag)

    if args.remove_tags:
        for tag in args.remove_tags.split(","):
            tag = tag.strip()
            if not tag:
                continue
            resp = requests.delete(f"{base}/{tag}?api_key={api_key}")
            if resp.status_code == 200:
                removed.append(tag)

    data = {"added": added, "removed": removed}
    parts = []
    if added:
        parts.append(f"Added tags: {', '.join(added)}")
    if removed:
        parts.append(f"Removed tags: {', '.join(removed)}")
    text = "; ".join(parts) if parts else "No tags modified"
    output(args, data, text=text)


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


def _add_delete(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser("delete", help="Delete images from workspace")
    p.add_argument("image_ids", help="Comma-separated image IDs")
    p.add_argument("-p", "--project", required=True, help="Project ID")
    p.set_defaults(func=_handle_delete)


def _handle_delete(args: argparse.Namespace) -> None:
    api_key = args.api_key or load_roboflow_api_key(args.workspace)
    if not api_key:
        output_error(args, "No API key found", hint="Set ROBOFLOW_API_KEY or run 'roboflow auth login'", exit_code=2)
        return

    workspace_url: str = args.workspace or _default_workspace() or ""
    if not workspace_url:
        output_error(args, "No workspace specified", hint="Use --workspace or run 'roboflow auth login'")
        return

    ids = [i.strip() for i in args.image_ids.split(",") if i.strip()]
    result = rfapi.workspace_delete_images(
        api_key=api_key,
        workspace_url=workspace_url,
        image_ids=ids,
    )

    deleted = result.get("deleted", 0)
    skipped = result.get("skipped", 0)
    data = {"deleted": deleted, "skipped": skipped}
    output(args, data, text=f"Deleted {deleted}, skipped {skipped}")


# ---------------------------------------------------------------------------
# annotate
# ---------------------------------------------------------------------------


def _add_annotate(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser("annotate", help="Upload annotation for an image")
    p.add_argument("image_id", help="Image ID")
    p.add_argument("-p", "--project", required=True, help="Project ID")
    p.add_argument("--annotation-file", required=True, help="Path to annotation file")
    p.add_argument("--format", default=None, dest="annotation_format", help="Annotation format name")
    p.add_argument("--labelmap", default=None, help="Path to labelmap file")
    p.set_defaults(func=_handle_annotate)


def _handle_annotate(args: argparse.Namespace) -> None:
    api_key = args.api_key or load_roboflow_api_key(args.workspace)
    if not api_key:
        output_error(args, "No API key found", hint="Set ROBOFLOW_API_KEY or run 'roboflow auth login'", exit_code=2)
        return

    annotation_path = args.annotation_file
    if not os.path.isfile(annotation_path):
        output_error(args, f"Annotation file not found: {annotation_path}")
        return

    with open(annotation_path) as f:
        annotation_string = f.read()

    annotation_name = os.path.basename(annotation_path)
    labelmap = None
    if args.labelmap:
        with open(args.labelmap) as f:
            labelmap = json.load(f)

    rfapi.save_annotation(
        api_key=api_key,
        project_url=args.project,
        annotation_name=annotation_name,
        annotation_string=annotation_string,
        image_id=args.image_id,
        annotation_labelmap=labelmap,
    )

    data = {"status": "saved"}
    output(args, data, text=f"Annotation saved for image {args.image_id}")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _default_workspace() -> str | None:
    from roboflow.config import get_conditional_configuration_variable

    return get_conditional_configuration_variable("RF_WORKSPACE", default=None)
