"""Image management commands: upload, get, search, tag, delete, annotate."""

from __future__ import annotations

from typing import Annotated, Optional

import typer

from roboflow.cli._compat import ctx_to_args

image_app = typer.Typer(help="Image management commands", no_args_is_help=True)


@image_app.command("upload")
def upload_image(
    ctx: typer.Context,
    path: Annotated[
        str, typer.Argument(help="Path to image file or directory (auto-detects single vs. directory bulk import)")
    ],
    project: Annotated[str, typer.Option("-p", "--project", help="Project ID")],
    annotation: Annotated[
        Optional[str], typer.Option("-a", "--annotation", help="Path to annotation file (single upload)")
    ] = None,
    split: Annotated[str, typer.Option("-s", "--split", help="Dataset split")] = "train",
    batch: Annotated[Optional[str], typer.Option("-b", "--batch", help="Batch name")] = None,
    tag: Annotated[Optional[str], typer.Option("-t", "--tag", help="Comma-separated tag names")] = None,
    metadata: Annotated[Optional[str], typer.Option(help="JSON string of key-value metadata")] = None,
    concurrency: Annotated[int, typer.Option("-c", "--concurrency", help="Concurrency for directory import")] = 10,
    retries: Annotated[int, typer.Option("-r", "--retries", help="Retry failed uploads N times")] = 0,
    labelmap: Annotated[Optional[str], typer.Option(help="Path to labelmap file")] = None,
    is_prediction: Annotated[bool, typer.Option("--is-prediction", help="Mark upload as prediction")] = False,
) -> None:
    """Upload an image file or import a directory."""
    args = ctx_to_args(
        ctx,
        path=path,
        project=project,
        annotation=annotation,
        split=split,
        batch=batch,
        tag=tag,
        metadata=metadata,
        concurrency=concurrency,
        retries=retries,
        labelmap=labelmap,
        is_prediction=is_prediction,
    )
    _handle_upload(args)


@image_app.command("get")
def get_image(
    ctx: typer.Context,
    image_id: Annotated[str, typer.Argument(help="Image ID")],
    project: Annotated[str, typer.Option("-p", "--project", help="Project ID")],
) -> None:
    """Get image details."""
    args = ctx_to_args(ctx, image_id=image_id, project=project)
    _handle_get(args)


@image_app.command("search")
def search_images(
    ctx: typer.Context,
    query: Annotated[str, typer.Argument(help="RoboQL search query")],
    project: Annotated[str, typer.Option("-p", "--project", help="Project ID (used in query filter)")],
    limit: Annotated[int, typer.Option(help="Number of results")] = 50,
    cursor: Annotated[Optional[str], typer.Option(help="Continuation token for pagination")] = None,
) -> None:
    """Search images in workspace."""
    args = ctx_to_args(ctx, query=query, project=project, limit=limit, cursor=cursor)
    _handle_search(args)


@image_app.command("tag")
def tag_image(
    ctx: typer.Context,
    image_id: Annotated[str, typer.Argument(help="Image ID")],
    project: Annotated[str, typer.Option("-p", "--project", help="Project ID")],
    add_tags: Annotated[Optional[str], typer.Option("--add", help="Comma-separated tags to add")] = None,
    remove_tags: Annotated[Optional[str], typer.Option("--remove", help="Comma-separated tags to remove")] = None,
) -> None:
    """Add or remove tags on an image."""
    args = ctx_to_args(ctx, image_id=image_id, project=project, add_tags=add_tags, remove_tags=remove_tags)
    _handle_tag(args)


@image_app.command("delete")
def delete_images(
    ctx: typer.Context,
    image_ids: Annotated[str, typer.Argument(help="Comma-separated image IDs")],
    project: Annotated[str, typer.Option("-p", "--project", help="Project ID")],
) -> None:
    """Delete images from workspace."""
    args = ctx_to_args(ctx, image_ids=image_ids, project=project)
    _handle_delete(args)


@image_app.command("annotate")
def annotate_image(
    ctx: typer.Context,
    image_id: Annotated[str, typer.Argument(help="Image ID")],
    project: Annotated[str, typer.Option("-p", "--project", help="Project ID")],
    annotation_file: Annotated[str, typer.Option("--annotation-file", help="Path to annotation file")],
    annotation_format: Annotated[Optional[str], typer.Option("--format", help="Annotation format name")] = None,
    labelmap: Annotated[Optional[str], typer.Option(help="Path to labelmap file")] = None,
) -> None:
    """Upload annotation for an image."""
    args = ctx_to_args(
        ctx,
        image_id=image_id,
        project=project,
        annotation_file=annotation_file,
        annotation_format=annotation_format,
        labelmap=labelmap,
    )
    _handle_annotate(args)


# ---------------------------------------------------------------------------
# Business logic (unchanged from argparse version)
# ---------------------------------------------------------------------------


def _handle_upload(args):  # noqa: ANN001
    import os

    from roboflow.cli._output import output_error
    from roboflow.config import load_roboflow_api_key

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


def _handle_upload_single(args, api_key: str, path: str) -> None:  # noqa: ANN001
    import json

    import roboflow
    from roboflow.cli._output import output, output_error, suppress_sdk_output

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


def _handle_upload_directory(args, api_key: str, path: str) -> None:  # noqa: ANN001
    import os

    import roboflow
    from roboflow.cli._output import output, output_error, suppress_sdk_output

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


def _handle_get(args):  # noqa: ANN001
    import json

    import requests

    from roboflow.cli._output import output, output_error
    from roboflow.config import API_URL, load_roboflow_api_key

    api_key = args.api_key or load_roboflow_api_key(args.workspace)
    if not api_key:
        output_error(args, "No API key found", hint="Set ROBOFLOW_API_KEY or run 'roboflow auth login'", exit_code=2)
        return

    workspace_url = args.workspace or _default_workspace()
    if not workspace_url:
        output_error(args, "No workspace specified", hint="Use --workspace or run 'roboflow auth login'")
        return

    url = f"{API_URL}/{workspace_url}/{args.project}/images/{args.image_id}"
    response = requests.get(url, params={"api_key": api_key})
    if response.status_code != 200:
        output_error(args, f"Failed to get image: {response.text}", exit_code=3)
        return

    data = response.json()
    output(args, data, text=json.dumps(data, indent=2))


def _handle_search(args):  # noqa: ANN001
    import json

    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error
    from roboflow.config import load_roboflow_api_key

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


def _handle_tag(args):  # noqa: ANN001
    import requests

    from roboflow.cli._output import output, output_error
    from roboflow.config import API_URL, load_roboflow_api_key

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
            resp = requests.post(base, params={"api_key": api_key}, json={"tag": tag})
            if resp.status_code == 200:
                added.append(tag)

    if args.remove_tags:
        for tag in args.remove_tags.split(","):
            tag = tag.strip()
            if not tag:
                continue
            resp = requests.delete(f"{base}/{tag}", params={"api_key": api_key})
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


def _handle_delete(args):  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error
    from roboflow.config import load_roboflow_api_key

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


def _handle_annotate(args):  # noqa: ANN001
    import json
    import os

    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error
    from roboflow.config import load_roboflow_api_key

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
