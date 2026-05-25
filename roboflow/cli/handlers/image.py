"""Image management commands: upload, get, search, tag, delete, annotate."""

from __future__ import annotations

from typing import Annotated, Optional

import typer

from roboflow.cli._compat import SortedGroup, ctx_to_args

image_app = typer.Typer(cls=SortedGroup, help="Image management commands", no_args_is_help=True)


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
    split: Annotated[
        Optional[str],
        typer.Option(
            "-s",
            "--split",
            help="Override split for all images (default: infer from folder for dirs, 'train' for files)",
        ),
    ] = None,
    batch: Annotated[Optional[str], typer.Option("-b", "--batch", help="Batch name")] = None,
    tag: Annotated[Optional[str], typer.Option("-t", "--tag", help="Comma-separated tag names")] = None,
    metadata: Annotated[Optional[str], typer.Option(help="JSON string of key-value metadata")] = None,
    concurrency: Annotated[int, typer.Option("-c", "--concurrency", help="Concurrency for directory import")] = 10,
    retries: Annotated[int, typer.Option("-r", "--retries", help="Retry failed uploads N times")] = 0,
    labelmap: Annotated[Optional[str], typer.Option(help="Path to labelmap file")] = None,
    is_prediction: Annotated[bool, typer.Option("--is-prediction", help="Mark upload as prediction")] = False,
    zip_upload: Annotated[
        bool,
        typer.Option("--zip-upload", help="Zip the directory client-side and use the async zip upload flow"),
    ] = False,
    no_wait: Annotated[
        bool,
        typer.Option("--no-wait", help="Zip flow: return immediately with task_id instead of polling"),
    ] = False,
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
        zip_upload=zip_upload,
        no_wait=no_wait,
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
    query: Annotated[str, typer.Argument(help="RoboQL search query (e.g. 'tag:review' or '*')")],
    project: Annotated[
        Optional[str], typer.Option("-p", "--project", help="Project ID (omit to search entire workspace)")
    ] = None,
    limit: Annotated[int, typer.Option(help="Number of results")] = 50,
    cursor: Annotated[Optional[str], typer.Option(help="Continuation token for pagination")] = None,
    export: Annotated[bool, typer.Option("--export", help="Export search results as a dataset")] = False,
    format: Annotated[str, typer.Option("-f", "--format", help="Annotation format for export")] = "coco",
    location: Annotated[Optional[str], typer.Option("-l", "--location", help="Local directory for export")] = None,
    dataset: Annotated[
        Optional[str], typer.Option("-d", "--dataset", help="Limit export to a specific dataset")
    ] = None,
    annotation_group: Annotated[
        Optional[str], typer.Option("-g", "--annotation-group", help="Annotation group")
    ] = None,
    name: Annotated[Optional[str], typer.Option(help="Optional name for the export")] = None,
    no_extract: Annotated[bool, typer.Option("--no-extract", help="Keep zip file, skip extraction")] = False,
) -> None:
    """Search images in workspace or project.

    Without -p/--project, searches across the entire workspace using RoboQL.
    With -p/--project, searches within a specific project.
    Use --export to download matching results as a dataset.
    """
    if project:
        # _handle_search scopes by injecting a `project:<slug>` RoboQL filter.
        args = ctx_to_args(ctx, query=query, project=project, limit=limit, cursor=cursor)
        _handle_search(args)
    elif export:
        # Workspace-level export
        from roboflow.cli.handlers.search import _search

        args = ctx_to_args(
            ctx,
            query=query,
            limit=limit,
            cursor=cursor,
            export=True,
            format=format,
            location=location,
            dataset=dataset,
            annotation_group=annotation_group,
            name=name,
            no_extract=no_extract,
        )
        _search(args)
    else:
        # Workspace-level search
        from roboflow.cli.handlers.search import _search

        args = ctx_to_args(
            ctx,
            query=query,
            limit=limit,
            cursor=cursor,
            export=False,
            format=format,
            location=location,
            dataset=dataset,
            annotation_group=annotation_group,
            name=name,
            no_extract=no_extract,
            fields=None,
        )
        _search(args)


def _metadata_command(
    ctx: typer.Context,
    image_ids: str,
    metadata: Optional[str] = None,
    remove_metadata: Optional[str] = None,
    tags: Optional[str] = None,
    remove_tags: Optional[str] = None,
    poll: bool = False,
    timeout: int = 1800,
) -> None:
    """Update metadata and/or tags on existing images.

    Single image ID: updates synchronously.
    Multiple comma-separated IDs: uses the batch async endpoint.
    """
    args = ctx_to_args(
        ctx,
        image_ids=image_ids,
        metadata=metadata,
        remove_metadata=remove_metadata,
        add_tags=tags,
        remove_tags=remove_tags,
        poll=poll,
        timeout=timeout,
    )
    _handle_metadata(args)


@image_app.command("metadata")
def metadata_image(
    ctx: typer.Context,
    image_ids: Annotated[str, typer.Argument(help="Comma-separated image IDs (batch mode if multiple)")],
    metadata: Annotated[
        Optional[str], typer.Option("-m", "--metadata", help="JSON string of key-value metadata to set")
    ] = None,
    remove_metadata: Annotated[
        Optional[str], typer.Option("--remove-metadata", help="Comma-separated metadata keys to remove")
    ] = None,
    tags: Annotated[Optional[str], typer.Option("--tags", help="Comma-separated tags to add")] = None,
    remove_tags: Annotated[Optional[str], typer.Option("--remove-tags", help="Comma-separated tags to remove")] = None,
    poll: Annotated[bool, typer.Option("--poll/--no-poll", help="For batch updates: poll until complete")] = False,
    timeout: Annotated[int, typer.Option("--timeout", help="Polling timeout in seconds")] = 1800,
) -> None:
    """Update metadata and/or tags on existing images."""
    _metadata_command(ctx, image_ids, metadata, remove_metadata, tags, remove_tags, poll, timeout)


@image_app.command("tag", hidden=True)
def tag_image(
    ctx: typer.Context,
    image_ids: Annotated[str, typer.Argument(help="Comma-separated image IDs (batch mode if multiple)")],
    metadata: Annotated[
        Optional[str], typer.Option("-m", "--metadata", help="JSON string of key-value metadata to set")
    ] = None,
    remove_metadata: Annotated[
        Optional[str], typer.Option("--remove-metadata", help="Comma-separated metadata keys to remove")
    ] = None,
    tags: Annotated[Optional[str], typer.Option("--tags", help="Comma-separated tags to add")] = None,
    remove_tags: Annotated[Optional[str], typer.Option("--remove-tags", help="Comma-separated tags to remove")] = None,
    poll: Annotated[bool, typer.Option("--poll/--no-poll", help="For batch updates: poll until complete")] = False,
    timeout: Annotated[int, typer.Option("--timeout", help="Polling timeout in seconds")] = 1800,
) -> None:
    """Alias for 'metadata'."""
    _metadata_command(ctx, image_ids, metadata, remove_metadata, tags, remove_tags, poll, timeout)


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
    if os.path.isdir(path) or (os.path.isfile(path) and path.lower().endswith(".zip")):
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
            split=args.split or "train",
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
    tag_raw = getattr(args, "tag", None)
    tags = [t.strip() for t in tag_raw.split(",") if t.strip()] if tag_raw else None
    wait = not getattr(args, "no_wait", False)

    try:
        result = workspace.upload_dataset(
            dataset_path=path,
            project_name=args.project,
            num_workers=args.concurrency,
            batch_name=getattr(args, "batch", None),
            num_retries=retries,
            is_prediction=getattr(args, "is_prediction", False),
            use_zip_upload=getattr(args, "zip_upload", False),
            split=getattr(args, "split", None),
            tags=tags,
            wait=wait,
        )
    except Exception as exc:
        output_error(args, str(exc))
        return

    if isinstance(result, dict):
        status = result.get("status", "unknown")
        data = {
            "status": status,
            "task_id": result.get("task_id") or result.get("taskId"),
            "path": path,
            "project": args.project,
            "result": result,
        }
        output(args, data, text=f"Imported {path} to {args.project} (zip upload, status={status})")
        return

    # Per-image fallback — count files via image extensions
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

    # search/v1 only scopes via a `project:<slug>` RoboQL filter (body params are
    # ignored). Leading space = implicit AND; `AND (...)` 500s on free-text queries.
    query = args.query
    project = getattr(args, "project", None)
    if project:
        query = f"project:{project} {args.query}"

    result = rfapi.workspace_search(
        api_key=api_key,
        workspace_url=workspace_url,
        query=query,
        page_size=args.limit,
        continuation_token=args.cursor,
    )
    output(args, result, text=json.dumps(result, indent=2))


def _handle_metadata(args):  # noqa: ANN001
    import json as json_mod

    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error
    from roboflow.cli._resolver import resolve_ws_and_key

    ids = [i.strip() for i in args.image_ids.split(",") if i.strip()]
    if not ids:
        output_error(args, "No image IDs provided")
        return

    metadata_dict = None
    if args.metadata:
        try:
            metadata_dict = json_mod.loads(args.metadata)
            if not isinstance(metadata_dict, dict):
                output_error(args, "Metadata must be a JSON object", hint='Example: \'{"key": "value"}\'')
                return
        except json_mod.JSONDecodeError as exc:
            output_error(args, f"Invalid metadata JSON: {exc}", hint='Example: \'{"key": "value"}\'')
            return

    remove_meta_list = (
        [k.strip() for k in args.remove_metadata.split(",") if k.strip()] if args.remove_metadata else None
    )
    add_tags_list = [t.strip() for t in args.add_tags.split(",") if t.strip()] if args.add_tags else None
    remove_tags_list = [t.strip() for t in args.remove_tags.split(",") if t.strip()] if args.remove_tags else None

    if metadata_dict is None and remove_meta_list is None and add_tags_list is None and remove_tags_list is None:
        output_error(
            args,
            "Nothing to update",
            hint="Specify at least one of --metadata, --remove-metadata, --tags, --remove-tags",
        )
        return

    resolved = resolve_ws_and_key(args)
    if not resolved:
        return
    workspace_url, api_key = resolved

    if len(ids) == 1:
        try:
            rfapi.update_image_metadata(
                api_key=api_key,
                workspace_url=workspace_url,
                image_id=ids[0],
                metadata=metadata_dict,
                remove_metadata=remove_meta_list,
                add_tags=add_tags_list,
                remove_tags=remove_tags_list,
            )
        except rfapi.RoboflowError as exc:
            output_error(args, str(exc), exit_code=1)
            return
        data = {"success": True, "imageId": ids[0]}
        output(args, data, text=f"Updated image {ids[0]}")
    else:
        _handle_metadata_batch(
            args, api_key, workspace_url, ids, metadata_dict, remove_meta_list, add_tags_list, remove_tags_list
        )


def _handle_metadata_batch(args, api_key, workspace_url, image_ids, metadata, remove_metadata, add_tags, remove_tags):  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error

    BATCH_LIMIT = 1000  # matches the workspace images/metadata endpoint limit
    if len(image_ids) > BATCH_LIMIT:
        output_error(
            args,
            f"Too many images: {len(image_ids)} (limit: {BATCH_LIMIT})",
            hint=f"Split into batches of {BATCH_LIMIT} or fewer",
        )
        return

    updates = []
    for img_id in image_ids:
        entry: dict = {"imageId": img_id}
        if metadata:
            entry["metadata"] = metadata
        if remove_metadata:
            entry["removeMetadata"] = remove_metadata
        if add_tags:
            entry["addTags"] = add_tags
        if remove_tags:
            entry["removeTags"] = remove_tags
        updates.append(entry)

    try:
        result = rfapi.batch_update_image_metadata(
            api_key=api_key,
            workspace_url=workspace_url,
            updates=updates,
        )
    except rfapi.RoboflowError as exc:
        output_error(args, str(exc), exit_code=1)
        return

    task_id = result.get("taskId")
    polling_url = result.get("url")

    if not args.poll:
        data = {"taskId": task_id, "url": polling_url, "imageCount": len(image_ids)}
        output(args, data, text=f"Batch update started: taskId={task_id} ({len(image_ids)} images)")
        return

    from roboflow.core.async_tasks import poll_until_terminal

    try:
        final = poll_until_terminal(
            api_key,
            workspace_url,
            task_id,
            timeout=args.timeout,
            polling_url=polling_url,
        )
    except rfapi.RoboflowError as exc:
        output_error(args, str(exc), exit_code=1)
        return
    except TimeoutError as exc:
        output_error(args, str(exc), exit_code=1)
        return

    result_data = final.get("result", {})
    data = {"taskId": task_id, "status": final.get("status"), **result_data}
    succeeded = result_data.get("succeeded", 0)
    failed = result_data.get("failed", 0)
    output(args, data, text=f"Batch update complete: {succeeded} succeeded, {failed} failed (taskId={task_id})")


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
