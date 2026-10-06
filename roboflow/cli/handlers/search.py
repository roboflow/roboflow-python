"""Search commands: query workspace media and export search results."""

from __future__ import annotations

from typing import Annotated, Any, Optional

import typer

from roboflow.cli._compat import ctx_to_args


def search_command(app: typer.Typer) -> None:
    """Register the top-level ``search`` command on *app*."""

    @app.command("search", hidden=True)
    def search(
        ctx: typer.Context,
        query: Annotated[str, typer.Argument(help="Search query (e.g. 'tag:review' or '*')")],
        limit: Annotated[int, typer.Option(help="Max results to return")] = 50,
        cursor: Annotated[Optional[str], typer.Option(help="Continuation token for pagination")] = None,
        fields: Annotated[Optional[str], typer.Option(help="Comma-separated list of fields to include")] = None,
        media_types: Annotated[
            Optional[str],
            typer.Option(
                "--media-types",
                help="Comma-separated media types to search: image, video, or image,video (default: image)",
            ),
        ] = None,
        export: Annotated[bool, typer.Option("--export", help="Export search results as a dataset")] = False,
        format: Annotated[str, typer.Option("-f", "--format", help="Annotation format for export")] = "coco",
        location: Annotated[Optional[str], typer.Option("-l", "--location", help="Local directory for export")] = None,
        dataset: Annotated[
            Optional[str], typer.Option("-d", "--dataset", help="Limit to a specific dataset (project slug)")
        ] = None,
        annotation_group: Annotated[
            Optional[str],
            typer.Option("-g", "--annotation-group", help="Limit export to a specific annotation group"),
        ] = None,
        name: Annotated[Optional[str], typer.Option(help="Optional name for the export")] = None,
        no_extract: Annotated[bool, typer.Option("--no-extract", help="Keep zip file, skip extraction")] = False,
    ) -> None:
        """Search workspace media or export results as a dataset.

        Searches images only unless --media-types asks otherwise.

        Examples:
            roboflow search "tag:review"
            roboflow search "*" --media-types video --fields id,filename,url
            roboflow search "tag:review" --media-types image,video
        """
        args = ctx_to_args(
            ctx,
            query=query,
            limit=limit,
            cursor=cursor,
            fields=fields,
            media_types=media_types,
            export=export,
            format=format,
            location=location,
            dataset=dataset,
            annotation_group=annotation_group,
            name=name,
            no_extract=no_extract,
        )
        _search(args)


def _search(args):  # noqa: ANN001
    import roboflow
    from roboflow.cli._output import output_error, suppress_sdk_output

    try:
        with suppress_sdk_output():
            # Forward the CLI --api-key; Roboflow() falls back to saved/env creds when None.
            rf = roboflow.Roboflow(api_key=args.api_key)
            workspace = rf.workspace(args.workspace)
    except Exception as exc:
        output_error(args, str(exc), exit_code=2)
        return

    if args.export:
        # `is not None` on purpose: an explicit but invalid value (e.g. --media-types "")
        # must not silently fall through to the export route.
        if getattr(args, "media_types", None) is not None:
            output_error(
                args,
                "--media-types is not supported with --export",
                hint="Drop --media-types to export, or omit --export to search with media type selection",
            )
            return
        _do_export(args, workspace)
    else:
        _do_search(args, workspace)


def _do_search(args: Any, workspace: Any) -> None:
    from roboflow.cli._output import output, output_error
    from roboflow.util.search_utils import parse_media_types_option

    fields = [field.strip() for field in args.fields.split(",") if field.strip()] if args.fields else None
    try:
        media_types = parse_media_types_option(getattr(args, "media_types", None))
    except ValueError as exc:
        output_error(args, str(exc), hint="Valid media types: image, video")
        return

    try:
        result = workspace.search(
            query=args.query,
            page_size=args.limit,
            fields=fields,
            continuation_token=args.cursor,
            media_types=media_types,
        )
    except Exception as exc:
        output_error(args, str(exc))
        return

    results = result.get("results", [])
    total = result.get("total", len(results))
    token = result.get("continuationToken")

    data = {"results": results, "total": total}
    if media_types:
        data["mediaTypes"] = media_types
    if token:
        data["cursor"] = token

    text_lines = [f"Found {total} result(s)."]
    for r in results:
        text_lines.append(f"  {_describe_hit(r)}")
    if token:
        text_lines.append(f"\nNext page: --cursor {token}")

    output(args, data, text="\n".join(text_lines))


def _describe_hit(hit: dict) -> str:
    """One text line per hit: label, media type, and the signed video URL when present."""
    label = hit.get("filename") or hit.get("name") or hit.get("id", "")
    parts = [str(label)]
    media_type = hit.get("mediaType")
    if media_type:
        parts.append(f"[{media_type}]")
    if hit.get("videoUrl"):
        parts.append(str(hit["videoUrl"]))
    return "  ".join(parts)


def _do_export(args: Any, workspace: Any) -> None:
    from roboflow.cli._output import output, output_error

    try:
        result_path = workspace.search_export(
            query=args.query,
            format=args.format,
            location=args.location,
            dataset=args.dataset,
            annotation_group=getattr(args, "annotation_group", None),
            name=args.name,
            extract_zip=not args.no_extract,
        )
    except Exception as exc:
        output_error(args, str(exc))
        return

    data = {"status": "completed", "path": str(result_path)}
    output(args, data, text=f"Export completed: {result_path}")
