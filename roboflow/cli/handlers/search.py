"""Search commands: query workspace images and export search results."""

from __future__ import annotations

from typing import Annotated, Any, Optional

import typer

from roboflow.cli._compat import ctx_to_args


def search_command(app: typer.Typer) -> None:
    """Register the top-level ``search`` command on *app*."""

    @app.command("search")
    def search(
        ctx: typer.Context,
        query: Annotated[str, typer.Argument(help="Search query (e.g. 'tag:review' or '*')")],
        limit: Annotated[int, typer.Option(help="Max results to return")] = 50,
        cursor: Annotated[Optional[str], typer.Option(help="Continuation token for pagination")] = None,
        fields: Annotated[Optional[str], typer.Option(help="Comma-separated list of fields to include")] = None,
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
        """Search workspace images or export results as a dataset."""
        args = ctx_to_args(
            ctx,
            query=query,
            limit=limit,
            cursor=cursor,
            fields=fields,
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
            rf = roboflow.Roboflow()
            workspace = rf.workspace(args.workspace)
    except Exception as exc:
        output_error(args, str(exc), exit_code=2)
        return

    if args.export:
        _do_export(args, workspace)
    else:
        _do_search(args, workspace)


def _do_search(args: Any, workspace: Any) -> None:
    from roboflow.cli._output import output, output_error

    fields = args.fields.split(",") if args.fields else None
    try:
        result = workspace.search(
            query=args.query,
            page_size=args.limit,
            fields=fields,
            continuation_token=args.cursor,
        )
    except Exception as exc:
        output_error(args, str(exc))
        return

    results = result.get("results", [])
    total = result.get("total", len(results))
    token = result.get("continuationToken")

    data = {"results": results, "total": total}
    if token:
        data["cursor"] = token

    text_lines = [f"Found {total} result(s)."]
    for r in results:
        text_lines.append(f"  {r.get('filename', r.get('id', ''))}")
    if token:
        text_lines.append(f"\nNext page: --cursor {token}")

    output(args, data, text="\n".join(text_lines))


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
