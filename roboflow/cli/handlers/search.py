"""Search commands: query workspace images and export search results."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import argparse


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``search`` command."""
    search_parser = subparsers.add_parser("search", help="Search workspace images or export results as a dataset")
    search_parser.add_argument("query", help="Search query (e.g. 'tag:review' or '*')")
    search_parser.add_argument("--limit", type=int, default=50, help="Max results to return (default: 50)")
    search_parser.add_argument("--cursor", default=None, help="Continuation token for pagination")
    search_parser.add_argument("--fields", default=None, help="Comma-separated list of fields to include")
    search_parser.add_argument(
        "--export", action="store_true", default=False, help="Export search results as a dataset"
    )
    search_parser.add_argument(
        "-f", "--format", dest="format", default="coco", help="Annotation format for export (default: coco)"
    )
    search_parser.add_argument("-l", "--location", dest="location", default=None, help="Local directory for export")
    search_parser.add_argument(
        "-d", "--dataset", dest="dataset", default=None, help="Limit to a specific dataset (project slug)"
    )
    search_parser.add_argument("--name", dest="name", default=None, help="Optional name for the export")
    search_parser.add_argument(
        "--no-extract", dest="no_extract", action="store_true", default=False, help="Keep zip file, skip extraction"
    )
    search_parser.set_defaults(func=_search)


def _search(args: argparse.Namespace) -> None:
    import contextlib
    import io

    import roboflow
    from roboflow.cli._output import output_error

    try:
        # Suppress "loading Roboflow workspace..." messages that corrupt --json output
        quiet = getattr(args, "json", False) or getattr(args, "quiet", False)
        if quiet:
            with contextlib.redirect_stdout(io.StringIO()):
                rf = roboflow.Roboflow()
                workspace = rf.workspace(args.workspace)
        else:  # noqa: PLR5501
            rf = roboflow.Roboflow()
            workspace = rf.workspace(args.workspace)
    except Exception as exc:
        output_error(args, str(exc), exit_code=2)
        return

    if args.export:
        _do_export(args, workspace)
    else:
        _do_search(args, workspace)


def _do_search(args: argparse.Namespace, workspace: Any) -> None:
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


def _do_export(args: argparse.Namespace, workspace: Any) -> None:
    from roboflow.cli._output import output, output_error

    try:
        result_path = workspace.search_export(
            query=args.query,
            format=args.format,
            location=args.location,
            dataset=args.dataset,
            name=args.name,
            extract_zip=not args.no_extract,
        )
    except Exception as exc:
        output_error(args, str(exc))
        return

    data = {"status": "completed", "path": str(result_path)}
    output(args, data, text=f"Export completed: {result_path}")
