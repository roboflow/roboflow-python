"""Universe search commands."""

from __future__ import annotations

from typing import Annotated, Optional

import typer

from roboflow.cli._compat import SortedGroup, ctx_to_args

universe_app = typer.Typer(cls=SortedGroup, help="Browse Roboflow Universe", no_args_is_help=True)


@universe_app.command("search")
def search(
    ctx: typer.Context,
    query: Annotated[str, typer.Argument(help="Search query")],
    type: Annotated[Optional[str], typer.Option(help="Filter by type (dataset or model)")] = None,
    limit: Annotated[int, typer.Option(help="Max results")] = 12,
) -> None:
    """Search Roboflow Universe."""
    args = ctx_to_args(ctx, query=query, type=type, limit=limit)
    _search(args)


# ---------------------------------------------------------------------------
# Business logic (unchanged from argparse version)
# ---------------------------------------------------------------------------


def _search(args) -> None:  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error
    from roboflow.cli._table import format_table
    from roboflow.config import load_roboflow_api_key

    api_key = args.api_key or load_roboflow_api_key(None)

    try:
        data = rfapi.search_universe(args.query, api_key=api_key, project_type=args.type, limit=args.limit)
    except rfapi.RoboflowError as exc:
        output_error(args, str(exc))
        return

    results = data.get("results", [])
    # The API may ignore the limit param; enforce it client-side
    if args.limit and len(results) > args.limit:
        results = results[: args.limit]
    rows = []
    for r in results:
        rows.append(
            {
                "name": r.get("name", r.get("id", "")),
                "type": r.get("type", ""),
                "images": r.get("images", 0),
                "url": r.get("url", ""),
            }
        )

    table = format_table(rows, columns=["name", "type", "images", "url"], headers=["NAME", "TYPE", "IMAGES", "URL"])
    output(args, results, text=table)
