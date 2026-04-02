"""Universe search commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``universe`` command group."""
    uni_parser = subparsers.add_parser("universe", help="Browse Roboflow Universe")
    uni_subs = uni_parser.add_subparsers(title="universe commands", dest="universe_command")

    # --- universe search ---
    search_p = uni_subs.add_parser("search", help="Search Roboflow Universe")
    search_p.add_argument("query", help="Search query")
    search_p.add_argument("--type", dest="type", choices=["dataset", "model"], default=None, help="Filter by type")
    search_p.add_argument("--limit", type=int, default=12, help="Max results (default: 12)")
    search_p.set_defaults(func=_search)

    # Default
    uni_parser.set_defaults(func=lambda args: uni_parser.print_help())


def _search(args: argparse.Namespace) -> None:
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error
    from roboflow.cli._table import format_table

    try:
        data = rfapi.search_universe(args.query, project_type=args.type, limit=args.limit)
    except rfapi.RoboflowError as exc:
        output_error(args, str(exc))
        return

    results = data.get("results", [])
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
