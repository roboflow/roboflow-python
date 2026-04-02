"""Universe search commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``universe`` command group."""
    from roboflow.cli._output import stub

    uni_parser = subparsers.add_parser("universe", help="Browse Roboflow Universe")
    uni_subs = uni_parser.add_subparsers(title="universe commands", dest="universe_command")

    # --- universe search ---
    search_p = uni_subs.add_parser("search", help="Search Roboflow Universe")
    search_p.add_argument("query", help="Search query")
    search_p.add_argument("--type", dest="type", choices=["dataset", "model"], default=None, help="Filter by type")
    search_p.add_argument("--limit", type=int, default=20, help="Max results (default: 20)")
    search_p.set_defaults(func=stub)

    # Default
    uni_parser.set_defaults(func=lambda args: uni_parser.print_help())
