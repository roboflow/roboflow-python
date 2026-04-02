"""Folder management commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``folder`` command group."""
    from roboflow.cli._output import stub

    folder_parser = subparsers.add_parser("folder", help="Manage workspace folders")
    folder_subs = folder_parser.add_subparsers(title="folder commands", dest="folder_command")

    # --- folder list ---
    list_p = folder_subs.add_parser("list", help="List folders")
    list_p.set_defaults(func=stub)

    # --- folder get ---
    get_p = folder_subs.add_parser("get", help="Show folder details")
    get_p.add_argument("folder_id", help="Folder ID")
    get_p.set_defaults(func=stub)

    # --- folder create ---
    create_p = folder_subs.add_parser("create", help="Create a folder")
    create_p.add_argument("name", help="Folder name")
    create_p.set_defaults(func=stub)

    # --- folder update ---
    update_p = folder_subs.add_parser("update", help="Update a folder")
    update_p.add_argument("folder_id", help="Folder ID")
    update_p.add_argument("--name", help="New folder name")
    update_p.set_defaults(func=stub)

    # --- folder delete ---
    delete_p = folder_subs.add_parser("delete", help="Delete a folder")
    delete_p.add_argument("folder_id", help="Folder ID")
    delete_p.set_defaults(func=stub)

    # Default
    folder_parser.set_defaults(func=lambda args: folder_parser.print_help())
