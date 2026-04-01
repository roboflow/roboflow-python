"""Folder management commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse


def _stub(args: argparse.Namespace) -> None:
    from roboflow.cli._output import output_error

    output_error(args, "This command is not yet implemented.", hint="Coming soon.", exit_code=1)


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``folder`` command group."""
    folder_parser = subparsers.add_parser("folder", help="Manage workspace folders")
    folder_subs = folder_parser.add_subparsers(title="folder commands", dest="folder_command")

    # --- folder list ---
    list_p = folder_subs.add_parser("list", help="List folders")
    list_p.set_defaults(func=_stub)

    # --- folder get ---
    get_p = folder_subs.add_parser("get", help="Show folder details")
    get_p.add_argument("folder_id", help="Folder ID")
    get_p.set_defaults(func=_stub)

    # --- folder create ---
    create_p = folder_subs.add_parser("create", help="Create a folder")
    create_p.add_argument("name", help="Folder name")
    create_p.set_defaults(func=_stub)

    # --- folder update ---
    update_p = folder_subs.add_parser("update", help="Update a folder")
    update_p.add_argument("folder_id", help="Folder ID")
    update_p.add_argument("--name", help="New folder name")
    update_p.set_defaults(func=_stub)

    # --- folder delete ---
    delete_p = folder_subs.add_parser("delete", help="Delete a folder")
    delete_p.add_argument("folder_id", help="Folder ID")
    delete_p.set_defaults(func=_stub)

    # Default
    folder_parser.set_defaults(func=lambda args: folder_parser.print_help())
