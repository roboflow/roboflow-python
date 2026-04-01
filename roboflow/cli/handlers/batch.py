"""Batch processing commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse


def _stub(args: argparse.Namespace) -> None:
    from roboflow.cli._output import output_error

    output_error(args, "This command is not yet implemented.", hint="Coming soon.", exit_code=1)


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``batch`` command group."""
    batch_parser = subparsers.add_parser("batch", help="Batch processing operations")
    batch_subs = batch_parser.add_subparsers(title="batch commands", dest="batch_command")

    # --- batch create ---
    create_p = batch_subs.add_parser("create", help="Create a batch processing job")
    create_p.set_defaults(func=_stub)

    # --- batch status ---
    status_p = batch_subs.add_parser("status", help="Check batch job status")
    status_p.add_argument("job_id", help="Batch job ID")
    status_p.set_defaults(func=_stub)

    # --- batch list ---
    list_p = batch_subs.add_parser("list", help="List batch jobs")
    list_p.set_defaults(func=_stub)

    # --- batch results ---
    results_p = batch_subs.add_parser("results", help="Get batch job results")
    results_p.add_argument("job_id", help="Batch job ID")
    results_p.set_defaults(func=_stub)

    # Default
    batch_parser.set_defaults(func=lambda args: batch_parser.print_help())
