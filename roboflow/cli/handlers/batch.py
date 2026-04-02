"""Batch processing commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``batch`` command group."""
    from roboflow.cli._output import stub

    batch_parser = subparsers.add_parser("batch", help="Batch processing operations")
    batch_subs = batch_parser.add_subparsers(title="batch commands", dest="batch_command")

    # --- batch create ---
    create_p = batch_subs.add_parser("create", help="Create a batch processing job")
    create_p.add_argument("--workflow", dest="workflow", required=True, help="Workflow ID to run")
    create_p.add_argument("--input", dest="input", required=True, help="Input path (image directory or video file)")
    create_p.add_argument("--model", dest="model", default=None, help="Model ID override (default: workflow model)")
    create_p.add_argument("--output", dest="output", default=None, help="Output directory for results")
    create_p.set_defaults(func=stub)

    # --- batch status ---
    status_p = batch_subs.add_parser("status", help="Check batch job status")
    status_p.add_argument("job_id", help="Batch job ID")
    status_p.set_defaults(func=stub)

    # --- batch list ---
    list_p = batch_subs.add_parser("list", help="List batch jobs")
    list_p.add_argument(
        "--status", dest="status", default=None, help="Filter by status (pending, running, completed, failed)"
    )
    list_p.set_defaults(func=stub)

    # --- batch results ---
    results_p = batch_subs.add_parser("results", help="Get batch job results")
    results_p.add_argument("job_id", help="Batch job ID")
    results_p.add_argument("--format", dest="format", default=None, help="Output format (json, csv)")
    results_p.set_defaults(func=stub)

    # Default
    batch_parser.set_defaults(func=lambda args: batch_parser.print_help())
