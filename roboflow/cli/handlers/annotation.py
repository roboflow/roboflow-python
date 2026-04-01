"""Annotation management commands: batch and job operations (stubs)."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``annotation`` command group."""
    ann_parser = subparsers.add_parser("annotation", help="Annotation management commands")
    ann_sub = ann_parser.add_subparsers(title="annotation commands", dest="annotation_command")

    _add_batch(ann_sub)
    _add_job(ann_sub)

    ann_parser.set_defaults(func=lambda args: ann_parser.print_help())


# ---------------------------------------------------------------------------
# batch
# ---------------------------------------------------------------------------


def _add_batch(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    batch_parser = sub.add_parser("batch", help="Annotation batch commands")
    batch_sub = batch_parser.add_subparsers(title="batch commands", dest="batch_command")

    # batch list
    p = batch_sub.add_parser("list", help="List annotation batches")
    p.add_argument("-p", "--project", required=True, help="Project ID")
    p.set_defaults(func=_stub)

    # batch get
    p = batch_sub.add_parser("get", help="Get annotation batch details")
    p.add_argument("batch_id", help="Batch ID")
    p.add_argument("-p", "--project", required=True, help="Project ID")
    p.set_defaults(func=_stub)

    batch_parser.set_defaults(func=lambda args: batch_parser.print_help())


# ---------------------------------------------------------------------------
# job
# ---------------------------------------------------------------------------


def _add_job(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    job_parser = sub.add_parser("job", help="Annotation job commands")
    job_sub = job_parser.add_subparsers(title="job commands", dest="job_command")

    # job list
    p = job_sub.add_parser("list", help="List annotation jobs")
    p.add_argument("-p", "--project", required=True, help="Project ID")
    p.set_defaults(func=_stub)

    # job get
    p = job_sub.add_parser("get", help="Get annotation job details")
    p.add_argument("job_id", help="Job ID")
    p.add_argument("-p", "--project", required=True, help="Project ID")
    p.set_defaults(func=_stub)

    # job create
    p = job_sub.add_parser("create", help="Create an annotation job")
    p.add_argument("-p", "--project", required=True, help="Project ID")
    p.add_argument("--name", required=True, help="Job name")
    p.add_argument("--batch", default=None, help="Batch ID to assign")
    p.add_argument("--assignees", default=None, help="Comma-separated assignee emails")
    p.set_defaults(func=_stub)

    job_parser.set_defaults(func=lambda args: job_parser.print_help())


# ---------------------------------------------------------------------------
# stub handler
# ---------------------------------------------------------------------------


def _stub(args: argparse.Namespace) -> None:
    """Placeholder for not-yet-implemented annotation commands."""
    if getattr(args, "json", False):
        import json

        print(json.dumps({"error": "not yet implemented"}), file=sys.stderr)
    else:
        print("not yet implemented", file=sys.stderr)
