"""Batch processing commands."""

from __future__ import annotations

from typing import Annotated, Optional

import typer

from roboflow.cli._compat import ctx_to_args

batch_app = typer.Typer(help="Batch processing operations", no_args_is_help=True)


def _stub(args) -> None:  # noqa: ANN001
    from roboflow.cli._output import output_error

    output_error(args, "This command is not yet implemented.", hint="Coming soon.", exit_code=1)


@batch_app.command("create")
def create(
    ctx: typer.Context,
    workflow: Annotated[str, typer.Option(help="Workflow ID to run")],
    input: Annotated[str, typer.Option(help="Input path (image directory or video file)")],
    model: Annotated[Optional[str], typer.Option(help="Model ID override (default: workflow model)")] = None,
    output_dir: Annotated[Optional[str], typer.Option("--output", help="Output directory for results")] = None,
) -> None:
    """Create a batch processing job."""
    args = ctx_to_args(ctx, workflow=workflow, input=input, model=model, output=output_dir)
    _stub(args)


@batch_app.command("status")
def status(
    ctx: typer.Context,
    job_id: Annotated[str, typer.Argument(help="Batch job ID")],
) -> None:
    """Check batch job status."""
    args = ctx_to_args(ctx, job_id=job_id)
    _stub(args)


@batch_app.command("list")
def list_jobs(
    ctx: typer.Context,
    status_filter: Annotated[
        Optional[str], typer.Option("--status", help="Filter by status (pending, running, completed, failed)")
    ] = None,
) -> None:
    """List batch jobs."""
    args = ctx_to_args(ctx, status=status_filter)
    _stub(args)


@batch_app.command("results")
def results(
    ctx: typer.Context,
    job_id: Annotated[str, typer.Argument(help="Batch job ID")],
    format: Annotated[Optional[str], typer.Option(help="Output format (json, csv)")] = None,
) -> None:
    """Get batch job results."""
    args = ctx_to_args(ctx, job_id=job_id, format=format)
    _stub(args)
