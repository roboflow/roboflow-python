"""Shell completion commands."""

from __future__ import annotations

import typer

from roboflow.cli._compat import ctx_to_args

completion_app = typer.Typer(help="Generate shell completions", no_args_is_help=True)


def _stub(args) -> None:  # noqa: ANN001
    from roboflow.cli._output import output_error

    output_error(args, "This command is not yet implemented.", hint="Coming soon.", exit_code=1)


@completion_app.command("bash")
def bash(ctx: typer.Context) -> None:
    """Generate bash completions."""
    args = ctx_to_args(ctx)
    _stub(args)


@completion_app.command("zsh")
def zsh(ctx: typer.Context) -> None:
    """Generate zsh completions."""
    args = ctx_to_args(ctx)
    _stub(args)


@completion_app.command("fish")
def fish(ctx: typer.Context) -> None:
    """Generate fish completions."""
    args = ctx_to_args(ctx)
    _stub(args)
