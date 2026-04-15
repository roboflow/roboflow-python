"""Shell completion commands.

Generates completion scripts for bash, zsh, and fish shells.
Uses Click's built-in completion generation via the ``_ROBOFLOW_COMPLETE``
environment variable.
"""

from __future__ import annotations

import sys

import click
import typer

from roboflow.cli._compat import SortedGroup

completion_app = typer.Typer(cls=SortedGroup, help="Generate shell completions", no_args_is_help=True)


def _generate_completion(shell: str) -> None:
    """Generate completion script for the given shell using Click's completion system."""
    from click.shell_completion import get_completion_class

    comp_cls = get_completion_class(shell)
    if comp_cls is None:
        print(f"Shell '{shell}' is not supported for completion.", file=sys.stderr)
        raise typer.Exit(code=1)

    from roboflow.cli import app

    # Access the underlying Click command
    click_app = typer.main.get_command(app)
    ctx = click.Context(click_app, info_name="roboflow")
    comp = comp_cls(click_app, ctx, "roboflow", "_ROBOFLOW_COMPLETE")  # type: ignore[arg-type]
    print(comp.source())  # noqa: T201


@completion_app.command("bash")
def bash() -> None:
    """Generate bash completion script.

    Usage: eval "$(roboflow completion bash)"
    Or save to a file: roboflow completion bash > ~/.roboflow-complete.bash
    """
    _generate_completion("bash")


@completion_app.command("zsh")
def zsh() -> None:
    """Generate zsh completion script.

    Usage: eval "$(roboflow completion zsh)"
    Or save to a file: roboflow completion zsh > ~/.roboflow-complete.zsh
    """
    _generate_completion("zsh")


@completion_app.command("fish")
def fish() -> None:
    """Generate fish completion script.

    Usage: roboflow completion fish | source
    Or save to a file: roboflow completion fish > ~/.config/fish/completions/roboflow.fish
    """
    _generate_completion("fish")
