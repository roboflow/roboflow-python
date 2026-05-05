"""Shell completion: install + raw script generators.

Delegates installation to ``typer.completion.install`` (which itself
wraps Click's ``shell_completion`` and auto-detects the shell via
shellingham). Hidden commands are filtered by Click automatically.
"""

from __future__ import annotations

import shutil
from typing import Annotated, Optional

import click
import typer
from typer._completion_classes import completion_init
from typer._completion_shared import get_completion_script
from typer.completion import install as typer_install

from roboflow.cli._compat import SortedGroup, ctx_to_args
from roboflow.cli._output import output, output_error

completion_app = typer.Typer(
    cls=SortedGroup,
    help="Generate and install shell completions",
    no_args_is_help=True,
)

completion_init()


def _generate_completion(shell: str) -> str:
    return get_completion_script(prog_name="roboflow", complete_var="_ROBOFLOW_COMPLETE", shell=shell)


@completion_app.command("bash")
def bash() -> None:
    """Print bash completion script. Usage: eval "$(roboflow completion bash)"."""
    print(_generate_completion("bash"))  # noqa: T201


@completion_app.command("zsh")
def zsh() -> None:
    """Print zsh completion script. Usage: eval "$(roboflow completion zsh)"."""
    print(_generate_completion("zsh"))  # noqa: T201


@completion_app.command("fish")
def fish() -> None:
    """Print fish completion script. Usage: roboflow completion fish | source."""
    print(_generate_completion("fish"))  # noqa: T201


@completion_app.command("install")
def install(
    ctx: typer.Context,
    shell: Annotated[
        Optional[str],
        typer.Option("--shell", help="bash, zsh, or fish. Auto-detected when omitted."),
    ] = None,
) -> None:
    """Install shell completion. Writes the script and updates your shell rc. Idempotent."""
    args = ctx_to_args(ctx, shell=shell)

    if shutil.which("roboflow") is None:
        output_error(
            args,
            "The 'roboflow' command is not on your PATH.",
            hint="Ensure your install bin directory (e.g. ~/.local/bin) is on PATH.",
            exit_code=1,
        )
        return

    try:
        installed_shell, path = typer_install(shell=shell, prog_name="roboflow", complete_var="_ROBOFLOW_COMPLETE")
    except click.exceptions.Exit:
        output_error(
            args,
            "Could not detect or install completion.",
            hint="Pass --shell with one of: bash, zsh, fish.",
            exit_code=3,
        )
        return

    output(
        args,
        {"shell": installed_shell, "path": str(path)},
        text=f"Installed {installed_shell} completion to {path}.\nOpen a new shell to enable it.",
    )
