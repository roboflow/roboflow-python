"""Shell completion commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse


def _stub(args: argparse.Namespace) -> None:
    from roboflow.cli._output import output_error

    output_error(args, "This command is not yet implemented.", hint="Coming soon.", exit_code=1)


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``completion`` command group."""
    comp_parser = subparsers.add_parser("completion", help="Generate shell completions")
    comp_subs = comp_parser.add_subparsers(title="completion commands", dest="completion_command")

    # --- completion bash ---
    bash_p = comp_subs.add_parser("bash", help="Generate bash completions")
    bash_p.set_defaults(func=_stub)

    # --- completion zsh ---
    zsh_p = comp_subs.add_parser("zsh", help="Generate zsh completions")
    zsh_p.set_defaults(func=_stub)

    # --- completion fish ---
    fish_p = comp_subs.add_parser("fish", help="Generate fish completions")
    fish_p.set_defaults(func=_stub)

    # Default
    comp_parser.set_defaults(func=lambda args: comp_parser.print_help())
