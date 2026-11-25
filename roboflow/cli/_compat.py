"""Bridge helpers for the argparse → typer migration.

Provides ``ctx_to_args()`` which converts a :class:`typer.Context` to a
:class:`types.SimpleNamespace` matching the shape that ``output()``,
``output_error()``, and other CLI helpers expect.  This allows existing
handler business logic to remain unchanged during migration.
"""

from __future__ import annotations

import types
from importlib import import_module
from typing import Any

import typer  # noqa: TC002 — needed at runtime for Context type

# Typer switched from external Click to its own implementation. Keep this
# compatibility boundary duck-typed so both runtime and mypy use the active one.
try:
    _click_exceptions = import_module("typer._click.exceptions")
except ModuleNotFoundError:
    _click_exceptions = import_module("click.exceptions")
UsageError = _click_exceptions.UsageError


def _sort_params(params: list[Any]) -> None:
    """Sort params in-place: required first, then alphabetical by option name."""
    params.sort(
        key=lambda p: (
            # --help always last
            "help" in (p.opts if hasattr(p, "opts") else [p.name or ""]),
            # Required options first
            not getattr(p, "required", False),
            # Arguments before options (positionals first)
            not isinstance(p, typer.core.TyperArgument),
            # Alphabetical by the first long option name
            (p.opts[0].lstrip("-") if hasattr(p, "opts") and p.opts else p.name or ""),
        )
    )


class SortedGroup(typer.core.TyperGroup):
    """Click Group that alphabetizes commands and options in --help output.

    Use as ``cls=SortedGroup`` when creating Typer apps so that subcommand
    help pages show options and commands in alphabetical order (with
    required options first).
    """

    def list_commands(self, ctx: Any) -> list[str]:
        return sorted(super().list_commands(ctx))

    def format_help(self, ctx: Any, formatter: Any) -> None:
        """Sort options alphabetically before rendering help."""
        _sort_params(self.params)
        super().format_help(ctx, formatter)

    def get_command(self, ctx: Any, cmd_name: str) -> Any:
        """Wrap returned commands to sort their options too."""
        cmd = super().get_command(ctx, cmd_name)
        if cmd is not None and not isinstance(cmd, SortedGroup):
            # Sort the command's params for its --help output
            _sort_params(cmd.params)
        return cmd


def ctx_to_args(ctx: typer.Context, **kwargs: Any) -> types.SimpleNamespace:
    """Convert a typer Context (with global opts in ``ctx.obj``) to an args namespace.

    Parameters
    ----------
    ctx:
        The typer Context, whose ``.obj`` dict holds the global options
        set by the root callback (``json``, ``api_key``, ``workspace``,
        ``quiet``).
    **kwargs:
        Command-specific parameters to include in the namespace.  These
        override anything in ``ctx.obj``.
    """
    obj = ctx.obj or {}
    return types.SimpleNamespace(
        json=obj.get("json", False),
        api_key=obj.get("api_key"),
        workspace=obj.get("workspace"),
        quiet=obj.get("quiet", False),
        **kwargs,
    )
