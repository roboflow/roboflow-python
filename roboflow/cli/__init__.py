# PYTHON_ARGCOMPLETE_OK
"""Roboflow CLI — computer vision at your fingertips.

This package implements the modular CLI for the Roboflow Python SDK.
Commands are auto-discovered from the ``handlers`` sub-package: any module
that exposes a ``register(subparsers)`` callable is loaded automatically.
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
from typing import Any

import roboflow
from roboflow.cli import handlers as _handlers_pkg


class _CleanHelpFormatter(argparse.HelpFormatter):
    """Custom formatter that hides SUPPRESS-ed subparser choices.

    The default argparse formatter includes *all* subparser names in the
    ``{a,b,c,...}`` usage line and shows ``==SUPPRESS==`` in the command
    list.  This formatter filters both so that hidden legacy aliases are
    truly invisible.
    """

    def _format_action(self, action: argparse.Action) -> str:
        # Hide subparser entries whose help is SUPPRESS
        if action.help == argparse.SUPPRESS:
            return ""
        return super()._format_action(action)

    def _metavar_formatter(
        self,
        action: argparse.Action,
        default_metavar: str,
    ) -> Any:
        if isinstance(action, argparse._SubParsersAction):
            # Filter choices to only those with visible help
            visible = [
                name
                for name, parser in action.choices.items()
                if not any(ca.dest == name and ca.help == argparse.SUPPRESS for ca in action._choices_actions)
                and name in [ca.dest for ca in action._choices_actions if ca.help != argparse.SUPPRESS]
            ]
            if visible:

                def _fmt(tuple_size: int) -> tuple[str, ...]:
                    result = "{" + ",".join(visible) + "}"
                    return (result,) * tuple_size if tuple_size > 1 else (result,)

                return _fmt
        return super()._metavar_formatter(action, default_metavar)


def build_parser() -> argparse.ArgumentParser:
    """Build the root argument parser with global flags and auto-discovered handlers."""
    parser = argparse.ArgumentParser(
        prog="roboflow",
        description="Roboflow CLI: computer vision at your fingertips",
        formatter_class=_CleanHelpFormatter,
    )

    # --- global flags ---
    parser.add_argument(
        "--json",
        "-j",
        dest="json",
        action="store_true",
        default=False,
        help="Output results as JSON (stable schema, for agents and piping)",
    )
    parser.add_argument(
        "--api-key",
        "-k",
        dest="api_key",
        default=None,
        help="API key override (default: $ROBOFLOW_API_KEY or config file)",
    )
    parser.add_argument(
        "--workspace",
        "-w",
        dest="workspace",
        default=None,
        help="Workspace URL or ID override (default: configured default)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        dest="quiet",
        action="store_true",
        default=False,
        help="Suppress non-essential output (progress bars, status messages)",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        default=False,
        help="Show package version and exit",
    )

    # --- subcommands ---
    subparsers = parser.add_subparsers(title="commands", dest="command")

    # Auto-discover handler modules (skip private modules starting with _)
    for _importer, modname, _ispkg in pkgutil.iter_modules(_handlers_pkg.__path__):
        if modname.startswith("_"):
            continue
        try:
            mod = importlib.import_module(f"roboflow.cli.handlers.{modname}")
            if hasattr(mod, "register"):
                mod.register(subparsers)
        except Exception as exc:  # noqa: BLE001
            # A broken handler must not take down the entire CLI
            import logging

            logging.getLogger("roboflow.cli").debug("Failed to load handler %s: %s", modname, exc)

    # Load aliases last so they can reference handler functions
    from roboflow.cli.handlers import _aliases

    _aliases.register(subparsers)

    parser.set_defaults(func=None)
    return parser


def _show_version(args: argparse.Namespace) -> None:
    if getattr(args, "json", False):
        import json

        print(json.dumps({"version": roboflow.__version__}))
    else:
        print(roboflow.__version__)


def _reorder_argv(argv: list[str]) -> list[str]:
    """Move known global flags that appear after the subcommand to the front.

    argparse only recognises global flags when they appear *before* the
    subcommand.  Many users (and AI agents) naturally write them at the end,
    e.g. ``roboflow project list --json``.  This helper transparently
    re-orders the argv so those flags are consumed by the root parser.
    """
    global_flags_with_value = {"--api-key", "-k", "--workspace", "-w"}
    global_flags_bool = {"--json", "-j", "--quiet", "-q", "--version"}

    reordered: list[str] = []
    rest: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in global_flags_bool:
            reordered.append(arg)
        elif arg in global_flags_with_value:
            reordered.append(arg)
            if i + 1 < len(argv):
                i += 1
                reordered.append(argv[i])
        else:
            rest.append(arg)
        i += 1
    return reordered + rest


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(_reorder_argv(sys.argv[1:]))

    if args.version:
        _show_version(args)
        sys.exit(0)

    if args.func is not None:
        args.func(args)
    else:
        parser.print_help()
        sys.exit(0)
