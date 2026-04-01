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

import roboflow
from roboflow.cli import handlers as _handlers_pkg


def build_parser() -> argparse.ArgumentParser:
    """Build the root argument parser with global flags and auto-discovered handlers."""
    parser = argparse.ArgumentParser(
        prog="roboflow",
        description="Roboflow CLI: computer vision at your fingertips",
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
        mod = importlib.import_module(f"roboflow.cli.handlers.{modname}")
        if hasattr(mod, "register"):
            mod.register(subparsers)

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


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if args.version:
        _show_version(args)
        sys.exit(0)

    if args.func is not None:
        args.func(args)
    else:
        parser.print_help()
        sys.exit(0)
