"""Top-level backwards-compatibility aliases.

Registers convenience commands at the root level (``roboflow login``,
``roboflow upload``, etc.) that delegate to the canonical noun-verb handlers.

This module is loaded *after* all other handlers by ``build_parser()`` so
that it can import their handler functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register top-level aliases for common commands."""
    # Aliases will be wired up in Wave 2 after all handlers are created.
    # For now this is a no-op skeleton that registers nothing so the
    # auto-discovery and import chain works end-to-end.
    pass
