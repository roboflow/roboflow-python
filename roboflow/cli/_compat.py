"""Bridge helpers for the argparse → typer migration.

Provides ``ctx_to_args()`` which converts a :class:`typer.Context` to a
:class:`types.SimpleNamespace` matching the shape that ``output()``,
``output_error()``, and other CLI helpers expect.  This allows existing
handler business logic to remain unchanged during migration.
"""

from __future__ import annotations

import types
from typing import Any

import typer


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
