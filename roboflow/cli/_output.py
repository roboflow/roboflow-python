"""Structured output helpers for the Roboflow CLI.

Every command should use ``output()`` for its result and ``output_error()``
for failures so that ``--json`` mode works uniformly.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Optional


def output(args: Any, data: Any, text: Optional[str] = None) -> None:
    """Print a command result in JSON or human-readable format.

    Parameters
    ----------
    args:
        The parsed argparse namespace (must have a ``json`` attribute).
    data:
        Structured data to emit when ``--json`` is active.  Also used as
        fallback when *text* is ``None``.
    text:
        Human-readable string printed in normal (non-JSON) mode.  When
        ``None``, *data* is pretty-printed as JSON regardless of mode.
    """
    if getattr(args, "json", False):
        print(json.dumps(data, indent=2, default=str))
    elif text is not None:
        print(text)
    else:
        # Fallback: pretty-print data even in non-JSON mode
        print(json.dumps(data, indent=2, default=str))


def output_error(
    args: Any,
    message: str,
    hint: Optional[str] = None,
    exit_code: int = 1,
) -> None:
    """Print an error and exit.

    Parameters
    ----------
    args:
        The parsed argparse namespace.
    message:
        What went wrong.
    hint:
        Actionable suggestion for the user / agent.
    exit_code:
        Process exit code.  Convention: 1 = general, 2 = auth, 3 = not found.
    """
    if getattr(args, "json", False):
        payload: dict[str, Any] = {"error": message}
        if hint:
            payload["hint"] = hint
        print(json.dumps(payload), file=sys.stderr)
    else:
        msg = f"Error: {message}"
        if hint:
            msg += f"\n  Hint: {hint}"
        print(msg, file=sys.stderr)
    sys.exit(exit_code)
