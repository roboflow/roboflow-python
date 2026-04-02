"""Structured output helpers for the Roboflow CLI.

Every command should use ``output()`` for its result and ``output_error()``
for failures so that ``--json`` mode works uniformly.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from typing import Any, Iterator, Optional


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


def _parse_error_message(raw: str) -> tuple[Optional[dict[str, Any]], str]:
    """Try to parse a raw error string that may contain embedded JSON.

    Returns ``(parsed_dict_or_None, human_readable_message)``.
    The *parsed_dict* is the deserialized JSON when the string is JSON,
    otherwise ``None``.  The *human_readable_message* drills into nested
    ``error.message`` structures so the text-mode output is clean.
    """
    text = raw.strip()
    # Strip status-code prefix like "404: {...}"
    colon_idx = text.find(": {")
    if 0 < colon_idx < 5:
        text = text[colon_idx + 2 :]
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            err = parsed.get("error", parsed)
            if isinstance(err, dict):
                human = str(err.get("message") or err.get("hint") or err)
            else:
                human = str(err)
            return parsed, human
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return None, raw


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
    parsed, human_message = _parse_error_message(message)

    if getattr(args, "json", False):
        error_value: Any = parsed if parsed is not None else message
        payload: dict[str, Any] = {"error": error_value}
        if hint:
            payload["hint"] = hint
        print(json.dumps(payload), file=sys.stderr)
    else:
        msg = f"Error: {human_message}"
        if hint:
            msg += f"\n  Hint: {hint}"
        print(msg, file=sys.stderr)
    sys.exit(exit_code)


@contextlib.contextmanager
def suppress_sdk_output(args: Any = None) -> Iterator[None]:
    """Suppress SDK stdout noise (e.g. 'loading Roboflow workspace...').

    Always active — the SDK's "loading Roboflow workspace..." messages
    are not useful CLI output in any mode.  The CLI controls its own
    output via ``output()`` and ``output_error()``.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        yield
