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


_PLAN_HINT_PATTERNS: list[tuple[str, str]] = [
    ("require", "This feature requires a higher plan. Visit https://roboflow.com/pricing to upgrade."),
    ("Growth plan", "This feature requires a Growth plan or higher. Visit https://roboflow.com/pricing to upgrade."),
    ("Enterprise", "This feature requires an Enterprise plan. Contact sales@roboflow.com to upgrade."),
    ("folder billing", "This feature requires folder billing. Visit https://app.roboflow.com/settings to enable it."),
    ("Unauthorized", "Check your API key and workspace permissions. Some features require specific plan tiers."),
    ("over_quota", "Your workspace has exceeded its quota. Visit https://roboflow.com/pricing to upgrade."),
]

# Patterns to translate raw API hints into CLI-friendly hints
_API_HINT_REPLACEMENTS: list[tuple[str, str]] = [
    (
        "You can see your active workspace by issuing a GET request to / with your api_key",
        "Check available resources with 'roboflow project list' or 'roboflow workspace get'.",
    ),
    (
        "You can find the API docs at https://docs.roboflow.com",
        "Run the command with --help for usage information.",
    ),
    (
        "You can see your available workspaces by issuing a GET request to /workspaces",
        "List workspaces with 'roboflow workspace list'.",
    ),
]


def _detect_plan_hint(message: str) -> Optional[str]:
    """Detect plan/billing-related errors and return an appropriate upgrade hint."""
    lower = message.lower()
    for pattern, hint in _PLAN_HINT_PATTERNS:
        if pattern.lower() in lower:
            return hint
    return None


def _translate_api_hints(message: str) -> str:
    """Replace raw API hints with CLI-friendly equivalents."""
    for api_hint, cli_hint in _API_HINT_REPLACEMENTS:
        message = message.replace(api_hint, cli_hint)
    # Generic fallback: strip any remaining "issuing a GET/POST request" phrasing
    import re

    message = re.sub(
        r"You can [^.]*(?:GET|POST|PUT|DELETE) request[^.]*\.",
        "Run the command with --help for usage information.",
        message,
    )
    return message


def _sanitize_credentials(text: str) -> str:
    """Strip API keys from URLs and other sensitive patterns in error messages."""
    import re

    return re.sub(r"api_key=[A-Za-z0-9_]+", "api_key=***", text)


def _parse_error_message(raw: str) -> tuple[Optional[dict[str, Any]], str]:
    """Try to parse a raw error string that may contain embedded JSON.

    Returns ``(parsed_dict_or_None, human_readable_message)``.
    The *parsed_dict* is the deserialized JSON when the string is JSON,
    otherwise ``None``.  The *human_readable_message* drills into nested
    ``error.message`` structures so the text-mode output is clean.
    """
    text = _translate_api_hints(_sanitize_credentials(raw.strip()))
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
                # Translate API hints in the parsed dict too
                if "hint" in err and isinstance(err["hint"], str):
                    err["hint"] = _translate_api_hints(err["hint"])
            else:
                human = str(err)
            return parsed, _translate_api_hints(human)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return None, text  # Return sanitized text, not the original raw


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

    # Auto-detect plan-gated errors and add upgrade hints when none provided
    if not hint:
        hint = _detect_plan_hint(human_message)

    if getattr(args, "json", False):
        # Normalise error to always be {"error": {"message": "..."}} so
        # consumers see a consistent schema regardless of error source.
        if parsed is not None and "error" in parsed:
            inner: Any = parsed["error"]
        elif parsed is not None:
            inner = parsed
        else:
            inner = None

        if isinstance(inner, dict):
            error_obj: dict[str, Any] = dict(inner)
            error_obj.setdefault("message", human_message)
        else:
            error_obj = {"message": human_message}

        if hint:
            error_obj.setdefault("hint", hint)
        payload: dict[str, Any] = {"error": error_obj}
        print(json.dumps(payload), file=sys.stderr)
    else:
        msg = f"Error: {human_message}"
        if hint:
            msg += f"\n  Hint: {hint}"
        print(msg, file=sys.stderr)
    sys.exit(exit_code)


def stub(args: Any) -> None:
    """Placeholder handler for not-yet-implemented commands."""
    output_error(args, "This command is not yet implemented.", hint="Coming soon.", exit_code=1)


@contextlib.contextmanager
def suppress_sdk_output(args: Any = None) -> Iterator[None]:
    """Suppress SDK stdout noise (e.g. 'loading Roboflow workspace...').

    Always active — the SDK's "loading Roboflow workspace..." messages
    are not useful CLI output in any mode.  The CLI controls its own
    output via ``output()`` and ``output_error()``.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        yield
