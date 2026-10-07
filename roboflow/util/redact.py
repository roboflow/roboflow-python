"""Keep API keys out of messages that are printed or end up in logs."""

import re
from typing import Any

# Match api_key=... up to the next whitespace, query separator, quote, or backslash.
_API_KEY_PARAM = re.compile(r"api_key=[^\s&\"'\\<>]+")


def redact_api_key(text: Any) -> Any:
    """Replace the value of every ``api_key=`` URL parameter in *text* with ``***``.

    Most SDK requests still pass the API key as a query parameter, and the
    messages of ``requests`` exceptions contain the full request URL.
    Values other than strings are returned unchanged.
    """
    if not isinstance(text, str):
        return text
    return _API_KEY_PARAM.sub("api_key=***", text)
