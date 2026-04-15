"""Universal resource shorthand resolver.

Parses compact resource identifiers into (workspace, project, version)
tuples, filling in the default workspace from configuration when omitted.

Disambiguation rule: version numbers are always numeric.  So ``x/y`` where
``y`` is numeric means project/version; where ``y`` is non-numeric means
workspace/project.

Examples
--------
- ``"my-project"``            → (default_ws, "my-project", None)
- ``"my-ws/my-project"``      → ("my-ws", "my-project", None)
- ``"my-project/3"``          → (default_ws, "my-project", 3)
- ``"my-ws/my-project/3"``    → ("my-ws", "my-project", 3)
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

from roboflow.config import get_conditional_configuration_variable


def resolve_default_workspace(api_key: Optional[str] = None) -> Optional[str]:
    """Return the default workspace URL, querying the API if necessary.

    Checks (in order): ``RF_WORKSPACE`` in config/env, then the API
    validation endpoint using the supplied *api_key* (or ``ROBOFLOW_API_KEY``).
    """
    ws = get_conditional_configuration_variable("RF_WORKSPACE", default=None)
    if ws:
        return ws

    key = api_key or os.getenv("ROBOFLOW_API_KEY")
    if not key:
        return None

    import requests

    from roboflow.config import API_URL

    try:
        resp = requests.post(API_URL + "/?api_key=" + key)
        if resp.status_code == 200:
            return resp.json().get("workspace") or None
    except Exception:  # noqa: BLE001
        pass
    return None


def resolve_resource(
    shorthand: str,
    workspace_override: Optional[str] = None,
) -> Tuple[str, str, Optional[int]]:
    """Parse a resource shorthand into (workspace, project, version).

    Parameters
    ----------
    shorthand:
        The compact identifier (see module docstring for formats).
    workspace_override:
        Explicit workspace from ``--workspace`` / ``-w``.  Takes precedence
        over the shorthand's workspace segment when the shorthand is
        ambiguous (single segment).

    Returns
    -------
    tuple[str, str, int | None]
        ``(workspace_url, project_slug, version_number_or_none)``

    Raises
    ------
    ValueError
        If the shorthand cannot be parsed or no workspace can be resolved.
    """
    parts = shorthand.strip("/").split("/")

    default_ws = workspace_override or resolve_default_workspace()

    if len(parts) == 1:
        # "my-project"
        if not default_ws:
            raise ValueError(
                f"Cannot resolve '{shorthand}': no workspace specified and no default configured. "
                "Use --workspace or run 'roboflow auth login'."
            )
        return (default_ws, parts[0], None)

    if len(parts) == 2:
        # Could be "workspace/project" OR "project/version"
        if parts[1].isdigit():
            # "project/3"
            if not default_ws:
                raise ValueError(
                    f"Cannot resolve '{shorthand}': no workspace specified and no default configured. "
                    "Use --workspace or run 'roboflow auth login'."
                )
            return (default_ws, parts[0], int(parts[1]))
        # "workspace/project"
        ws = workspace_override or parts[0]
        return (ws, parts[1], None)

    if len(parts) == 3:
        # "workspace/project/version"
        if not parts[2].isdigit():
            raise ValueError(f"Cannot resolve '{shorthand}': expected numeric version but got '{parts[2]}'.")
        ws = workspace_override or parts[0]
        return (ws, parts[1], int(parts[2]))

    raise ValueError(
        f"Cannot resolve '{shorthand}': expected 1-3 path segments "
        "(project, workspace/project, or workspace/project/version)."
    )


def resolve_ws_and_key(args) -> Optional[Tuple[str, str]]:
    """Resolve workspace and API key from CLI args.

    Returns (workspace_url, api_key) or ``None`` after calling
    ``output_error`` on failure.
    """
    from roboflow.cli._output import output_error
    from roboflow.config import load_roboflow_api_key

    ws = getattr(args, "workspace", None) or resolve_default_workspace(api_key=getattr(args, "api_key", None))
    if not ws:
        output_error(args, "No workspace specified.", hint="Use --workspace or run 'roboflow auth login'.", exit_code=2)
        return None

    api_key = getattr(args, "api_key", None) or load_roboflow_api_key(ws)
    if not api_key:
        output_error(args, "No API key found.", hint="Set ROBOFLOW_API_KEY or run 'roboflow auth login'.", exit_code=2)
        return None

    return ws, api_key
