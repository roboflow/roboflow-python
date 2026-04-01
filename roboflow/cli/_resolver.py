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

from typing import Optional, Tuple

from roboflow.config import get_conditional_configuration_variable


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

    default_ws = workspace_override or get_conditional_configuration_variable("RF_WORKSPACE", default=None)

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
