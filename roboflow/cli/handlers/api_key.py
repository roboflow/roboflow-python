"""API key management commands."""

from __future__ import annotations

from typing import Annotated, List, Optional

import typer

from roboflow.cli._compat import SortedGroup, ctx_to_args

api_key_app = typer.Typer(cls=SortedGroup, help="Manage workspace API keys", no_args_is_help=True)


@api_key_app.command("list")
def list_keys(
    ctx: typer.Context,
    include_disabled: Annotated[bool, typer.Option("--include-disabled", help="Include disabled keys")] = False,
    include_folders: Annotated[bool, typer.Option("--include-folders", help="Include folder ID details")] = False,
) -> None:
    """List all API keys for the workspace."""
    args = ctx_to_args(ctx, include_disabled=include_disabled, include_folders=include_folders)
    _list_keys(args)


@api_key_app.command("get")
def get_key(
    ctx: typer.Context,
    key_id: Annotated[str, typer.Argument(help="Key ID (keyId) to retrieve")],
) -> None:
    """Show details for a single API key."""
    args = ctx_to_args(ctx, key_id=key_id)
    _get_key(args)


@api_key_app.command("publishable")
def get_publishable(ctx: typer.Context) -> None:
    """Print the workspace publishable key (rf_<workspaceId>).

    This is the non-secret key used for browser / inference.js requests.
    It is safe to embed in client-side code. To use it programmatically:

        roboflow --json api-key publishable | jq -r .publishableKey
    """
    args = ctx_to_args(ctx)
    _get_publishable(args)


@api_key_app.command("create")
def create_key(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Display name for the new key")],
    scope: Annotated[
        Optional[List[str]],
        typer.Option(
            "--scope",
            help="Scope or role:<name> preset (repeatable). Omit to inherit the calling key's scopes.",
        ),
    ] = None,
    no_scopes: Annotated[
        bool,
        typer.Option(
            "--no-scopes",
            help="Create a key with an empty scope list (no abilities). Mutually exclusive with --scope/--full-access.",
        ),
    ] = False,
    full_access: Annotated[
        bool,
        typer.Option(
            "--full-access",
            help="Create an unscoped, full-access key (sends null). Mutually exclusive with --scope/--no-scopes.",
        ),
    ] = False,
    folder: Annotated[
        Optional[List[str]],
        typer.Option("--folder", help="Folder ID to restrict access to (repeatable)."),
    ] = None,
    metadata: Annotated[
        Optional[List[str]],
        typer.Option("--metadata", help="Custom metadata as KEY=VALUE (repeatable)."),
    ] = None,
    protected: Annotated[
        bool,
        typer.Option("--protected", help="Mark key as protected (cannot be revoked/disabled via CLI)"),
    ] = False,
) -> None:
    """Create a new API key.

    The secret key value is printed ONCE — save it immediately.
    To capture it programmatically use --json and pipe to jq:

        roboflow --json api-key create MY-KEY | jq -r .key

    Scope selection is three-way: omit all scope flags to inherit the calling
    key's scopes, pass --scope (repeatable) to scope the key, --no-scopes for a
    key with no abilities, or --full-access for an unscoped/full-access key.
    """
    args = ctx_to_args(
        ctx,
        name=name,
        scope=scope,
        no_scopes=no_scopes,
        full_access=full_access,
        folder=folder,
        metadata=metadata,
        protected=protected,
    )
    _create_key(args)


@api_key_app.command("update")
def update_key(
    ctx: typer.Context,
    key_id: Annotated[str, typer.Argument(help="Key ID (keyId) to update")],
    name: Annotated[Optional[str], typer.Option("--name", help="New display name")] = None,
    scope: Annotated[
        Optional[List[str]],
        typer.Option(
            "--scope",
            help="Scope or role:<name> preset (repeatable). Replaces existing scopes.",
        ),
    ] = None,
    no_scopes: Annotated[
        bool,
        typer.Option(
            "--no-scopes",
            help="Replace scopes with an empty list (no abilities). Mutually exclusive with --scope/--full-access.",
        ),
    ] = False,
    full_access: Annotated[
        bool,
        typer.Option(
            "--full-access",
            help="Make the key unscoped/full access (sends null). Mutually exclusive with --scope/--no-scopes.",
        ),
    ] = False,
    metadata: Annotated[
        Optional[List[str]],
        typer.Option("--metadata", help="Custom metadata as KEY=VALUE (repeatable)."),
    ] = None,
    clear_metadata: Annotated[
        bool,
        typer.Option(
            "--clear-metadata",
            help="Clear all custom metadata (sends {}). Mutually exclusive with --metadata.",
        ),
    ] = False,
) -> None:
    """Update an API key's display name, scopes, or metadata.

    Scopes are three-way: --scope (repeatable) replaces scopes, --no-scopes
    replaces them with an empty list (no abilities), and --full-access makes the
    key unscoped/full access. Metadata: --metadata sets custom KEY=VALUE pairs,
    --clear-metadata removes all custom metadata.
    """
    args = ctx_to_args(
        ctx,
        key_id=key_id,
        name=name,
        scope=scope,
        no_scopes=no_scopes,
        full_access=full_access,
        metadata=metadata,
        clear_metadata=clear_metadata,
    )
    _update_key(args)


@api_key_app.command("protect")
def protect_key(
    ctx: typer.Context,
    key_id: Annotated[str, typer.Argument(help="Key ID (keyId) to protect")],
) -> None:
    """Mark an API key as protected.

    Protected keys cannot be revoked or disabled via the CLI or API.
    To unprotect a key, visit app.roboflow.com/settings/api.
    """
    args = ctx_to_args(ctx, key_id=key_id)
    _protect_key(args)


@api_key_app.command("disable")
def disable_key(
    ctx: typer.Context,
    key_id: Annotated[str, typer.Argument(help="Key ID (keyId) to enable/disable")],
    enable: Annotated[
        bool, typer.Option("--enable/--disable", help="Enable (--enable) or disable (--disable) the key")
    ] = False,
) -> None:
    """Enable or disable an API key (default: --disable).

    Disabled keys are rejected by the API but can be re-enabled.
    Protected keys cannot be disabled — revoke them from the dashboard.
    """
    args = ctx_to_args(ctx, key_id=key_id, enable=enable)
    _disable_key(args)


@api_key_app.command("revoke")
def revoke_key(
    ctx: typer.Context,
    key_id: Annotated[str, typer.Argument(help="Key ID (keyId) to revoke")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation prompt")] = False,
) -> None:
    """Permanently revoke an API key. This action cannot be undone.

    Protected keys cannot be revoked via the CLI — use the dashboard.
    """
    args = ctx_to_args(ctx, key_id=key_id, yes=yes)
    _revoke_key(args)


# ---------------------------------------------------------------------------
# Business logic
# ---------------------------------------------------------------------------


def _resolve_ws_and_key(args):  # noqa: ANN001
    from roboflow.cli._resolver import resolve_ws_and_key

    return resolve_ws_and_key(args)


def _parse_metadata(args, pairs: Optional[List[str]]) -> Optional[dict]:  # noqa: ANN001
    """Parse repeated ``KEY=VALUE`` strings into a dict, or ``None`` if empty."""
    if not pairs:
        return None
    from roboflow.cli._output import output_error

    metadata: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            output_error(
                args,
                f"Invalid --metadata value '{pair}'. Expected KEY=VALUE.",
                hint="Example: --metadata team=vision --metadata env=prod",
                exit_code=1,
            )
        metadata[key] = value
    return metadata


# Sentinel meaning "field not provided" — distinct from an explicit ``None``/``[]``/``{}``.
_UNSET = object()


def _resolve_scopes(args):  # noqa: ANN001
    """Resolve the three-way ``--scope`` / ``--no-scopes`` / ``--full-access`` selection.

    Returns one of:
      * ``_UNSET`` — no scope flag given (inherit / leave unchanged),
      * a list — explicit ``--scope`` values (``[]`` for ``--no-scopes``),
      * ``rfapi.FULL_ACCESS`` — ``--full-access`` (send ``null``).

    Exits 1 if more than one of the three is supplied.
    """
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output_error

    scope = getattr(args, "scope", None) or None
    no_scopes = getattr(args, "no_scopes", False)
    full_access = getattr(args, "full_access", False)

    if sum([scope is not None, no_scopes, full_access]) > 1:
        output_error(
            args,
            "Only one of --scope, --no-scopes, or --full-access may be used.",
            hint="Use --scope to scope the key, --no-scopes for no abilities, or --full-access for unscoped access.",
            exit_code=1,
        )

    if full_access:
        return rfapi.FULL_ACCESS
    if no_scopes:
        return []
    if scope is not None:
        return scope
    return _UNSET


def _resolve_metadata(args):  # noqa: ANN001
    """Resolve the ``--metadata`` / ``--clear-metadata`` selection.

    Returns one of:
      * ``_UNSET`` — neither flag given (leave unchanged),
      * a dict — parsed ``--metadata`` pairs (``{}`` for ``--clear-metadata``).

    Exits 1 if both are supplied.
    """
    from roboflow.cli._output import output_error

    metadata = getattr(args, "metadata", None)
    clear_metadata = getattr(args, "clear_metadata", False)

    if metadata and clear_metadata:
        output_error(
            args,
            "Only one of --metadata or --clear-metadata may be used.",
            hint="Use --metadata KEY=VALUE to set metadata, or --clear-metadata to remove it.",
            exit_code=1,
        )

    if clear_metadata:
        return {}
    parsed = _parse_metadata(args, metadata)
    if parsed is not None:
        return parsed
    return _UNSET


def _list_keys(args) -> None:  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_api_error
    from roboflow.cli._table import format_table

    resolved = _resolve_ws_and_key(args)
    if not resolved:
        return
    ws, api_key = resolved

    try:
        result = rfapi.list_api_keys(
            api_key,
            ws,
            include_disabled=getattr(args, "include_disabled", False),
            include_folders=getattr(args, "include_folders", False),
        )
    except rfapi.RoboflowError as exc:
        output_api_error(args, exc)
        return

    keys = result.get("apiKeys", [])
    rows = []
    for k in keys:
        rows.append(
            {
                "keyId": k.get("keyId", ""),
                "name": k.get("name", ""),
                "prefix": k.get("prefix", ""),
                "default": str(k.get("default", False)),
                "protected": str(k.get("protected", False)),
                "disabled": str(k.get("disabled", False)),
            }
        )
    table = format_table(
        rows,
        columns=["keyId", "name", "prefix", "default", "protected", "disabled"],
        headers=["KEY ID", "NAME", "PREFIX", "DEFAULT", "PROTECTED", "DISABLED"],
    )
    output(args, result, text=table)


def _get_key(args) -> None:  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_api_error

    resolved = _resolve_ws_and_key(args)
    if not resolved:
        return
    ws, api_key = resolved

    try:
        result = rfapi.get_api_key(api_key, ws, args.key_id)
    except rfapi.RoboflowError as exc:
        output_api_error(args, exc, not_found_hint=f"No API key with ID '{args.key_id}' in this workspace.")
        return

    key_obj = result.get("apiKey", result)
    lines = [
        f"Key ID: {key_obj.get('keyId', '')}",
        f"  Name: {key_obj.get('name', '')}",
        f"  Prefix: {key_obj.get('prefix', '')}",
        f"  Default: {key_obj.get('default', False)}",
        f"  Protected: {key_obj.get('protected', False)}",
        f"  Disabled: {key_obj.get('disabled', False)}",
    ]
    if key_obj.get("scopes"):
        lines.append(f"  Scopes: {', '.join(key_obj['scopes'])}")
    if key_obj.get("created_on"):
        lines.append(f"  Created: {key_obj['created_on']}")
    output(args, result, text="\n".join(lines))


def _get_publishable(args) -> None:  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_api_error

    resolved = _resolve_ws_and_key(args)
    if not resolved:
        return
    ws, api_key = resolved

    try:
        result = rfapi.get_publishable_key(api_key, ws)
    except rfapi.RoboflowError as exc:
        output_api_error(args, exc)
        return

    pub_key = result.get("publishableKey", "")
    output(args, result, text=pub_key)


def _create_key(args) -> None:  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_api_error

    resolved = _resolve_ws_and_key(args)
    if not resolved:
        return
    ws, api_key = resolved

    scopes = _resolve_scopes(args)
    folder_ids = getattr(args, "folder", None) or None
    metadata = _resolve_metadata(args)

    try:
        result = rfapi.create_api_key(
            api_key,
            ws,
            name=args.name,
            scopes=None if scopes is _UNSET else scopes,
            folder_ids=folder_ids,
            custom_metadata=None if metadata is _UNSET else metadata,
            protected=getattr(args, "protected", False),
        )
    except rfapi.RoboflowError as exc:
        status = getattr(exc, "status_code", None)
        if status in (403, 404):
            output_api_error(
                args,
                exc,
                hint=(
                    "Creating API keys requires an unscoped key or one granted the "
                    "'api-key:create' scope (OAuth also needs the create_api_key permission). "
                    "A workspace-wide key additionally requires access to all folders. Use an "
                    "unscoped key, grant this key 'api-key:create', or create the key at "
                    "app.roboflow.com/settings/api."
                ),
            )
        else:
            output_api_error(
                args,
                exc,
                hint="Scopes, folders, and metadata require the Advanced API Keys plan feature.",
            )
        return

    secret = result.get("key", "")
    key_id = result.get("keyId", "")

    lines = [
        f"Created API key '{args.name}' (keyId: {key_id})",
        "",
        "WARNING: This is the only time the secret key will be shown.",
        "Save it somewhere secure now.",
        "",
        f"  Key: {secret}",
    ]
    output(args, result, text="\n".join(lines))


def _update_key(args) -> None:  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_api_error

    resolved = _resolve_ws_and_key(args)
    if not resolved:
        return
    ws, api_key = resolved

    fields = {}
    if getattr(args, "name", None) is not None:
        fields["name"] = args.name
    scopes = _resolve_scopes(args)
    if scopes is not _UNSET:
        fields["scopes"] = scopes
    metadata = _resolve_metadata(args)
    if metadata is not _UNSET:
        fields["custom_metadata"] = metadata

    try:
        result = rfapi.update_api_key(api_key, ws, args.key_id, **fields)
    except rfapi.RoboflowError as exc:
        output_api_error(
            args,
            exc,
            not_found_hint=f"No API key with ID '{args.key_id}' in this workspace.",
        )
        return

    output(args, result, text=f"Updated API key '{args.key_id}'")


def _protect_key(args) -> None:  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_api_error

    resolved = _resolve_ws_and_key(args)
    if not resolved:
        return
    ws, api_key = resolved

    try:
        result = rfapi.update_api_key(api_key, ws, args.key_id, protected=True)
    except rfapi.RoboflowError as exc:
        output_api_error(
            args,
            exc,
            hint="The API cannot unprotect a key. To unprotect, visit app.roboflow.com/settings/api.",
            not_found_hint=f"No API key with ID '{args.key_id}' in this workspace.",
        )
        return

    output(args, result, text=f"Marked API key '{args.key_id}' as protected.")


def _disable_key(args) -> None:  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_api_error, output_error

    resolved = _resolve_ws_and_key(args)
    if not resolved:
        return
    ws, api_key = resolved

    enable = getattr(args, "enable", False)
    disabled_value = not enable

    try:
        result = rfapi.update_api_key(api_key, ws, args.key_id, disabled=disabled_value)
    except rfapi.RoboflowError as exc:
        if getattr(exc, "status_code", None) == 409:
            output_error(
                args,
                str(exc),
                hint="Protected keys cannot be disabled. To disable, visit app.roboflow.com/settings/api.",
                exit_code=1,
            )
        elif getattr(exc, "status_code", None) == 403:
            output_error(
                args,
                str(exc),
                hint="Enabling/disabling keys requires the Advanced API Keys plan feature.",
                exit_code=1,
            )
        else:
            output_api_error(
                args,
                exc,
                not_found_hint=f"No API key with ID '{args.key_id}' in this workspace.",
            )
        return

    action = "Enabled" if enable else "Disabled"
    output(args, result, text=f"{action} API key '{args.key_id}'.")


def _revoke_key(args) -> None:  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import confirm_destructive, output, output_api_error, output_error

    resolved = _resolve_ws_and_key(args)
    if not resolved:
        return
    ws, api_key = resolved

    if not confirm_destructive(args, f"Permanently revoke API key '{args.key_id}'?"):
        return

    try:
        result = rfapi.revoke_api_key(api_key, ws, args.key_id)
    except rfapi.RoboflowError as exc:
        if getattr(exc, "status_code", None) == 409:
            output_error(
                args,
                str(exc),
                hint=("Protected keys cannot be revoked via the CLI. To revoke, visit app.roboflow.com/settings/api."),
                exit_code=1,
            )
        else:
            output_api_error(
                args,
                exc,
                not_found_hint=f"No API key with ID '{args.key_id}' in this workspace.",
            )
        return

    output(args, result, text=f"Revoked API key '{args.key_id}'.")
