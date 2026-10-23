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
    rotation_due_at: Annotated[
        Optional[str],
        typer.Option(
            "--rotation-due-at",
            help="Rotation due date (ISO date, UTC, in the future). Owners get one reminder email before it.",
        ),
    ] = None,
    owner: Annotated[
        Optional[List[str]],
        typer.Option("--owner", help="Member ID reminded before the due date (repeatable). Omit to default to you."),
    ] = None,
) -> None:
    """Create a new API key.

    The secret key value is printed ONCE — save it immediately.
    To capture it programmatically use --json and pipe to jq:

        roboflow --json api-key create MY-KEY | jq -r .key

    Scope selection is three-way: omit all scope flags to inherit the calling
    key's scopes, pass --scope (repeatable) to scope the key, --no-scopes for a
    key with no abilities, or --full-access for an unscoped/full-access key.

    --rotation-due-at sets the date the key should be rotated by; --owner
    (repeatable) picks who gets the reminder email. Both require Advanced API Keys.
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
        rotation_due_at=rotation_due_at,
        owner=owner,
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
    rotation_due_at: Annotated[
        Optional[str],
        typer.Option("--rotation-due-at", help="New rotation due date (ISO date, UTC, in the future)."),
    ] = None,
    clear_rotation_due_at: Annotated[
        bool,
        typer.Option(
            "--clear-rotation-due-at",
            help="Remove the rotation due date (sends null). Mutually exclusive with --rotation-due-at.",
        ),
    ] = False,
    owner: Annotated[
        Optional[List[str]],
        typer.Option("--owner", help="Member ID reminded before the due date (repeatable). Replaces the owners."),
    ] = None,
    clear_owners: Annotated[
        bool,
        typer.Option(
            "--clear-owners",
            help="Remind everyone who can roll the key (sends []). Mutually exclusive with --owner.",
        ),
    ] = False,
) -> None:
    """Update an API key's display name, scopes, metadata or rotation schedule.

    Scopes are three-way: --scope (repeatable) replaces scopes, --no-scopes
    replaces them with an empty list (no abilities), and --full-access makes the
    key unscoped/full access. Metadata: --metadata sets custom KEY=VALUE pairs,
    --clear-metadata removes all custom metadata. Rotation: --rotation-due-at
    sets the date, --clear-rotation-due-at removes it; --owner (repeatable)
    replaces the reminded members, --clear-owners reminds everyone who can roll
    the key. Rotation fields require Advanced API Keys.
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
        rotation_due_at=rotation_due_at,
        clear_rotation_due_at=clear_rotation_due_at,
        owner=owner,
        clear_owners=clear_owners,
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


@api_key_app.command("roll")
def roll_key(
    ctx: typer.Context,
    key_id: Annotated[str, typer.Argument(help="Key ID (keyId) to roll")],
    rotation_due_at: Annotated[
        Optional[str],
        typer.Option(
            "--rotation-due-at",
            help="Rotation due date for the replacement (ISO date, UTC). Omit to carry the old key's date over.",
        ),
    ] = None,
    owner: Annotated[
        Optional[List[str]],
        typer.Option(
            "--owner", help="Member ID reminded before the due date (repeatable). Omit to keep the old key's owners."
        ),
    ] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation prompt")] = False,
) -> None:
    """Roll an API key: revoke it and create its replacement in one step.

    The replacement keeps the old key's name, scopes, folders, metadata and
    protection. Everything using the old key stops working until it gets the
    new value. The new secret is printed ONCE — save it immediately:

        roboflow --json api-key roll KEY_ID --yes | jq -r .key

    Protected keys cannot be rolled — unprotect them in the dashboard first.
    """
    args = ctx_to_args(ctx, key_id=key_id, rotation_due_at=rotation_due_at, owner=owner, yes=yes)
    _roll_key(args)


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


def _resolve_rotation(args) -> dict:  # noqa: ANN001
    """Resolve the rotation flags into ``update_api_key`` keyword arguments.

    ``--rotation-due-at`` sets the date and ``--clear-rotation-due-at`` sends
    ``CLEAR`` (JSON null); ``--owner`` replaces the owners and ``--clear-owners``
    sends ``[]``. Flags not given are left out (unchanged). Exits 1 when a set and
    a clear flag for the same field are both supplied.
    """
    from roboflow.adapters.rfapi import CLEAR
    from roboflow.cli._output import output_error

    rotation_due_at = getattr(args, "rotation_due_at", None)
    clear_due = getattr(args, "clear_rotation_due_at", False)
    owners = getattr(args, "owner", None)
    clear_owners = getattr(args, "clear_owners", False)

    if rotation_due_at is not None and clear_due:
        output_error(
            args,
            "Only one of --rotation-due-at or --clear-rotation-due-at may be used.",
            hint="Use --rotation-due-at DATE to set the date, or --clear-rotation-due-at to remove it.",
            exit_code=1,
        )
    if owners and clear_owners:
        output_error(
            args,
            "Only one of --owner or --clear-owners may be used.",
            hint="Use --owner MEMBER_ID to pick the reminded members, or --clear-owners to remind everyone.",
            exit_code=1,
        )

    fields: dict = {}
    if clear_due:
        fields["rotation_due_at"] = CLEAR
    elif rotation_due_at is not None:
        fields["rotation_due_at"] = rotation_due_at
    if clear_owners:
        fields["owners"] = []
    elif owners:
        fields["owners"] = list(owners)
    return fields


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
    rotation = _resolve_rotation(args)

    try:
        result = rfapi.create_api_key(
            api_key,
            ws,
            name=args.name,
            scopes=None if scopes is _UNSET else scopes,
            folder_ids=folder_ids,
            custom_metadata=None if metadata is _UNSET else metadata,
            protected=getattr(args, "protected", False),
            **rotation,
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
                hint="Scopes, folders, metadata and the rotation schedule require the Advanced API Keys plan feature.",
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
    fields.update(_resolve_rotation(args))

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


def _roll_key(args) -> None:  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import confirm_destructive, output, output_api_error, output_error

    resolved = _resolve_ws_and_key(args)
    if not resolved:
        return
    ws, api_key = resolved

    if not confirm_destructive(args, f"Roll API key '{args.key_id}'? The current key stops working immediately."):
        return

    try:
        result = rfapi.roll_api_key(
            api_key,
            ws,
            args.key_id,
            rotation_due_at=getattr(args, "rotation_due_at", None),
            owners=getattr(args, "owner", None) or None,
        )
    except rfapi.RoboflowError as exc:
        status = getattr(exc, "status_code", None)
        if status == 409:
            output_error(
                args,
                str(exc),
                hint=(
                    "The key is protected, is already being rolled, or is disabled while backing a "
                    "Dedicated Deployment. Unprotect it at app.roboflow.com/settings/api, or retry once "
                    "the other roll finishes."
                ),
                exit_code=1,
            )
        elif status == 403:
            output_api_error(
                args,
                exc,
                hint=(
                    "Rolling needs the 'api-key:create' and 'api-key:revoke' scopes, and the calling key "
                    "must hold at least the target key's scopes and folders. --rotation-due-at and --owner "
                    "require the Advanced API Keys plan feature."
                ),
            )
        else:
            output_api_error(
                args,
                exc,
                not_found_hint=(
                    f"No active API key with ID '{args.key_id}' in this workspace. "
                    "Rolled and revoked keys are not listed."
                ),
            )
        return

    secret = result.get("key", "")
    new_key_id = result.get("keyId", "")
    name = result.get("name") or args.key_id

    lines = [
        f"Rolled API key '{name}': revoked {args.key_id}, replacement keyId: {new_key_id}",
        "",
        "WARNING: This is the only time the new secret key will be shown.",
        "Save it somewhere secure now and update everything that used the old key.",
        "",
        f"  Key: {secret}",
    ]
    output(args, result, text="\n".join(lines))
