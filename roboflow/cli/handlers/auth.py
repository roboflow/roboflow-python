"""Auth commands: login, logout, status, set-region, set-workspace."""

from __future__ import annotations

from typing import Annotated, Optional

import typer

from roboflow.cli._compat import SortedGroup, ctx_to_args

auth_app = typer.Typer(cls=SortedGroup, help="Manage authentication and credentials", no_args_is_help=True)


@auth_app.command("login")
def login(
    ctx: typer.Context,
    login_api_key: Annotated[Optional[str], typer.Option("--api-key", help="API key (skip interactive prompt)")] = None,
    login_workspace: Annotated[
        Optional[str], typer.Option("--workspace", help="Set default workspace during login")
    ] = None,
    region: Annotated[
        Optional[str], typer.Option("--region", metavar="{us,eu}", help="Roboflow platform region")
    ] = None,
    force: Annotated[bool, typer.Option("--force", "-f", help="Force re-login even if already logged in")] = False,
) -> None:
    """Log in to Roboflow."""
    args = ctx_to_args(
        ctx,
        login_api_key=login_api_key,
        login_workspace=login_workspace,
        region=region,
        force=force,
    )
    _login(args)


@auth_app.command("status")
def status(ctx: typer.Context) -> None:
    """Show current auth status."""
    args = ctx_to_args(ctx)
    _status(args)


@auth_app.command("set-workspace")
def set_workspace(
    ctx: typer.Context,
    workspace_id: Annotated[str, typer.Argument(help="Workspace URL or ID to set as default")],
) -> None:
    """Set the default workspace."""
    args = ctx_to_args(ctx, workspace_id=workspace_id)
    _set_workspace(args)


@auth_app.command("set-region")
def set_region(
    ctx: typer.Context,
    region: Annotated[str, typer.Argument(metavar="{us,eu}", help="Roboflow platform region")],
) -> None:
    """Set the Roboflow platform region."""
    args = ctx_to_args(ctx, region=region)
    _set_region(args)


@auth_app.command("logout")
def logout(ctx: typer.Context) -> None:
    """Remove stored credentials."""
    args = ctx_to_args(ctx)
    _logout(args)


# ---------------------------------------------------------------------------
# Business logic (unchanged from argparse version)
# ---------------------------------------------------------------------------


def _get_config_path() -> str:
    import os
    from pathlib import Path

    if os.name == "nt":
        default_path = str(Path.home() / "roboflow" / "config.json")
    else:
        default_path = str(Path.home() / ".config" / "roboflow" / "config.json")
    return os.getenv("ROBOFLOW_CONFIG_DIR", default=default_path)


def _load_config() -> dict:
    import json
    import os

    path = _get_config_path()
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_config(config: dict) -> None:
    import json
    import os
    import stat

    path = _get_config_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Write with owner-only permissions (0600) since the file contains API keys
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
    with os.fdopen(fd, "w") as f:
        json.dump(config, f, indent=2)


def _mask_key(key: str) -> str:
    if not key or len(key) <= 4:
        return "****"
    return key[:2] + "*" * (len(key) - 4) + key[-2:]


def _validate_region(args, region: Optional[str]) -> Optional[str]:  # noqa: ANN001
    """Normalize and validate a region supplied explicitly on the CLI."""
    if region is None:
        return None

    normalized = region.lower()
    if normalized not in {"us", "eu"}:
        from roboflow.cli._output import output_error

        output_error(
            args,
            f"Invalid region '{region}'.",
            hint="Region must be 'us' or 'eu'.",
            exit_code=2,
        )
    return normalized


def _region_status() -> tuple[dict[str, str], list[str]]:
    """Return the effective region metadata in structured and text forms."""
    from roboflow.config import get_effective_region, resolve_url

    region = get_effective_region()
    api_url = resolve_url("API_URL", region=region)
    app_url = resolve_url("APP_URL", region=region)
    return (
        {"region": region, "api_url": api_url, "app_url": app_url},
        [f"Region: {region}", f"API URL: {api_url}", f"App URL: {app_url}"],
    )


def _print_completion_tip(args) -> None:  # noqa: ANN001
    """Nudge users towards shell completion after a successful login.

    Suppressed under --json (would corrupt the JSON output) and --quiet
    (user explicitly opted out of non-essential output).
    """
    if getattr(args, "json", False) or getattr(args, "quiet", False):
        return
    print("\nTip: enable shell completion with 'roboflow completion install'")  # noqa: T201


def _login(args):  # noqa: ANN001
    from roboflow.cli._output import output, output_error

    api_key = getattr(args, "login_api_key", None) or getattr(args, "api_key", None)
    workspace_id = getattr(args, "login_workspace", None) or getattr(args, "workspace", None)
    region = _validate_region(args, getattr(args, "region", None))
    force = getattr(args, "force", False)

    if api_key:
        # Non-interactive: validate key and fetch workspace info
        import requests

        from roboflow.config import resolve_url

        api_url = resolve_url("API_URL", region=region)
        app_url = resolve_url("APP_URL", region=region)
        resp = requests.post(api_url + "/?api_key=" + api_key)
        if resp.status_code == 401:
            output_error(args, "Invalid API key.", hint=f"Check your key at {app_url}/settings", exit_code=2)
            return
        if resp.status_code != 200:
            output_error(args, f"API error ({resp.status_code}).", exit_code=1)
            return

        r_login = resp.json()
        if r_login is None:
            output_error(args, "Invalid API key.", exit_code=2)
            return

        # The validation endpoint returns {"workspace": "<url>", ...}
        ws_url = workspace_id or r_login.get("workspace", "")
        if not ws_url:
            output_error(args, "Could not determine workspace.", hint="Pass --workspace explicitly.", exit_code=1)
            return

        # Fetch workspace name from the API
        ws_name = ws_url
        try:
            ws_resp = requests.get(f"{api_url}/{ws_url}?api_key={api_key}")
            ws_resp.raise_for_status()
            ws_json = ws_resp.json()
            ws_detail = ws_json.get("workspace", ws_json)
            ws_name = ws_detail.get("name", ws_url)
        except Exception:  # noqa: BLE001
            pass  # Fall back to using the URL as the name

        # Build config with workspace info
        config = _load_config()
        workspaces = config.get("workspaces", {})
        workspaces[ws_url] = {"url": ws_url, "name": ws_name, "apiKey": api_key}
        config["workspaces"] = workspaces
        config["RF_WORKSPACE"] = ws_url
        if region is not None:
            config["ROBOFLOW_REGION"] = region
        _save_config(config)

        note = ""
        if len(workspaces) == 1:
            note = "\n  Note: API key login stores only the key's workspace. Use interactive login for all workspaces."
        output(
            args,
            {"status": "logged_in", "workspace": ws_url, "api_key": _mask_key(api_key)},
            text=f"Logged in. Default workspace: {ws_url}{note}",
        )
        _print_completion_tip(args)
    else:
        # Interactive flow
        import roboflow

        conf_path = _get_config_path()
        import os

        if os.path.isfile(conf_path) and not force:
            # Already logged in — show status
            config = _load_config()
            ws = config.get("RF_WORKSPACE", "unknown")
            output(
                args,
                {"status": "logged_in", "workspace": ws, "api_key": "****"},
                text=f"Already logged in. Default workspace: {ws}\nUse --force to re-login.",
            )
            return

        roboflow.login(workspace=workspace_id, force=force, region=region)
        # Re-read config after interactive login
        config = _load_config()
        ws = config.get("RF_WORKSPACE", "unknown")
        output(
            args,
            {"status": "logged_in", "workspace": ws, "api_key": "****"},
            text=f"Logged in. Default workspace: {ws}",
        )
        _print_completion_tip(args)
        _print_completion_tip(args)


def _status(args):  # noqa: ANN001
    import os

    from roboflow.cli._output import output, output_error

    region_data, region_lines = _region_status()
    config = _load_config()
    workspaces = config.get("workspaces", {})
    default_ws_url = config.get("RF_WORKSPACE")

    # Explicit --api-key flag takes priority, then env var
    explicit_api_key = getattr(args, "api_key", None)
    api_key = explicit_api_key or os.getenv("ROBOFLOW_API_KEY")

    # When an explicit --api-key is provided, always validate it against the
    # API rather than showing saved config — the user wants to check *this* key.
    if explicit_api_key or (api_key and not default_ws_url):
        import requests

        from roboflow.config import resolve_url

        assert api_key is not None  # guaranteed by the condition above
        resp = requests.post(resolve_url("API_URL", region=region_data["region"]) + "/?api_key=" + api_key)
        if resp.status_code == 200:
            ws_url = resp.json().get("workspace", "unknown")
            data = {
                "url": ws_url,
                "name": ws_url,
                "apiKey": _mask_key(api_key),
                **region_data,
            }
            lines = region_lines + [
                f"Workspace: {ws_url}",
                f"  URL: {ws_url}",
                f"  API Key: {_mask_key(api_key)}",
                "  (authenticated via --api-key or ROBOFLOW_API_KEY)",
            ]
            output(args, data, text="\n".join(lines))
        else:
            output_error(args, "API key is invalid or expired.", exit_code=2)
        return

    if not workspaces and not default_ws_url and not api_key:
        if getattr(args, "json", False):
            import json
            import sys

            payload = {
                "error": {
                    "message": "Not logged in.",
                    "hint": "Run 'roboflow auth login' to authenticate.",
                },
                **region_data,
            }
            print(json.dumps(payload), file=sys.stderr)
            raise SystemExit(2)

        output(args, region_data, text="\n".join(region_lines))
        output_error(args, "Not logged in.", hint="Run 'roboflow auth login' to authenticate.", exit_code=2)
        return  # unreachable, but helps mypy

    if not default_ws_url:
        output_error(args, "No default workspace configured.", hint="Run 'roboflow auth set-workspace <id>'.")
        return  # unreachable, but helps mypy

    workspaces_by_url = {w["url"]: w for w in workspaces.values()}
    default_ws = workspaces_by_url.get(default_ws_url)

    if default_ws:
        # Use stored API key, or fall back to env var
        display_key = api_key or default_ws.get("apiKey", "")
        masked = dict(default_ws)
        masked["apiKey"] = _mask_key(display_key)
        masked.update(region_data)
        lines = region_lines + [
            f"Workspace: {masked.get('name', 'unknown')}",
            f"  URL: {masked.get('url', 'unknown')}",
            f"  API Key: {masked['apiKey']}",
        ]
        output(args, masked, text="\n".join(lines))
    else:
        # RF_WORKSPACE is set but no matching workspace details
        data = {"url": default_ws_url, "name": default_ws_url, **region_data}
        output(
            args,
            data,
            text="\n".join(region_lines + [f"Workspace: {default_ws_url}", "  (no detailed info available)"]),
        )


def _set_region(args):  # noqa: ANN001
    from roboflow.cli._output import output
    from roboflow.config import get_effective_region, resolve_url

    region = _validate_region(args, args.region)
    assert region is not None
    previous_region = get_effective_region()

    config = _load_config()
    config["ROBOFLOW_REGION"] = region
    _save_config(config)

    effective_region = get_effective_region()
    api_url = resolve_url("API_URL")
    app_url = resolve_url("APP_URL")
    warning = (
        f"Stored credentials were issued by the previously configured {previous_region.upper()} platform. "
        "EU and US use separate authentication backends and API keys, so "
        f"'roboflow auth login --force --region {region}' may be needed."
    )
    environment_note = ""
    if effective_region != region:
        environment_note = (
            f"\nNote: ROBOFLOW_REGION overrides the saved choice; the effective region remains {effective_region}."
        )
    output(
        args,
        {
            "region": effective_region,
            "configured_region": region,
            "api_url": api_url,
            "app_url": app_url,
            "warning": warning,
        },
        text=(f"Region set to: {region}{environment_note}\nAPI URL: {api_url}\nApp URL: {app_url}\nWarning: {warning}"),
    )


def _set_workspace(args):  # noqa: ANN001
    from roboflow.cli._output import output

    workspace_id = args.workspace_id
    config = _load_config()
    config["RF_WORKSPACE"] = workspace_id
    _save_config(config)
    output(
        args,
        {"default_workspace": workspace_id},
        text=f"Default workspace set to: {workspace_id}",
    )


def _logout(args):  # noqa: ANN001
    import os

    from roboflow.cli._output import output

    conf_path = _get_config_path()
    if os.path.isfile(conf_path):
        os.remove(conf_path)

    output(
        args,
        {"status": "logged_out"},
        text="Logged out. Credentials removed.",
    )
