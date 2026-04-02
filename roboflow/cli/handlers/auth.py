"""Auth commands: login, logout, status, set-workspace."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``auth`` command group."""
    auth_parser = subparsers.add_parser("auth", help="Manage authentication and credentials")
    auth_sub = auth_parser.add_subparsers(title="auth commands", dest="auth_command")

    # --- auth login ---
    login_p = auth_sub.add_parser("login", help="Log in to Roboflow")
    login_p.add_argument(
        "--api-key",
        dest="login_api_key",
        default=None,
        help="API key (skip interactive prompt)",
    )
    login_p.add_argument(
        "--workspace",
        dest="login_workspace",
        default=None,
        help="Set default workspace during login",
    )
    login_p.add_argument(
        "--force",
        "-f",
        action="store_true",
        default=False,
        help="Force re-login even if already logged in",
    )
    login_p.set_defaults(func=_login)

    # --- auth status ---
    status_p = auth_sub.add_parser("status", help="Show current auth status")
    status_p.set_defaults(func=_status)

    # --- auth set-workspace ---
    sw_p = auth_sub.add_parser("set-workspace", help="Set the default workspace")
    sw_p.add_argument("workspace_id", help="Workspace URL or ID to set as default")
    sw_p.set_defaults(func=_set_workspace)

    # --- auth logout ---
    logout_p = auth_sub.add_parser("logout", help="Remove stored credentials")
    logout_p.set_defaults(func=_logout)

    # Default: show help when no subcommand given
    auth_parser.set_defaults(func=lambda args: auth_parser.print_help())


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

    path = _get_config_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def _mask_key(key: str) -> str:
    if not key or len(key) <= 4:
        return "****"
    return key[:2] + "*" * (len(key) - 4) + key[-2:]


def _login(args: argparse.Namespace) -> None:
    from roboflow.cli._output import output, output_error

    api_key = getattr(args, "login_api_key", None) or getattr(args, "api_key", None)
    workspace_id = getattr(args, "login_workspace", None) or getattr(args, "workspace", None)
    force = getattr(args, "force", False)

    if api_key:
        # Non-interactive: validate key and fetch workspace info
        import requests

        from roboflow.config import API_URL

        resp = requests.post(API_URL + "/?api_key=" + api_key)
        if resp.status_code == 401:
            output_error(args, "Invalid API key.", hint="Check your key at app.roboflow.com/settings", exit_code=2)
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
            from roboflow.adapters import rfapi

            ws_json = rfapi.get_workspace(api_key, ws_url)
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
        _save_config(config)

        note = ""
        if len(workspaces) == 1:
            note = "\n  Note: API key login stores only the key's workspace. Use interactive login for all workspaces."
        output(
            args,
            {"status": "logged_in", "workspace": ws_url, "api_key": _mask_key(api_key)},
            text=f"Logged in. Default workspace: {ws_url}{note}",
        )
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

        roboflow.login(workspace=workspace_id, force=force)
        # Re-read config after interactive login
        config = _load_config()
        ws = config.get("RF_WORKSPACE", "unknown")
        output(
            args,
            {"status": "logged_in", "workspace": ws, "api_key": "****"},
            text=f"Logged in. Default workspace: {ws}",
        )


def _status(args: argparse.Namespace) -> None:
    import os

    from roboflow.cli._output import output, output_error

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

        from roboflow.config import API_URL

        assert api_key is not None  # guaranteed by the condition above
        resp = requests.post(API_URL + "/?api_key=" + api_key)
        if resp.status_code == 200:
            ws_url = resp.json().get("workspace", "unknown")
            data = {"url": ws_url, "name": ws_url, "apiKey": _mask_key(api_key)}
            lines = [
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
        lines = [
            f"Workspace: {masked.get('name', 'unknown')}",
            f"  URL: {masked.get('url', 'unknown')}",
            f"  API Key: {masked['apiKey']}",
        ]
        output(args, masked, text="\n".join(lines))
    else:
        # RF_WORKSPACE is set but no matching workspace details
        data = {"url": default_ws_url, "name": default_ws_url}
        output(
            args,
            data,
            text=f"Workspace: {default_ws_url}\n  (no detailed info available)",
        )


def _set_workspace(args: argparse.Namespace) -> None:
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


def _logout(args: argparse.Namespace) -> None:
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
