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
        # Non-interactive: store key directly
        import requests

        from roboflow.config import API_URL

        # Validate the key
        resp = requests.post(API_URL + "/?api_key=" + api_key)
        if resp.status_code == 401:
            output_error(args, "Invalid API key.", hint="Check your key at app.roboflow.com/settings", exit_code=2)
        if resp.status_code != 200:
            output_error(args, f"API error ({resp.status_code}).", exit_code=1)

        r_login = resp.json()
        if r_login is None:
            output_error(args, "Invalid API key.", exit_code=2)

        config = {"workspaces": r_login}
        # Set default workspace
        first_ws_id = list(r_login.keys())[0]
        ws_url = r_login[first_ws_id]["url"]
        if workspace_id:
            # Verify requested workspace exists
            ws_by_url = {w["url"]: w for w in r_login.values()}
            if workspace_id in ws_by_url:
                ws_url = workspace_id
        config["RF_WORKSPACE"] = ws_url
        _save_config(config)

        output(
            args,
            {"status": "logged_in", "workspace": ws_url, "api_key": _mask_key(api_key)},
            text=f"Logged in. Default workspace: {ws_url}",
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
    from roboflow.cli._output import output, output_error
    from roboflow.config import get_conditional_configuration_variable

    workspaces = get_conditional_configuration_variable("workspaces", default={})
    if not workspaces:
        output_error(args, "Not logged in.", hint="Run 'roboflow auth login' to authenticate.", exit_code=2)
        return  # unreachable, but helps mypy

    workspaces_by_url = {w["url"]: w for w in workspaces.values()}
    default_ws_url = get_conditional_configuration_variable("RF_WORKSPACE", default=None)
    default_ws = workspaces_by_url.get(default_ws_url)

    if not default_ws:
        output_error(args, "No default workspace configured.", hint="Run 'roboflow auth set-workspace <id>'.")
        return  # unreachable, but helps mypy

    # Mask the API key
    masked = dict(default_ws)
    masked["apiKey"] = _mask_key(masked.get("apiKey", ""))

    lines = [
        f"Workspace: {masked.get('name', 'unknown')}",
        f"  URL: {masked.get('url', 'unknown')}",
        f"  API Key: {masked['apiKey']}",
    ]
    output(args, masked, text="\n".join(lines))


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
