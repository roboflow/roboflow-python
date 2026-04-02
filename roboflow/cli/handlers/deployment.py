"""Deployment management commands.

Builds clean, kebab-case subcommands that delegate to the handler
functions in ``roboflow.deployment``.  Legacy snake_case names are
registered as hidden aliases (``argparse.SUPPRESS``) so old scripts
keep working.
"""

from __future__ import annotations

import argparse
import io
import sys
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Wrapper that captures legacy handler stdout/exit and normalises output
# ---------------------------------------------------------------------------


def _wrap(func: Callable[..., Any]) -> Callable[..., None]:
    """Wrap a legacy deployment handler for structured errors + JSON output."""

    def _wrapped(args: argparse.Namespace) -> None:
        from roboflow.cli._output import output, output_error

        captured = io.StringIO()
        orig_stdout = sys.stdout
        try:
            sys.stdout = captured
            func(args)
        except SystemExit as exc:
            sys.stdout = orig_stdout
            code = exc.code if isinstance(exc.code, int) else 1
            text = captured.getvalue().strip()
            if text:
                output_error(args, text, exit_code=min(code, 3) if code else 1)
            else:
                output_error(args, "Deployment command failed.", exit_code=1)
            return
        finally:
            sys.stdout = orig_stdout

        text = captured.getvalue()
        if text:
            if getattr(args, "json", False):
                import json

                try:
                    data = json.loads(text)
                    output(args, data)
                except (ValueError, TypeError):
                    print(text, end="")
            else:
                print(text, end="")

    return _wrapped


# ---------------------------------------------------------------------------
# Hidden-alias helper
# ---------------------------------------------------------------------------

_HIDDEN = argparse.SUPPRESS


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``deployment`` command group with clean kebab-case names."""
    from roboflow.cli import _CleanHelpFormatter
    from roboflow.deployment import (
        add_deployment,
        delete_deployment,
        get_deployment,
        get_deployment_log,
        get_deployment_usage,
        get_workspace_usage,
        list_deployment,
        list_machine_types,
        pause_deployment,
        resume_deployment,
    )

    dep = subparsers.add_parser("deployment", help="Manage dedicated deployments", formatter_class=_CleanHelpFormatter)
    sub = dep.add_subparsers(title="deployment commands", dest="deployment_command")

    # --- machine-type (canonical) ---
    mt = sub.add_parser("machine-type", help="List available machine types")
    mt.set_defaults(func=_wrap(list_machine_types))

    # --- create (canonical, replaces "add") ---
    create = sub.add_parser("create", help="Create a dedicated deployment")
    create.add_argument("deployment_name", help="Deployment name (5-15 lowercase chars, starts with letter)")
    create.add_argument(
        "-m",
        "--machine-type",
        dest="machine_type",
        required=True,
        help="Machine type (run 'roboflow deployment machine-type' to list options)",
    )
    create.add_argument(
        "-e",
        "--email",
        dest="creator_email",
        required=True,
        help="Your email (must be a workspace member)",
    )
    create.add_argument("--duration", type=float, default=3, help="Duration in hours (default: 3)")
    create.add_argument(
        "--no-delete-on-expiration",
        dest="no_delete_on_expiration",
        action="store_true",
        help="Keep deployment when it expires",
    )
    create.add_argument(
        "--inference-version",
        dest="inference_version",
        default="latest",
        help="Inference server version (default: latest)",
    )
    create.add_argument("--wait", dest="wait_on_pending", action="store_true", help="Wait until deployment is ready")
    create.set_defaults(func=_wrap(add_deployment))

    # --- get ---
    get = sub.add_parser("get", help="Show details for a deployment")
    get.add_argument("deployment_name", help="Deployment name")
    get.add_argument("--wait", dest="wait_on_pending", action="store_true", help="Wait if deployment is pending")
    get.set_defaults(func=_wrap(get_deployment))

    # --- list ---
    ls = sub.add_parser("list", help="List deployments in workspace")
    ls.set_defaults(func=_wrap(list_deployment))

    # --- usage ---
    usage = sub.add_parser("usage", help="Show usage statistics")
    usage.add_argument("deployment_name", nargs="?", default=None, help="Deployment name (omit for workspace-wide)")
    usage.add_argument("--from", dest="from_timestamp", default=None, help="Start time (ISO 8601)")
    usage.add_argument("--to", dest="to_timestamp", default=None, help="End time (ISO 8601)")
    usage.set_defaults(func=_usage_handler)

    # --- pause ---
    pause = sub.add_parser("pause", help="Pause a deployment")
    pause.add_argument("deployment_name", help="Deployment name")
    pause.set_defaults(func=_wrap(pause_deployment))

    # --- resume ---
    resume = sub.add_parser("resume", help="Resume a paused deployment")
    resume.add_argument("deployment_name", help="Deployment name")
    resume.set_defaults(func=_wrap(resume_deployment))

    # --- delete ---
    delete = sub.add_parser("delete", help="Delete a deployment")
    delete.add_argument("deployment_name", help="Deployment name")
    delete.set_defaults(func=_wrap(delete_deployment))

    # --- log ---
    log = sub.add_parser("log", help="Show deployment logs")
    log.add_argument("deployment_name", help="Deployment name")
    log.add_argument("-d", "--duration", type=int, default=3600, help="Log window in seconds (default: 3600)")
    log.add_argument("-n", "--tail", type=int, default=10, help="Lines to show from end (max 50)")
    log.add_argument("-f", "--follow", action="store_true", help="Follow log output")
    log.set_defaults(func=_wrap(get_deployment_log))

    # --- hidden legacy aliases (exact old flag signatures for backwards compat) ---

    # machine_type → machine-type
    legacy_mt = sub.add_parser("machine_type", help=_HIDDEN)
    legacy_mt.add_argument("-a", "--api_key", default=None)
    legacy_mt.set_defaults(func=_wrap(list_machine_types))

    # add → create (with old flag names: -m/--machine_type, -e/--creator_email, etc.)
    legacy_add = sub.add_parser("add", help=_HIDDEN)
    legacy_add.add_argument("deployment_name")
    legacy_add.add_argument("-a", "--api_key", default=None)
    legacy_add.add_argument("-m", "--machine_type", required=True)
    legacy_add.add_argument("-e", "--creator_email", required=True)
    legacy_add.add_argument("-t", "--duration", type=float, default=3)
    legacy_add.add_argument("-nodel", "--no_delete_on_expiration", action="store_true")
    legacy_add.add_argument("-v", "--inference_version", default="latest")
    legacy_add.add_argument("-w", "--wait_on_pending", action="store_true")
    legacy_add.set_defaults(func=_wrap(add_deployment))

    # usage_workspace
    legacy_uw = sub.add_parser("usage_workspace", help=_HIDDEN)
    legacy_uw.add_argument("-a", "--api_key", default=None)
    legacy_uw.add_argument("-f", "--from_timestamp", default=None)
    legacy_uw.add_argument("-t", "--to_timestamp", default=None)
    legacy_uw.set_defaults(func=_wrap(get_workspace_usage))

    # usage_deployment
    legacy_ud = sub.add_parser("usage_deployment", help=_HIDDEN)
    legacy_ud.add_argument("-a", "--api_key", default=None)
    legacy_ud.add_argument("deployment_name")
    legacy_ud.add_argument("-f", "--from_timestamp", default=None)
    legacy_ud.add_argument("-t", "--to_timestamp", default=None)
    legacy_ud.set_defaults(func=_wrap(get_deployment_usage))

    # Default: show help when no subcommand given
    dep.set_defaults(func=lambda args: dep.print_help())


def _usage_handler(args: argparse.Namespace) -> None:
    """Dispatch to workspace or deployment usage based on whether a name was given."""
    from roboflow.deployment import get_deployment_usage, get_workspace_usage

    if args.deployment_name:
        _wrap(get_deployment_usage)(args)
    else:
        _wrap(get_workspace_usage)(args)
