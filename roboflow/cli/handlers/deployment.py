"""Deployment management commands (thin wrapper around roboflow.deployment)."""

from __future__ import annotations

import io
import sys
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import argparse


def _wrap_deployment_func(func: Callable[..., Any]) -> Callable[..., None]:
    """Wrap a legacy deployment handler to produce structured errors.

    The functions in ``roboflow.deployment`` use bare ``print()`` + ``exit()``
    for errors.  This wrapper intercepts both so that ``--json`` mode gets
    valid JSON on stderr and exit codes are normalised.
    """

    def _wrapped(args: argparse.Namespace) -> None:
        from roboflow.cli._output import output_error

        captured = io.StringIO()
        orig_stdout = sys.stdout

        try:
            # Capture stdout so we can inspect bare-text error messages
            sys.stdout = captured
            func(args)
        except SystemExit as exc:
            sys.stdout = orig_stdout
            code = exc.code if isinstance(exc.code, int) else 1
            text = captured.getvalue().strip()
            if text:
                # Normalise exit code: anything > 3 becomes 1
                output_error(args, text, exit_code=min(code, 3) if code else 1)
            else:
                output_error(args, "Deployment command failed.", exit_code=1)
            return
        finally:
            sys.stdout = orig_stdout

        # Success path: replay captured output
        text = captured.getvalue()
        if text:
            print(text, end="")

    return _wrapped


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``deployment`` command group by delegating to the existing module."""
    from roboflow.deployment import add_deployment, add_deployment_parser, list_machine_types

    add_deployment_parser(subparsers)

    # The deployment parser was just added to subparsers.choices
    deployment_parser = subparsers.choices.get("deployment")
    if deployment_parser is None:
        return

    # Set default so `roboflow deployment` (no subcommand) shows its own help
    deployment_parser.set_defaults(func=lambda args: deployment_parser.print_help())

    # Walk the parser's _actions list to find its _SubParsersAction.
    deployment_subs = None
    for action in deployment_parser._actions:
        if isinstance(action, type(subparsers)):
            deployment_subs = action
            break

    if deployment_subs is None:
        return

    # Wrap all existing deployment subcommand handlers for structured errors
    for name, sub_parser in list(deployment_subs.choices.items()):
        defaults = sub_parser._defaults
        if "func" in defaults:
            defaults["func"] = _wrap_deployment_func(defaults["func"])

    # --- "create" as alias for "add" ---
    create_parser = deployment_subs.add_parser("create", help="Create a dedicated deployment (alias for 'add')")
    create_parser.add_argument("-a", "--api_key", help="api key")
    create_parser.add_argument(
        "deployment_name",
        help="deployment name, must contain 5-15 lowercase characters, first character must be a letter",
    )
    create_parser.add_argument(
        "-m",
        "--machine_type",
        help="machine type, run `roboflow deployment machine_type` to see available options",
        required=True,
    )
    create_parser.add_argument(
        "-e", "--creator_email", help="your email address (must be added to the workspace)", required=True
    )
    create_parser.add_argument(
        "-t",
        "--duration",
        help="duration, how long you want to keep the deployment (unit: hour, default: 3)",
        type=float,
        default=3,
    )
    create_parser.add_argument(
        "-nodel", "--no_delete_on_expiration", help="keep when expired (default: False)", action="store_true"
    )
    create_parser.add_argument(
        "-v",
        "--inference_version",
        help="inference server version (default: latest)",
        default="latest",
    )
    create_parser.add_argument("-w", "--wait_on_pending", help="wait if deployment is pending", action="store_true")
    create_parser.set_defaults(func=_wrap_deployment_func(add_deployment))

    # --- "machine-type" as alias for "machine_type" ---
    mt_parser = deployment_subs.add_parser("machine-type", help="List machine types (alias for 'machine_type')")
    mt_parser.add_argument("-a", "--api_key", help="api key")
    mt_parser.set_defaults(func=_wrap_deployment_func(list_machine_types))
