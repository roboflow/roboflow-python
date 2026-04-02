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

    It also bridges the global ``--api-key`` flag to the legacy ``-a`` flag
    that deployment handlers expect as ``args.api_key``.
    """

    def _wrapped(args: argparse.Namespace) -> None:
        from roboflow.cli._output import output_error

        # Bridge global --api-key (dest="api_key") to legacy -a (also dest="api_key")
        # The global flag may have set it; legacy handlers read args.api_key too.
        # No-op if both point to the same dest, but ensures it's set.

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

        # Success path: if --json, try to parse and re-emit as structured output
        text = captured.getvalue()
        if text:
            if getattr(args, "json", False):
                import json as _json

                from roboflow.cli._output import output

                try:
                    data = _json.loads(text)
                    output(args, data)
                except (ValueError, TypeError):
                    print(text, end="")
            else:
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

    # Improve help text to match other handlers
    deployment_parser.description = "Manage dedicated deployments"

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
    for _name, sub_parser in list(deployment_subs.choices.items()):
        defaults = sub_parser._defaults
        if "func" in defaults:
            defaults["func"] = _wrap_deployment_func(defaults["func"])

    # --- "create" as alias for "add" ---
    create_parser = deployment_subs.add_parser("create", help="Create a dedicated deployment (alias for 'add')")
    create_parser.add_argument(
        "deployment_name",
        help="Deployment name (5-15 lowercase chars, must start with a letter)",
    )
    create_parser.add_argument(
        "-m",
        "--machine-type",
        dest="machine_type",
        help="Machine type (run 'roboflow deployment machine-type' to see options)",
        required=True,
    )
    create_parser.add_argument(
        "-e", "--email", dest="creator_email", help="Your email address (must be a workspace member)", required=True
    )
    create_parser.add_argument(
        "-t",
        "--duration",
        help="Duration in hours (default: 3)",
        type=float,
        default=3,
    )
    create_parser.add_argument(
        "--no-delete-on-expiration",
        dest="no_delete_on_expiration",
        help="Keep deployment when expired",
        action="store_true",
    )
    create_parser.add_argument(
        "--inference-version",
        dest="inference_version",
        help="Inference server version (default: latest)",
        default="latest",
    )
    create_parser.add_argument(
        "--wait", dest="wait_on_pending", help="Wait for deployment to be ready", action="store_true"
    )
    create_parser.set_defaults(func=_wrap_deployment_func(add_deployment))

    # --- "machine-type" as alias for "machine_type" ---
    mt_parser = deployment_subs.add_parser("machine-type", help="List available machine types")
    mt_parser.set_defaults(func=_wrap_deployment_func(list_machine_types))
