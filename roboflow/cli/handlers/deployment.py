"""Deployment management commands (thin wrapper around roboflow.deployment)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``deployment`` command group by delegating to the existing module."""
    from roboflow.deployment import add_deployment, add_deployment_parser, list_machine_types

    add_deployment_parser(subparsers)

    # The deployment parser was just added to subparsers.choices
    deployment_parser = subparsers.choices.get("deployment")
    if deployment_parser is None:
        return

    # Walk the parser's _actions list to find its _SubParsersAction.
    # This avoids poking at the private _subparsers._group_actions chain.
    deployment_subs = None
    for action in deployment_parser._actions:
        if isinstance(action, type(subparsers)):
            deployment_subs = action
            break

    if deployment_subs is None:
        return

    # Add "create" as alias for "add"
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
    create_parser.set_defaults(func=add_deployment)

    # Add "machine-type" as alias for "machine_type"
    mt_parser = deployment_subs.add_parser("machine-type", help="List machine types (alias for 'machine_type')")
    mt_parser.add_argument("-a", "--api_key", help="api key")
    mt_parser.set_defaults(func=list_machine_types)
