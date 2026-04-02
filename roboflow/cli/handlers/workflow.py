"""Workflow management commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``workflow`` command group."""
    from roboflow.cli._output import stub

    wf_parser = subparsers.add_parser("workflow", help="Manage workflows")
    wf_subs = wf_parser.add_subparsers(title="workflow commands", dest="workflow_command")

    # --- workflow list ---
    list_p = wf_subs.add_parser("list", help="List workflows in a workspace")
    list_p.set_defaults(func=stub)

    # --- workflow get ---
    get_p = wf_subs.add_parser("get", help="Show details for a workflow")
    get_p.add_argument("workflow_url", help="Workflow URL or ID")
    get_p.set_defaults(func=stub)

    # --- workflow create ---
    create_p = wf_subs.add_parser("create", help="Create a new workflow")
    create_p.add_argument("--name", required=True, help="Workflow name")
    create_p.add_argument("--definition", help="Path to JSON definition file")
    create_p.add_argument("--description", default=None, help="Workflow description")
    create_p.set_defaults(func=stub)

    # --- workflow update ---
    update_p = wf_subs.add_parser("update", help="Update an existing workflow")
    update_p.add_argument("workflow_url", help="Workflow URL or ID")
    update_p.add_argument("--definition", help="Path to JSON definition file")
    update_p.set_defaults(func=stub)

    # --- workflow version ---
    version_p = wf_subs.add_parser("version", help="Manage workflow versions")
    version_subs = version_p.add_subparsers(title="workflow version commands", dest="workflow_version_command")
    version_list_p = version_subs.add_parser("list", help="List versions of a workflow")
    version_list_p.add_argument("workflow_url", help="Workflow URL or ID")
    version_list_p.set_defaults(func=stub)
    version_p.set_defaults(func=lambda args: version_p.print_help())

    # --- workflow fork ---
    fork_p = wf_subs.add_parser("fork", help="Fork a workflow")
    fork_p.add_argument("workflow_url", help="Workflow URL or ID")
    fork_p.set_defaults(func=stub)

    # --- workflow build ---
    build_p = wf_subs.add_parser("build", help="Build a workflow from a prompt")
    build_p.add_argument("prompt", help="Natural language prompt describing the workflow")
    build_p.set_defaults(func=stub)

    # --- workflow run ---
    run_p = wf_subs.add_parser("run", help="Run a workflow")
    run_p.add_argument("workflow_url", help="Workflow URL or ID")
    run_p.add_argument("--input", dest="input", help="Input file or URL")
    run_p.set_defaults(func=stub)

    # --- workflow deploy ---
    deploy_p = wf_subs.add_parser("deploy", help="Deploy a workflow")
    deploy_p.add_argument("workflow_url", help="Workflow URL or ID")
    deploy_p.set_defaults(func=stub)

    # Default
    wf_parser.set_defaults(func=lambda args: wf_parser.print_help())
