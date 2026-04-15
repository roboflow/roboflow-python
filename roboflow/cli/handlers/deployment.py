"""Deployment management commands.

Builds clean, kebab-case subcommands that delegate to the handler
functions in ``roboflow.deployment``.  Legacy snake_case names are
registered as hidden aliases so old scripts keep working.
"""

from __future__ import annotations

import io
import sys
from typing import Annotated, Any, Callable, Optional

import typer

from roboflow.cli._compat import SortedGroup, ctx_to_args

# ---------------------------------------------------------------------------
# Wrapper that captures legacy handler stdout/exit and normalises output
# ---------------------------------------------------------------------------


def _wrap(func: Callable[..., Any]) -> Callable[..., None]:
    """Wrap a legacy deployment handler for structured errors + JSON output."""

    def _wrapped(args):  # noqa: ANN001
        from roboflow.cli._output import output, output_error

        captured = io.StringIO()
        orig_stdout = sys.stdout
        try:
            sys.stdout = captured
            func(args)
        except SystemExit as exc:
            sys.stdout = orig_stdout
            code = exc.code if isinstance(exc.code, int) else 1
            exit_code = {0: 1, 1: 1, 2: 2, 3: 3}.get(code, 1) if code else 1
            text = captured.getvalue().strip()
            if text:
                output_error(args, text, exit_code=exit_code)
            else:
                output_error(args, "Deployment command failed.", exit_code=1)
            return
        except Exception as exc:
            sys.stdout = orig_stdout
            output_error(
                args,
                f"Deployment service unavailable: {type(exc).__name__}",
                hint="The dedicated deployment service may be down or unreachable. Try again later.",
                exit_code=1,
            )
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


deployment_app = typer.Typer(cls=SortedGroup, help="Manage dedicated deployments", no_args_is_help=True)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@deployment_app.command("machine-type")
def machine_type(ctx: typer.Context) -> None:
    """List available machine types."""
    from roboflow.deployment import list_machine_types

    args = ctx_to_args(ctx)
    _wrap(list_machine_types)(args)


@deployment_app.command("create")
def create_deployment(
    ctx: typer.Context,
    deployment_name: Annotated[str, typer.Argument(help="Deployment name (5-15 lowercase chars, starts with letter)")],
    machine_type_opt: Annotated[
        str, typer.Option("-m", "--machine-type", help="Machine type (run 'roboflow deployment machine-type' to list)")
    ],
    creator_email: Annotated[str, typer.Option("-e", "--email", help="Your email (must be a workspace member)")],
    duration: Annotated[float, typer.Option(help="Duration in hours")] = 3,
    no_delete_on_expiration: Annotated[
        bool, typer.Option("--no-delete-on-expiration", help="Keep deployment when it expires")
    ] = False,
    inference_version: Annotated[str, typer.Option("--inference-version", help="Inference server version")] = "latest",
    wait_on_pending: Annotated[bool, typer.Option("--wait", help="Wait until deployment is ready")] = False,
) -> None:
    """Create a dedicated deployment."""
    from roboflow.deployment import add_deployment

    args = ctx_to_args(
        ctx,
        deployment_name=deployment_name,
        machine_type=machine_type_opt,
        creator_email=creator_email,
        duration=duration,
        no_delete_on_expiration=no_delete_on_expiration,
        inference_version=inference_version,
        wait_on_pending=wait_on_pending,
    )
    _wrap(add_deployment)(args)


@deployment_app.command("get")
def get_deployment(
    ctx: typer.Context,
    deployment_name: Annotated[str, typer.Argument(help="Deployment name")],
    wait_on_pending: Annotated[bool, typer.Option("--wait", help="Wait if deployment is pending")] = False,
) -> None:
    """Show details for a deployment."""
    from roboflow.deployment import get_deployment

    args = ctx_to_args(ctx, deployment_name=deployment_name, wait_on_pending=wait_on_pending)
    _wrap(get_deployment)(args)


@deployment_app.command("list")
def list_deployments(ctx: typer.Context) -> None:
    """List deployments in workspace."""
    from roboflow.deployment import list_deployment

    args = ctx_to_args(ctx)
    _wrap(list_deployment)(args)


@deployment_app.command("usage")
def usage(
    ctx: typer.Context,
    deployment_name: Annotated[Optional[str], typer.Argument(help="Deployment name (omit for workspace-wide)")] = None,
    from_timestamp: Annotated[Optional[str], typer.Option("--from", help="Start time (ISO 8601)")] = None,
    to_timestamp: Annotated[Optional[str], typer.Option("--to", help="End time (ISO 8601)")] = None,
) -> None:
    """Show usage statistics."""
    from roboflow.deployment import get_deployment_usage, get_workspace_usage

    args = ctx_to_args(
        ctx,
        deployment_name=deployment_name,
        from_timestamp=from_timestamp,
        to_timestamp=to_timestamp,
    )
    if deployment_name:
        _wrap(get_deployment_usage)(args)
    else:
        _wrap(get_workspace_usage)(args)


@deployment_app.command("pause")
def pause_deployment(
    ctx: typer.Context,
    deployment_name: Annotated[str, typer.Argument(help="Deployment name")],
) -> None:
    """Pause a deployment."""
    from roboflow.deployment import pause_deployment

    args = ctx_to_args(ctx, deployment_name=deployment_name)
    _wrap(pause_deployment)(args)


@deployment_app.command("resume")
def resume_deployment(
    ctx: typer.Context,
    deployment_name: Annotated[str, typer.Argument(help="Deployment name")],
) -> None:
    """Resume a paused deployment."""
    from roboflow.deployment import resume_deployment

    args = ctx_to_args(ctx, deployment_name=deployment_name)
    _wrap(resume_deployment)(args)


@deployment_app.command("delete")
def delete_deployment(
    ctx: typer.Context,
    deployment_name: Annotated[str, typer.Argument(help="Deployment name")],
) -> None:
    """Delete a deployment."""
    from roboflow.deployment import delete_deployment

    args = ctx_to_args(ctx, deployment_name=deployment_name)
    _wrap(delete_deployment)(args)


@deployment_app.command("log")
def deployment_log(
    ctx: typer.Context,
    deployment_name: Annotated[str, typer.Argument(help="Deployment name")],
    duration: Annotated[int, typer.Option("-d", "--duration", help="Log window in seconds")] = 3600,
    tail: Annotated[int, typer.Option("-n", "--tail", help="Lines to show from end (max 50)")] = 10,
    follow: Annotated[bool, typer.Option("-f", "--follow", help="Follow log output")] = False,
) -> None:
    """Show deployment logs."""
    from roboflow.deployment import get_deployment_log

    args = ctx_to_args(
        ctx,
        deployment_name=deployment_name,
        duration=duration,
        tail=tail,
        follow=follow,
    )
    _wrap(get_deployment_log)(args)


# ---------------------------------------------------------------------------
# Hidden legacy aliases
# ---------------------------------------------------------------------------


@deployment_app.command("machine_type", hidden=True)
def legacy_machine_type(
    ctx: typer.Context,
    api_key: Annotated[Optional[str], typer.Option("-a", "--api_key")] = None,
) -> None:
    """Legacy alias for machine-type."""
    from roboflow.deployment import list_machine_types

    args = ctx_to_args(ctx)
    if api_key:
        args.api_key = api_key
    _wrap(list_machine_types)(args)


@deployment_app.command("add", hidden=True)
def legacy_add(
    ctx: typer.Context,
    deployment_name: Annotated[str, typer.Argument()],
    machine_type_opt: Annotated[str, typer.Option("-m", "--machine_type")],
    creator_email: Annotated[str, typer.Option("-e", "--creator_email")],
    api_key: Annotated[Optional[str], typer.Option("-a", "--api_key")] = None,
    duration: Annotated[float, typer.Option("-t", "--duration")] = 3,
    no_delete_on_expiration: Annotated[bool, typer.Option("-nodel", "--no_delete_on_expiration")] = False,
    inference_version: Annotated[str, typer.Option("-v", "--inference_version")] = "latest",
    wait_on_pending: Annotated[bool, typer.Option("-w", "--wait_on_pending")] = False,
) -> None:
    """Legacy alias for create."""
    from roboflow.deployment import add_deployment

    args = ctx_to_args(
        ctx,
        deployment_name=deployment_name,
        machine_type=machine_type_opt,
        creator_email=creator_email,
        duration=duration,
        no_delete_on_expiration=no_delete_on_expiration,
        inference_version=inference_version,
        wait_on_pending=wait_on_pending,
    )
    if api_key:
        args.api_key = api_key
    _wrap(add_deployment)(args)


@deployment_app.command("usage_workspace", hidden=True)
def legacy_usage_workspace(
    ctx: typer.Context,
    api_key: Annotated[Optional[str], typer.Option("-a", "--api_key")] = None,
    from_timestamp: Annotated[Optional[str], typer.Option("-f", "--from_timestamp")] = None,
    to_timestamp: Annotated[Optional[str], typer.Option("-t", "--to_timestamp")] = None,
) -> None:
    """Legacy alias for usage (workspace)."""
    from roboflow.deployment import get_workspace_usage

    args = ctx_to_args(ctx, from_timestamp=from_timestamp, to_timestamp=to_timestamp)
    if api_key:
        args.api_key = api_key
    _wrap(get_workspace_usage)(args)


@deployment_app.command("usage_deployment", hidden=True)
def legacy_usage_deployment(
    ctx: typer.Context,
    deployment_name: Annotated[str, typer.Argument()],
    api_key: Annotated[Optional[str], typer.Option("-a", "--api_key")] = None,
    from_timestamp: Annotated[Optional[str], typer.Option("-f", "--from_timestamp")] = None,
    to_timestamp: Annotated[Optional[str], typer.Option("-t", "--to_timestamp")] = None,
) -> None:
    """Legacy alias for usage (deployment)."""
    from roboflow.deployment import get_deployment_usage

    args = ctx_to_args(
        ctx,
        deployment_name=deployment_name,
        from_timestamp=from_timestamp,
        to_timestamp=to_timestamp,
    )
    if api_key:
        args.api_key = api_key
    _wrap(get_deployment_usage)(args)
