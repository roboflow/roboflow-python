"""Async task polling commands.

These mirror the generic ``GET /:workspace/asynctasks/:id`` endpoint so any
backend operation that returns ``{taskId, url}`` can be inspected with the
same CLI tools.
"""

from __future__ import annotations

from typing import Annotated

import typer

from roboflow.cli._compat import SortedGroup, ctx_to_args

asynctasks_app = typer.Typer(
    cls=SortedGroup,
    help="Inspect async background tasks (e.g. project forks)",
    no_args_is_help=True,
)


@asynctasks_app.command("get")
def get_async_task(
    ctx: typer.Context,
    task_id: Annotated[str, typer.Argument(help="Async task id (returned by /projects/fork etc.)")],
) -> None:
    """Show the current status of an async task."""
    args = ctx_to_args(ctx, task_id=task_id)
    _get_async_task(args)


@asynctasks_app.command("wait")
def wait_async_task(
    ctx: typer.Context,
    task_id: Annotated[str, typer.Argument(help="Async task id")],
    timeout: Annotated[
        int,
        typer.Option("--timeout", help="Seconds to wait for completion (0 = no timeout)."),
    ] = 1800,
) -> None:
    """Block until an async task is completed or failed."""
    args = ctx_to_args(ctx, task_id=task_id, timeout=timeout)
    _wait_async_task(args)


# ---------------------------------------------------------------------------
# Business logic
# ---------------------------------------------------------------------------


def _resolve_ws_and_key(args):  # noqa: ANN001
    from roboflow.cli._output import output_error
    from roboflow.cli._resolver import resolve_default_workspace
    from roboflow.config import load_roboflow_api_key

    workspace_url = args.workspace or resolve_default_workspace(api_key=args.api_key)
    if not workspace_url:
        output_error(
            args,
            "No workspace specified.",
            hint="Use --workspace or run 'roboflow auth login'.",
            exit_code=2,
        )
        return None, None
    api_key = args.api_key or load_roboflow_api_key(workspace_url)
    if not api_key:
        output_error(
            args,
            "No API key found.",
            hint="Set ROBOFLOW_API_KEY or run 'roboflow auth login'.",
            exit_code=2,
        )
        return None, None
    return workspace_url, api_key


def _get_async_task(args):  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error

    workspace_url, api_key = _resolve_ws_and_key(args)
    if not api_key:
        return

    try:
        status = rfapi.get_async_task(api_key, workspace_url, args.task_id)
    except rfapi.RoboflowError as exc:
        # Server returns 404 for unknown ids OR cross-workspace probes.
        output_error(args, str(exc), exit_code=3)
        return

    output(args, status, text=f"taskId={status.get('taskId')} status={status.get('status')}")


def _wait_async_task(args):  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error
    from roboflow.core.async_tasks import poll_until_terminal

    workspace_url, api_key = _resolve_ws_and_key(args)
    if not api_key:
        return

    def _print_progress(status):  # noqa: ANN001
        if args.json:
            return
        progress = status.get("progress")
        if not isinstance(progress, dict):
            return
        # Don't use `or` here: `current == 0` is a legitimate value.
        current = progress["current"] if "current" in progress else progress.get("completed")
        total = progress.get("total")
        if current is not None and total is not None:
            print(f"Task progress: {current}/{total}", flush=True)

    try:
        final = poll_until_terminal(
            api_key,
            workspace_url,
            args.task_id,
            timeout=args.timeout,
            on_update=_print_progress,
        )
    except rfapi.RoboflowError as exc:
        output_error(args, str(exc), exit_code=3)
        return
    except TimeoutError as exc:
        output_error(args, str(exc))
        return

    if final.get("status") == "failed":
        output_error(args, final.get("error") or "Task failed.")
        return

    output(args, final, text=f"taskId={final.get('taskId')} status={final.get('status')}")
