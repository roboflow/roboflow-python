"""Async task polling helper shared by ``project fork`` and ``asynctasks wait``."""

from __future__ import annotations

import time
from typing import Any, Dict

from roboflow.adapters import rfapi

TERMINAL_STATUSES = {"completed", "failed"}


def poll_until_terminal(
    api_key: str,
    workspace_url: str,
    task_id: str,
    *,
    interval: float = 2.0,
    timeout: float = 1800.0,
) -> Dict[str, Any]:
    """Poll ``/{ws}/asynctasks/{id}`` until status is terminal or timeout elapses.

    A non-positive ``timeout`` disables the timeout. Returns the final
    status dict on terminal status. ``RoboflowError`` from the underlying
    API call is propagated; ``TimeoutError`` is raised if the deadline
    passes before a terminal status is observed.
    """
    deadline = None if timeout is None or timeout <= 0 else time.monotonic() + timeout
    while True:
        status = rfapi.get_async_task(api_key, workspace_url, task_id)
        if status.get("status") in TERMINAL_STATUSES:
            return status
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out after {timeout:.0f}s waiting for task {task_id} "
                f"(last status: {status.get('status')})."
            )
        time.sleep(interval)
