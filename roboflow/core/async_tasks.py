"""Helpers for polling Roboflow async tasks."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

from roboflow.adapters import rfapi

NON_TERMINAL_STATUSES = frozenset({"created", "running"})


def poll_until_terminal(
    api_key: str,
    workspace_url: str,
    task_id: str,
    *,
    interval: float = 4.0,
    timeout: float = 1800.0,
    on_update: Optional[Callable[[Dict[str, Any]], None]] = None,
    polling_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Poll an async task until status is terminal or timeout elapses.

    If ``polling_url`` is provided, hit it verbatim (the server returns one
    alongside ``taskId`` from enqueue endpoints; it may point at a different
    host than ``API_URL``). Otherwise build the URL from ``API_URL`` /
    ``workspace_url`` / ``task_id`` via :func:`rfapi.get_async_task`.

    A non-positive ``timeout`` disables the timeout. Returns the final
    status dict on terminal status. ``RoboflowError`` from the underlying
    API call is propagated; ``TimeoutError`` is raised if the deadline
    passes before a terminal status is observed.
    """
    deadline = None if timeout <= 0 else time.monotonic() + timeout
    while True:
        if polling_url:
            status = rfapi.get_async_task_at(api_key, polling_url)
        else:
            status = rfapi.get_async_task(api_key, workspace_url, task_id)
        if status.get("status") not in NON_TERMINAL_STATUSES:
            return status
        if on_update:
            on_update(status)
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out after {timeout:.0f}s waiting for task {task_id} (last status: {status.get('status')})."
            )
        time.sleep(interval)
