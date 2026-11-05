"""Workflow deploy gates: wraps the public ``/workspaces/{workspace}/workflows`` gate API.

A deploy gate is a condition a saved Workflow version must meet before it
publishes. Today the only type is ``workflow_eval``: a fresh Run of the version
on one Workflow Eval must reach every check's minimum pass rate. While a
workflow has a gate, ``updateWorkflow`` no longer publishes its config; a
version publishes only through :meth:`WorkflowDeployGates.deploy`, once it
passed every gate. Get a client from
:meth:`roboflow.core.workspace.Workspace.workflow_deploy_gates`::

    import roboflow

    gates = roboflow.Roboflow().workspace().workflow_deploy_gates()
    saved = gates.save_version(workflow_id, definition)
    version_id = saved["version"]["id"]
    gates.run(workflow_id, version_id)
    verdicts = gates.wait_for_version(workflow_id, version_id)
    if verdicts["publishable"]:
        gates.deploy(workflow_id, version_id)

On an Auto-Publish workflow a version that passed every gate publishes itself
while it is still the latest saved draft, so :meth:`deploy` is only needed on a
Protected one.

Conventions shared by every method:

* Methods take the workflow document id (``workflow["id"]`` from
  :meth:`Workspace.get_workflow`), not its URL slug.
* :meth:`save_version` and :meth:`run` send an ``Idempotency-Key``. One is
  generated when ``idempotency_key`` is omitted; pass your own to safely retry
  a request whose response was lost.
* Methods return the server's JSON as plain dicts.
* Errors raise :class:`roboflow.adapters.workflowdeploygatesapi.WorkflowDeployGateError`.

Scopes: reads need ``workflow:read``, writes ``workflow:update``. :meth:`run`
also needs ``workflow-evals:run`` and ``model:infer``, and
:meth:`set_eval_gate_new_check_thresholds` also ``workflow-evals:write``. A
check's Eval Run results come back only with ``workflow-evals:read``.
Folder-scoped keys are refused.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Union

from roboflow.adapters import workflowdeploygatesapi as api
from roboflow.adapters.rfapi import _normalize_workflow_config
from roboflow.adapters.workflowevalsapi import new_idempotency_key, segment

ThresholdsArg = Union[Mapping[str, float], Iterable[Dict[str, Any]]]


class _Unset:
    def __repr__(self) -> str:
        return "UNSET"


# Default of ``expected_updated_at``: leave the stale-write check out.
UNSET: Any = _Unset()


def _thresholds(check_thresholds: ThresholdsArg) -> List[Dict[str, Any]]:
    """Accept ``{check_id: rate}`` or the API's ``[{checkId, minimumPassRate}]``."""
    if isinstance(check_thresholds, Mapping):
        return [{"checkId": check_id, "minimumPassRate": rate} for check_id, rate in check_thresholds.items()]
    return list(check_thresholds)


class WorkflowDeployGates:
    """Client for one workspace's workflow deploy gates."""

    def __init__(self, api_key: str, workspace_url: str) -> None:
        self._api_key = api_key
        self.workspace_url = workspace_url

    # -- transport ----------------------------------------------------------

    def _call(self, method: str, path: str, **kwargs: Any) -> Any:
        return api.request(self._api_key, self.workspace_url, method, path, **kwargs)

    @staticmethod
    def _workflow_path(workflow_id: str) -> str:
        return f"/{segment(workflow_id)}"

    @classmethod
    def _version_path(cls, workflow_id: str, version_id: str) -> str:
        return f"{cls._workflow_path(workflow_id)}/versions/{segment(version_id)}"

    @classmethod
    def _eval_gate_path(cls, workflow_id: str) -> str:
        return f"{cls._workflow_path(workflow_id)}/deploy-gates/workflow-eval"

    # -- discovery ----------------------------------------------------------

    def openapi(self) -> Dict[str, Any]:
        """Return the deploy gates API's OpenAPI document."""
        return self._call("GET", "/deploy-gates/openapi.json")

    def list(self, *, gate_type: Optional[str] = None, eval_id: Optional[str] = None) -> Dict[str, Any]:
        """List the workspace's deploy gates, optionally of one type or backed by one Eval."""
        return self._call("GET", "/deploy-gates", params={"type": gate_type, "evalId": eval_id})

    def list_for_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Return ``{"items": [{type, status}]}``, every gate the workflow has."""
        return self._call("GET", f"{self._workflow_path(workflow_id)}/deploy-gates")

    # -- versions and publishing --------------------------------------------

    def save_version(
        self,
        workflow_id: str,
        config: Union[str, Dict[str, Any]],
        *,
        version_name: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Save a config as the workflow's latest version. It never publishes.

        ``config`` is the Workflow JSON as a dict or a JSON string; a bare
        definition (``{"version", "inputs", "steps", "outputs"}``) is wrapped as
        ``update_workflow`` does. Returns ``{workflowId, version: {id, name,
        createdAt, email}, replayed}``; ``replayed`` is true when this
        ``idempotency_key`` had already saved the version.
        """
        body: Dict[str, Any] = {"config": _normalize_workflow_config(config)}
        if version_name is not None:
            body["versionName"] = version_name
        return self._call(
            "POST",
            f"{self._workflow_path(workflow_id)}/versions",
            body=body,
            idempotency_key=idempotency_key or new_idempotency_key(),
        )

    def get_version(self, workflow_id: str, version_id: str) -> Dict[str, Any]:
        """Return every gate's verdict on the version: ``{versionId, gates, publishable}``.

        Each gate is ``{type, status, reason}`` with ``status`` one of
        ``not_started``, ``pending``, ``passed`` or ``failed``.
        """
        return self._call("GET", f"{self._version_path(workflow_id, version_id)}/deploy-gates")

    def run(self, workflow_id: str, version_id: str, *, idempotency_key: Optional[str] = None) -> Dict[str, Any]:
        """Start every deploy gate of the workflow on the version and return the verdicts.

        A replay with the same ``idempotency_key`` returns the check it started,
        even a failed one, and never starts another Run. A new key reuses a
        pending or passed check and starts a new one on a failed, errored,
        cancelled or outdated check. A Workflow Eval gate runs the Eval on the
        server, which may consume credits.
        """
        return self._call(
            "POST",
            f"{self._version_path(workflow_id, version_id)}/deploy-gates/run",
            body={},
            idempotency_key=idempotency_key or new_idempotency_key(),
        )

    def wait_for_version(
        self,
        workflow_id: str,
        version_id: str,
        *,
        timeout: float = 3600,
        interval: float = 5,
        on_poll: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Poll :meth:`get_version` until no gate is ``pending`` and return the verdicts.

        Check ``publishable`` on the result: a gate can end ``failed``, or
        ``not_started`` when nothing was started. Stopping the wait never
        cancels a check. Raises ``TimeoutError`` when ``timeout`` seconds elapse
        first.
        """
        deadline = time.monotonic() + timeout
        while True:
            verdicts = self.get_version(workflow_id, version_id)
            if on_poll:
                on_poll(verdicts)
            if not any(gate.get("status") == "pending" for gate in verdicts.get("gates") or []):
                return verdicts
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Version {version_id} still has pending deploy gates after {timeout:g}s")
            time.sleep(interval)

    def deploy(self, workflow_id: str, version_id: str) -> Dict[str, Any]:
        """Publish the version and return ``{workflowId, workflowUrl, versionId, status}``.

        On a gated workflow it is refused with ``409 deploy_check_not_passed``
        until the version passed every gate. Needs a plan with workflow version
        history.
        """
        return self._call("POST", f"{self._version_path(workflow_id, version_id)}/deploy", body={})

    # -- workflow_eval gate -------------------------------------------------

    def get_eval_gate(self, workflow_id: str) -> Dict[str, Any]:
        """Return the Eval gate's status: ``{gate, checks, eval, subject}`` (``gate`` is null without one)."""
        return self._call("GET", self._eval_gate_path(workflow_id))

    def set_eval_gate(
        self,
        workflow_id: str,
        eval_id: str,
        *,
        check_thresholds: Optional[ThresholdsArg] = None,
        expected_updated_at: Optional[str] = UNSET,
    ) -> Dict[str, Any]:
        """Gate the workflow on an Eval, with optional per-check minimum pass rates.

        ``check_thresholds`` is ``{check_id: rate}`` or a list of
        ``{"checkId", "minimumPassRate"}``, with ``0 < rate <= 1``; checks left
        out need every applicable Case to pass. ``expected_updated_at`` is the
        gate's ``updatedAt`` as you read it, or ``None`` when you read no gate:
        the write is refused with 409 if the gate changed since. Leave it out
        to skip that check. Setting the gate retires earlier checks.
        """
        body: Dict[str, Any] = {"evalId": eval_id}
        if check_thresholds is not None:
            body["checkThresholds"] = _thresholds(check_thresholds)
        if expected_updated_at is not UNSET:
            body["expectedUpdatedAt"] = expected_updated_at
        return self._call("PUT", self._eval_gate_path(workflow_id), body=body)

    def update_eval_gate(
        self,
        workflow_id: str,
        check_thresholds: ThresholdsArg,
        *,
        expected_updated_at: Optional[str] = UNSET,
    ) -> Dict[str, Any]:
        """Change some checks' minimum pass rates; the others keep theirs.

        Same ``check_thresholds`` and ``expected_updated_at`` as :meth:`set_eval_gate`.
        """
        body: Dict[str, Any] = {"checkThresholds": _thresholds(check_thresholds)}
        if expected_updated_at is not UNSET:
            body["expectedUpdatedAt"] = expected_updated_at
        return self._call("PATCH", self._eval_gate_path(workflow_id), body=body)

    def set_eval_gate_new_check_thresholds(self, workflow_id: str, check_thresholds: ThresholdsArg) -> Dict[str, Any]:
        """Set thresholds for checks added to the Eval after the gate was set.

        Only checks without a threshold in the gate yet are accepted; until set
        they need every applicable Case to pass.
        """
        return self._call(
            "POST",
            f"{self._eval_gate_path(workflow_id)}/new-check-thresholds",
            body={"checkThresholds": _thresholds(check_thresholds)},
        )

    def remove_eval_gate(self, workflow_id: str) -> Dict[str, Any]:
        """Remove the Eval gate and return the gate status after the removal."""
        return self._call("DELETE", self._eval_gate_path(workflow_id))

    def get_eval_checks(self, workflow_id: str, *, version_id: Optional[str] = None) -> Dict[str, Any]:
        """Return one version's Eval deploy check, or ``{"items": [...]}`` for every version.

        With ``version_id`` the server answers 404 until a check was started.
        """
        return self._call("GET", f"{self._eval_gate_path(workflow_id)}/checks", params={"versionId": version_id})
