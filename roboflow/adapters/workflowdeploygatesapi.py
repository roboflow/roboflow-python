"""Transport for the public Workflow deploy gates API.

Every route lives under ``{API_URL}/workspaces/{workspace}/workflows``. The API
shares the Workflow Evals API's bearer auth, ``Idempotency-Key`` rule and error
envelope, so this module reuses that transport's helpers. The per-endpoint
surface lives in :class:`roboflow.core.workflow_deploy_gates.WorkflowDeployGates`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional
from urllib.parse import quote

from roboflow.adapters.workflowevalsapi import (
    DEFAULT_TIMEOUT,
    WorkflowEvalError,
    _clean_params,
    _headers,
    error_from_response,
)
from roboflow.config import API_URL
from roboflow.util import http

if TYPE_CHECKING:
    import requests

# The codes a deploy gate refusal carries, on the deploy gates API and on the
# legacy publish paths (``POST /{workspace}/updateWorkflow``).
DEPLOY_GATE_ERROR_CODES = frozenset(
    {
        "deploy_check_required",
        "deploy_check_not_passed",
        "deploy_check_conflict",
        "deploy_check_unavailable",
        "deploy_check_local_inference_not_allowed",
        "deploy_gate_invalid_eval",
        "deploy_gate_feature_unavailable",
        "deploy_gate_permission_required",
    }
)


class WorkflowDeployGateError(WorkflowEvalError):
    """Error returned by the deploy gates API, or a publish a deploy gate refused.

    Same envelope as :class:`WorkflowEvalError`: ``code``, ``category``,
    ``retryable``, ``request_id``, ``message`` and ``details``. A refused publish
    (``deploy_check_required`` from ``updateWorkflow``, ``deploy_check_not_passed``
    from :meth:`WorkflowDeployGates.deploy`) lists the requests that publish the
    workflow in :attr:`next_steps`.
    """

    @property
    def next_steps(self) -> List[Dict[str, Any]]:
        """``[{action, method, path, description}]`` that move the workflow towards publishing."""
        steps = self.details.get("nextSteps")
        return steps if isinstance(steps, list) else []

    @property
    def gate_types(self) -> List[str]:
        """The gate types behind the refusal, e.g. ``["workflow_eval"]``."""
        gate_types = self.details.get("gateTypes")
        if isinstance(gate_types, list):
            return gate_types
        gate_type = self.details.get("gateType")
        return [gate_type] if isinstance(gate_type, str) else []


def deploy_gate_refusal(response: requests.Response) -> Optional[WorkflowDeployGateError]:
    """Return the deploy gate refusal a non-2xx legacy response carries, else ``None``."""
    error = error_from_response(response, WorkflowDeployGateError)
    if isinstance(error, WorkflowDeployGateError) and error.code in DEPLOY_GATE_ERROR_CODES:
        return error
    return None


def base_url(workspace_url: str) -> str:
    return f"{API_URL}/workspaces/{quote(workspace_url, safe='')}/workflows"


def request(
    api_key: str,
    workspace_url: str,
    method: str,
    path: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    body: Any = None,
    idempotency_key: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Any:
    """Call one deploy gates route and return the decoded JSON body.

    Raises :class:`WorkflowDeployGateError` for any non-2xx response.
    """
    response = http.request(
        method.upper(),
        f"{base_url(workspace_url)}{path}",
        params=_clean_params(params),
        json=body,
        headers=_headers(api_key, idempotency_key=idempotency_key),
        timeout=timeout,
    )
    if not 200 <= response.status_code < 300:
        raise error_from_response(response, WorkflowDeployGateError)
    if response.status_code == 204 or not response.content:
        return None
    try:
        return response.json()
    except ValueError:
        raise WorkflowDeployGateError(
            f"Expected JSON from {method.upper()} {path}", status_code=response.status_code
        ) from None
