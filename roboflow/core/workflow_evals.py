"""Workflow Evals — wraps the public ``/workspaces/{workspace}/workflow-evals`` API.

A :class:`WorkflowEvals` client is bound to one workspace and exposes one
method per public endpoint, returning the server's JSON as plain dicts so new
response fields work without an SDK update. Get one from
:meth:`roboflow.core.workspace.Workspace.workflow_evals`::

    import roboflow

    evals = roboflow.Roboflow().workspace().workflow_evals()
    draft = evals.create(name="Boolean answer")
    spec = evals.create_spec({"name": "Boolean answer", "spec": {...}})

Conventions shared by every method:

* Commands that create resources or start background work send an
  ``Idempotency-Key``. One is generated when ``idempotency_key`` is omitted;
  pass your own to safely retry a request whose response was lost.
* Updates take the resource's current numeric ``revision`` (sent as
  ``If-Match``). A stale revision fails with HTTP 412.
* List methods return ``{"items": [...], "nextCursor": ...}``; pass
  ``cursor`` to read the next page.
* Errors raise :class:`roboflow.adapters.workflowevalsapi.WorkflowEvalError`.

The workspace needs the Workflow Evals feature, and the API key needs the
``workflow-evals:read|write|run|export`` scopes for the operations it calls.
"""

from __future__ import annotations

import mimetypes
import os
import time
from typing import Any, Callable, Dict, Iterator, List, Optional

from roboflow.adapters import workflowevalsapi as api
from roboflow.adapters.workflowevalsapi import WorkflowEvalError

RUN_TERMINAL_STATES = frozenset({"completed", "partially_completed", "failed", "cancelled"})
TASK_TERMINAL_STATES = frozenset({"completed", "partially_completed", "failed", "cancelled"})


def _seg(value: Any) -> str:
    return api.segment(value)


def _body(**fields: Any) -> Dict[str, Any]:
    return {key: value for key, value in fields.items() if value is not None}


class WorkflowEvals:
    """Client for one workspace's Workflow Evals."""

    def __init__(self, api_key: str, workspace_url: str) -> None:
        self._api_key = api_key
        self.workspace_url = workspace_url

    # -- transport ----------------------------------------------------------

    def _call(self, method: str, path: str = "", **kwargs: Any) -> Any:
        return api.request(self._api_key, self.workspace_url, method, path, **kwargs)

    def _command(self, path: str, body: Any = None, idempotency_key: Optional[str] = None, **kwargs: Any) -> Any:
        return self._call(
            "POST",
            path,
            body=body if body is not None else {},
            idempotency_key=idempotency_key or api.new_idempotency_key(),
            **kwargs,
        )

    @staticmethod
    def _run_path(eval_id: str, run_id: str) -> str:
        return f"/{_seg(eval_id)}/runs/{_seg(run_id)}"

    @classmethod
    def _execution_path(cls, eval_id: str, run_id: str, execution_id: str) -> str:
        return f"{cls._run_path(eval_id, run_id)}/executions/{_seg(execution_id)}"

    # -- discovery ----------------------------------------------------------

    def capabilities(self) -> Dict[str, Any]:
        """Describe the public operations and the caller's effective actions."""
        return self._call("GET", "/capabilities")

    def openapi(self) -> Dict[str, Any]:
        """Return the machine-readable OpenAPI 3.1 contract."""
        return self._call("GET", "/openapi.json")

    def list_evaluators(self, *, semantic_type: Optional[str] = None, family: Optional[str] = None) -> Dict[str, Any]:
        """List the installed engine's evaluator catalog."""
        return self._call("GET", "/evaluators", params={"semanticType": semantic_type, "family": family})

    def get_evaluator(self, evaluator_type: str) -> Dict[str, Any]:
        """Return one evaluator with its configuration schema."""
        return self._call("GET", f"/evaluators/{_seg(evaluator_type)}")

    def get_schema(self, schema_name: str) -> Dict[str, Any]:
        """Return an authoring JSON Schema.

        ``schema_name`` is one of ``spec``, ``eval-dataset``, ``case``,
        ``binding-set``, ``subject`` or ``evaluator-config``.
        """
        return self._call("GET", f"/schemas/{_seg(schema_name)}")

    def get_operation(self, operation_id: str) -> Dict[str, Any]:
        """Read a deletion operation's status receipt."""
        return self._call("GET", f"/operations/{_seg(operation_id)}")

    # -- agent guidance -----------------------------------------------------

    def agent_manifest(self) -> Dict[str, Any]:
        """Return ``{engineVersion, manifest}`` for MCP/agent clients."""
        return self._call("GET", "/agent/manifest")

    def list_agent_skills(self) -> Dict[str, Any]:
        return self._call("GET", "/agent/skills")

    def get_agent_skill(self, resource_id: str) -> Dict[str, Any]:
        """Return a skill's Markdown, e.g. ``skill:create-eval``."""
        return self._call("GET", f"/agent/skills/{_seg(resource_id)}")

    def list_agent_bundles(self) -> Dict[str, Any]:
        return self._call("GET", "/agent/bundles")

    def get_agent_bundle(self, resource_id: str) -> Dict[str, Any]:
        """Return a parsed bundle, e.g. ``bundle:create-eval``."""
        return self._call("GET", f"/agent/bundles/{_seg(resource_id)}")

    def get_agent_asset(self, resource_id: str) -> Dict[str, Any]:
        """Return the original content of a registered bundle dependency."""
        return self._call("GET", f"/agent/assets/{_seg(resource_id)}")

    # -- evals --------------------------------------------------------------

    def list(
        self,
        *,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
        include_spec_summary: bool = False,
        search: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List Evals, newest first."""
        params = {
            "limit": limit,
            "cursor": cursor,
            "include": "specSummary" if include_spec_summary else None,
            "search": search,
        }
        return self._call("GET", "", params=params)

    def create(
        self,
        body: Optional[Dict[str, Any]] = None,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        spec_id: Optional[str] = None,
        eval_dataset_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create an Eval. Every field is optional; ``{}`` creates an untitled draft.

        ``body`` may carry any other accepted field (``subject``,
        ``subjectBindings``, ``serverConfiguration``); keyword arguments win.
        """
        payload = {
            **(body or {}),
            **_body(name=name, description=description, specId=spec_id, evalDatasetId=eval_dataset_id),
        }
        return self._command("", payload, idempotency_key)

    def get(self, eval_id: str) -> Dict[str, Any]:
        return self._call("GET", f"/{_seg(eval_id)}")

    def get_by_url(self, eval_url: str) -> Dict[str, Any]:
        return self._call("GET", f"/by-url/{_seg(eval_url)}")

    def update(self, eval_id: str, body: Dict[str, Any], *, revision: Any) -> Dict[str, Any]:
        """Patch an Eval (name, description, specId, evalDatasetId, subject(s), bindings, server config)."""
        return self._call("PATCH", f"/{_seg(eval_id)}", body=body, revision=revision)

    def delete(self, eval_id: str, *, deletion_key: Optional[str] = None) -> Dict[str, Any]:
        """Delete an Eval and everything it exclusively owns.

        Without ``deletion_key`` the server answers ``409
        deletion_confirmation_required`` with the impact and a short-lived key
        in ``error.details``; resend with that key to confirm. See
        :meth:`delete_confirmed` for the two-step flow in one call.
        """
        return self._call("DELETE", f"/{_seg(eval_id)}", deletion_key=deletion_key)

    def delete_confirmed(self, eval_id: str) -> Dict[str, Any]:
        """Request the deletion key and immediately confirm the Eval deletion."""
        return self._delete_confirmed(lambda key: self.delete(eval_id, deletion_key=key))

    @staticmethod
    def _delete_confirmed(send: Callable[[Optional[str]], Dict[str, Any]]) -> Dict[str, Any]:
        try:
            return send(None)
        except WorkflowEvalError as exc:
            key = exc.details.get("deletionKey") if exc.code == "deletion_confirmation_required" else None
            if not key:
                raise
            return send(key)

    # -- specs --------------------------------------------------------------

    def list_specs(self, *, limit: Optional[int] = None, cursor: Optional[str] = None) -> Dict[str, Any]:
        return self._call("GET", "/specs", params={"limit": limit, "cursor": cursor})

    def create_spec(self, body: Dict[str, Any], *, idempotency_key: Optional[str] = None) -> Dict[str, Any]:
        """Validate, compile and save a Spec: ``{name?, description?, spec}``."""
        return self._command("/specs", body, idempotency_key)

    def get_spec(self, spec_id: str) -> Dict[str, Any]:
        return self._call("GET", f"/specs/{_seg(spec_id)}")

    def update_spec(self, spec_id: str, body: Dict[str, Any], *, revision: Any) -> Dict[str, Any]:
        return self._call("PATCH", f"/specs/{_seg(spec_id)}", body=body, revision=revision)

    def validate_spec(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Validate an unsaved Spec. Accepts the engine Spec or ``{"spec": ...}``."""
        body = spec if "spec" in spec else {"spec": spec}
        return self._call("POST", "/specs/validate", body=body)

    def list_compatible_datasets(
        self, spec_id: str, *, limit: Optional[int] = None, cursor: Optional[str] = None
    ) -> Dict[str, Any]:
        return self._call(
            "GET", f"/specs/{_seg(spec_id)}/compatible-eval-datasets", params={"limit": limit, "cursor": cursor}
        )

    def iter_ai_draft(
        self, eval_id: str, instructions: Optional[str] = None, *, idempotency_key: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """Stream AI drafting events (``status``, ``delta``, ``draft``, ``error``, ``done``).

        Consumes AI credits. Reusing a completed ``idempotency_key`` replays the
        final draft without another model call.
        """
        return api.stream_events(
            self._api_key,
            self.workspace_url,
            "/specs/ai-draft",
            body=_body(evalId=eval_id, instructions=instructions),
            idempotency_key=idempotency_key or api.new_idempotency_key(),
        )

    def ai_draft(
        self,
        eval_id: str,
        instructions: Optional[str] = None,
        *,
        idempotency_key: Optional[str] = None,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Draft a Spec and bindings with AI and return the terminal ``draft`` payload.

        The draft is not persisted; save the Spec and confirm bindings
        separately.
        """
        draft: Optional[Dict[str, Any]] = None
        for event in self.iter_ai_draft(eval_id, instructions, idempotency_key=idempotency_key):
            if on_event:
                on_event(event)
            if event["event"] == "draft":
                draft = event["data"]
            elif event["event"] == "error":
                data = event["data"]
                message = data.get("message") if isinstance(data, dict) else str(data)
                raise WorkflowEvalError(message or "AI draft failed", code="ai_draft_failed")
        if draft is None:
            raise WorkflowEvalError("AI draft stream ended without a draft", code="ai_draft_incomplete")
        return draft

    # -- eval datasets ------------------------------------------------------

    def list_datasets(self, *, limit: Optional[int] = None, cursor: Optional[str] = None) -> Dict[str, Any]:
        return self._call("GET", "/eval-datasets", params={"limit": limit, "cursor": cursor})

    def create_dataset(self, body: Dict[str, Any], *, idempotency_key: Optional[str] = None) -> Dict[str, Any]:
        """Create an empty Eval Dataset: ``{name, description?, inputs, groundTruthContract}``."""
        return self._command("/eval-datasets", body, idempotency_key)

    def get_dataset(self, dataset_id: str) -> Dict[str, Any]:
        return self._call("GET", f"/eval-datasets/{_seg(dataset_id)}")

    def update_dataset(self, dataset_id: str, body: Dict[str, Any], *, revision: Any) -> Dict[str, Any]:
        """Update Dataset metadata (``name``, ``description``)."""
        return self._call("PATCH", f"/eval-datasets/{_seg(dataset_id)}", body=body, revision=revision)

    def mutate_dataset_contract(self, dataset_id: str, body: Dict[str, Any], *, revision: Any) -> Dict[str, Any]:
        """Replace declared ``inputs`` and/or ``groundTruthContract``."""
        return self._call("POST", f"/eval-datasets/{_seg(dataset_id)}/contract-mutations", body=body, revision=revision)

    def list_compatible_specs(
        self, dataset_id: str, *, limit: Optional[int] = None, cursor: Optional[str] = None
    ) -> Dict[str, Any]:
        return self._call(
            "GET", f"/eval-datasets/{_seg(dataset_id)}/compatible-specs", params={"limit": limit, "cursor": cursor}
        )

    # -- cases --------------------------------------------------------------

    def list_cases(
        self,
        dataset_id: str,
        *,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
        slices: Optional[List[str]] = None,
        slice_match: Optional[str] = None,
        ground_truth_completeness: Optional[str] = None,
    ) -> Dict[str, Any]:
        params = {
            "limit": limit,
            "cursor": cursor,
            "slice": slices,
            "sliceMatch": slice_match,
            "groundTruthCompleteness": ground_truth_completeness,
        }
        return self._call("GET", f"/eval-datasets/{_seg(dataset_id)}/cases", params=params)

    def add_case(
        self, dataset_id: str, body: Dict[str, Any], *, idempotency_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create one Case: ``{name, inputs, groundTruth, slices?, evidenceRefs?, url?}``."""
        return self._command(f"/eval-datasets/{_seg(dataset_id)}/cases", body, idempotency_key)

    def get_case(self, dataset_id: str, case_id: str) -> Dict[str, Any]:
        return self._call("GET", f"/eval-datasets/{_seg(dataset_id)}/cases/{_seg(case_id)}")

    def replace_case(self, dataset_id: str, case_id: str, body: Dict[str, Any], *, revision: Any) -> Dict[str, Any]:
        """Replace one Case atomically."""
        return self._call(
            "PUT", f"/eval-datasets/{_seg(dataset_id)}/cases/{_seg(case_id)}", body=body, revision=revision
        )

    def delete_case(self, dataset_id: str, case_id: str, *, revision: Any) -> None:
        """Remove a Case from the live Dataset. Runs that froze it are unaffected."""
        self._call("DELETE", f"/eval-datasets/{_seg(dataset_id)}/cases/{_seg(case_id)}", revision=revision)

    def import_cases(
        self, dataset_id: str, body: Dict[str, Any], *, idempotency_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Import platform Sources, Datasets or uploaded artifacts as incomplete Cases.

        ``body`` is ``{inputField, items: [{kind: "source"|"dataset"|"artifact", ...}], slices?}``.
        Returns ``{intakeId, asyncTaskId, status, pollUrl}``.
        """
        return self._command(f"/eval-datasets/{_seg(dataset_id)}/cases/import", body, idempotency_key)

    def get_case_import(self, dataset_id: str, intake_id: str) -> Dict[str, Any]:
        return self._call("GET", f"/eval-datasets/{_seg(dataset_id)}/case-imports/{_seg(intake_id)}")

    def prepare_case_asset_upload(
        self,
        dataset_id: str,
        *,
        asset_name: str,
        content_type: str,
        expected_size: int,
        role: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Request a signed upload URL for a Case image (JPEG, PNG, WebP or GIF)."""
        body = _body(assetName=asset_name, contentType=content_type, expectedSize=expected_size, role=role)
        return self._command(f"/eval-datasets/{_seg(dataset_id)}/case-assets/uploads", body, idempotency_key)

    def complete_case_asset_upload(self, dataset_id: str, upload_id: str) -> Dict[str, Any]:
        """Verify an uploaded object and publish it as a Case asset artifact."""
        return self._call("POST", f"/eval-datasets/{_seg(dataset_id)}/case-assets/uploads/{_seg(upload_id)}/complete")

    def upload_case_asset(
        self,
        dataset_id: str,
        path: str,
        *,
        content_type: Optional[str] = None,
        role: Optional[str] = None,
        asset_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Upload a local image and return the completed asset (with ``artifactId``).

        Use ``{"artifactId": ...}`` as a Case input value afterwards.
        """
        content_type = content_type or mimetypes.guess_type(path)[0]
        if not content_type:
            raise ValueError(f"Cannot infer the content type of {path}; pass content_type explicitly")
        with open(path, "rb") as handle:
            data = handle.read()
        intent = self.prepare_case_asset_upload(
            dataset_id,
            asset_name=asset_name or os.path.basename(path),
            content_type=content_type,
            expected_size=len(data),
            role=role,
        )
        api.put_signed_upload(intent["uploadUrl"], data, intent.get("requiredHeaders") or {})
        return self.complete_case_asset_upload(dataset_id, intent["uploadId"])

    # -- bindings -----------------------------------------------------------

    def suggest_bindings(self, spec_id: str, eval_dataset_id: str, subject: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest a BindingSet for one WorkflowSubject. Nothing is persisted."""
        body = {"specId": spec_id, "evalDatasetId": eval_dataset_id, "subject": subject}
        return self._call("POST", "/bindings/suggest", body=body)

    def validate_bindings(
        self, spec_id: str, eval_dataset_id: str, subject: Dict[str, Any], binding_set: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Link a BindingSet against the Spec, Dataset and Subject and return diagnostics."""
        body = {"specId": spec_id, "evalDatasetId": eval_dataset_id, "subject": subject, "bindingSet": binding_set}
        return self._call("POST", "/bindings/validate", body=body)

    # -- case preparations --------------------------------------------------

    def start_case_preparation(
        self, eval_id: str, body: Dict[str, Any], *, idempotency_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run the Eval's Subject over Cases to produce candidate outputs for labeling.

        ``body`` is ``{runtime: {kind: "serverless"|"dedicated", url?}, caseSelection?, caseImportId?}``.
        """
        return self._command(f"/{_seg(eval_id)}/case-preparations", body, idempotency_key)

    def get_current_case_preparation(self, eval_id: str) -> Optional[Dict[str, Any]]:
        """Return the latest preparation for the current setup, or ``None``."""
        return self._call("GET", f"/{_seg(eval_id)}/case-preparations/current")

    def get_case_preparation(self, eval_id: str, preparation_id: str) -> Dict[str, Any]:
        return self._call("GET", f"/{_seg(eval_id)}/case-preparations/{_seg(preparation_id)}")

    def resume_case_preparation(self, eval_id: str, preparation_id: str) -> Dict[str, Any]:
        return self._call("POST", f"/{_seg(eval_id)}/case-preparations/{_seg(preparation_id)}/resume")

    def list_case_preparation_cases(
        self, eval_id: str, preparation_id: str, *, limit: Optional[int] = None, cursor: Optional[str] = None
    ) -> Dict[str, Any]:
        return self._call(
            "GET",
            f"/{_seg(eval_id)}/case-preparations/{_seg(preparation_id)}/cases",
            params={"limit": limit, "cursor": cursor},
        )

    # -- runs ---------------------------------------------------------------

    def list_runs(
        self,
        eval_id: str,
        *,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
        include_check_count: bool = False,
    ) -> Dict[str, Any]:
        """List Runs newest first; ``limit=1`` returns the latest Run."""
        params = {"limit": limit, "cursor": cursor, "include": "checkCount" if include_check_count else None}
        return self._call("GET", f"/{_seg(eval_id)}/runs", params=params)

    def start_run(self, eval_id: str, body: Dict[str, Any], *, idempotency_key: Optional[str] = None) -> Dict[str, Any]:
        """Start one grouped Run: ``{executions: [{subject, bindingSet}], caseSelection?, ...}``."""
        return self._command(f"/{_seg(eval_id)}/runs", body, idempotency_key)

    def get_run(self, eval_id: str, run_id: str) -> Dict[str, Any]:
        return self._call("GET", self._run_path(eval_id, run_id))

    def get_run_configuration(self, eval_id: str, run_id: str) -> Dict[str, Any]:
        return self._call("GET", f"{self._run_path(eval_id, run_id)}/configuration")

    def cancel_run(self, eval_id: str, run_id: str) -> Dict[str, Any]:
        return self._call("POST", f"{self._run_path(eval_id, run_id)}/cancel")

    def retry_run(self, eval_id: str, run_id: str, *, idempotency_key: Optional[str] = None) -> Dict[str, Any]:
        """Retry missing or operationally failed Cases in the same Run."""
        return self._command(f"{self._run_path(eval_id, run_id)}/retry-failed", {}, idempotency_key)

    def replay_run(
        self,
        eval_id: str,
        run_id: str,
        body: Optional[Dict[str, Any]] = None,
        *,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a derived Run, optionally with replacement Subjects and bindings."""
        return self._command(f"{self._run_path(eval_id, run_id)}/replay", body or {}, idempotency_key)

    def rescore_run(
        self, eval_id: str, run_id: str, body: Dict[str, Any], *, idempotency_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Rescore retained captures with the Eval's current saved Spec: ``{specId, executions}``."""
        return self._command(f"{self._run_path(eval_id, run_id)}/rescore", body, idempotency_key)

    def delete_run(self, eval_id: str, run_id: str, *, deletion_key: Optional[str] = None) -> Dict[str, Any]:
        """Delete a Run. Same two-step confirmation protocol as :meth:`delete`."""
        return self._call("DELETE", self._run_path(eval_id, run_id), deletion_key=deletion_key)

    def delete_run_confirmed(self, eval_id: str, run_id: str) -> Dict[str, Any]:
        return self._delete_confirmed(lambda key: self.delete_run(eval_id, run_id, deletion_key=key))

    def wait_for_run(
        self,
        eval_id: str,
        run_id: str,
        *,
        timeout: float = 3600,
        interval: float = 5,
        on_poll: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Poll a Run until it reaches a terminal state and return it.

        Stopping the wait never cancels the Run. Raises ``TimeoutError`` when
        ``timeout`` seconds elapse first.
        """
        deadline = time.monotonic() + timeout
        while True:
            run = self.get_run(eval_id, run_id)
            if on_poll:
                on_poll(run)
            state = run.get("state") or (run.get("run") or {}).get("state")
            if state in RUN_TERMINAL_STATES:
                return run
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Run {run_id} is still {state} after {timeout:g}s")
            time.sleep(interval)

    # -- executions and results ---------------------------------------------

    def list_executions(self, eval_id: str, run_id: str) -> Dict[str, Any]:
        return self._call("GET", f"{self._run_path(eval_id, run_id)}/executions")

    def get_execution(self, eval_id: str, run_id: str, execution_id: str) -> Dict[str, Any]:
        return self._call("GET", self._execution_path(eval_id, run_id, execution_id))

    def get_execution_configuration(self, eval_id: str, run_id: str, execution_id: str) -> Dict[str, Any]:
        return self._call("GET", f"{self._execution_path(eval_id, run_id, execution_id)}/configuration")

    def list_execution_input_cases(
        self,
        eval_id: str,
        run_id: str,
        execution_id: str,
        *,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Read frozen Case inputs for a local (client-run) Execution."""
        return self._call(
            "GET",
            f"{self._execution_path(eval_id, run_id, execution_id)}/input-cases",
            params={"limit": limit, "cursor": cursor},
        )

    def submit_capture(
        self, eval_id: str, run_id: str, execution_id: str, case_id: str, body: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Submit one locally captured Workflow output for server-side scoring.

        ``body`` is ``{attemptCount, shardIndex, output, runtime?}`` or
        ``{attemptCount, shardIndex, resume: true}``.
        """
        return self._call(
            "POST", f"{self._execution_path(eval_id, run_id, execution_id)}/cases/{_seg(case_id)}/capture", body=body
        )

    def get_results_overview(
        self,
        eval_id: str,
        run_id: str,
        execution_id: str,
        *,
        slices: Optional[List[str]] = None,
        slice_match: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._call(
            "GET",
            f"{self._execution_path(eval_id, run_id, execution_id)}/results/overview",
            params={"slice": slices, "sliceMatch": slice_match},
        )

    def list_case_results(
        self,
        eval_id: str,
        run_id: str,
        execution_id: str,
        *,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
        slices: Optional[List[str]] = None,
        slice_match: Optional[str] = None,
        order: Optional[str] = None,
        state: Optional[str] = None,
        case_id: Optional[str] = None,
        check_id: Optional[str] = None,
        judgment: Optional[str] = None,
        failed_check: Optional[bool] = None,
    ) -> Dict[str, Any]:
        params = {
            "limit": limit,
            "cursor": cursor,
            "slice": slices,
            "sliceMatch": slice_match,
            "order": order,
            "state": state,
            "caseId": case_id,
            "checkId": check_id,
            "judgment": judgment,
            "failedCheck": failed_check,
        }
        return self._call("GET", f"{self._execution_path(eval_id, run_id, execution_id)}/cases", params=params)

    def get_case_result(self, eval_id: str, run_id: str, execution_id: str, case_id: str) -> Dict[str, Any]:
        return self._call("GET", f"{self._execution_path(eval_id, run_id, execution_id)}/cases/{_seg(case_id)}")

    def compare(self, eval_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two or more completed Executions: ``{executionIds, limit?, cursor?, ...}``."""
        return self._call("POST", f"/{_seg(eval_id)}/comparisons", body=body)

    # -- exports ------------------------------------------------------------

    def start_export(
        self, eval_id: str, run_id: str, body: Dict[str, Any], *, idempotency_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Start a ``csv``/``json``/``xlsx`` export of a stable Run's results."""
        return self._command(f"{self._run_path(eval_id, run_id)}/exports", body, idempotency_key)

    def get_export_job(self, eval_id: str, run_id: str, async_task_id: str) -> Dict[str, Any]:
        return self._call("GET", f"{self._run_path(eval_id, run_id)}/export-jobs/{_seg(async_task_id)}")

    def get_export_download(self, eval_id: str, run_id: str, artifact_id: str) -> Dict[str, Any]:
        """Mint a fresh short-lived download URL for a completed export."""
        return self._call("GET", f"{self._run_path(eval_id, run_id)}/exports/{_seg(artifact_id)}")

    def wait_for_export(
        self, eval_id: str, run_id: str, async_task_id: str, *, timeout: float = 1800, interval: float = 3
    ) -> Dict[str, Any]:
        """Poll an export job until it finishes and return its final status."""
        deadline = time.monotonic() + timeout
        while True:
            job = self.get_export_job(eval_id, run_id, async_task_id)
            if job.get("state") in TASK_TERMINAL_STATES:
                return job
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Export {async_task_id} is still {job.get('state')} after {timeout:g}s")
            time.sleep(interval)

    # -- embedding analyses -------------------------------------------------

    def start_embeddings(
        self,
        eval_id: str,
        run_id: str,
        execution_id: str,
        provider: str,
        *,
        input_key: Optional[str] = None,
        retry: Optional[bool] = None,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate (or reuse) image embeddings for an Execution's Case inputs."""
        body = _body(provider=provider, inputKey=input_key, retry=retry)
        return self._command(
            f"{self._execution_path(eval_id, run_id, execution_id)}/analyses/embeddings", body, idempotency_key
        )

    def get_embeddings(
        self,
        eval_id: str,
        run_id: str,
        execution_id: str,
        *,
        provider: Optional[str] = None,
        async_task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Discover the job for ``provider`` or read a specific ``async_task_id`` attempt."""
        return self._call(
            "GET",
            f"{self._execution_path(eval_id, run_id, execution_id)}/analyses/embeddings",
            params={"provider": provider, "asyncTaskId": async_task_id},
        )

    def cancel_embeddings(self, eval_id: str, run_id: str, execution_id: str, async_task_id: str) -> Dict[str, Any]:
        return self._call(
            "POST",
            f"{self._execution_path(eval_id, run_id, execution_id)}/analyses/embeddings/{_seg(async_task_id)}/cancel",
        )
