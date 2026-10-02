"""Workflow Evals commands.

Wraps the public ``/workspaces/{workspace}/workflow-evals`` API: Evals,
Specs, Eval Datasets and Cases, binding suggestion/validation, Case
preparation, grouped Runs, Execution results, comparisons, exports,
embedding analyses, the engine catalog and agent guidance.

Request bodies follow the API's JSON contracts. Options named ``--body``
(and ``--subject`` / ``--binding-set``) accept inline JSON, ``@path.json``,
a path to a JSON file, or ``-`` for stdin. Explicit flags are merged on top
of ``--body``. Commands that create resources or start work send a fresh
``Idempotency-Key`` unless ``--idempotency-key`` is given; updates require the
resource's current ``--revision``.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Annotated, Any, Callable, Dict, List, Optional

import typer

from roboflow.cli._compat import SortedGroup, ctx_to_args

workflow_eval_app = typer.Typer(
    cls=SortedGroup,
    help="Evaluate Workflows against golden datasets (Workflow Evals)",
    no_args_is_help=True,
)
spec_app = typer.Typer(cls=SortedGroup, help="Author and validate evaluation Specs", no_args_is_help=True)
dataset_app = typer.Typer(cls=SortedGroup, help="Manage Eval Datasets", no_args_is_help=True)
case_app = typer.Typer(cls=SortedGroup, help="Manage Eval Dataset Cases", no_args_is_help=True)
binding_app = typer.Typer(cls=SortedGroup, help="Suggest and validate BindingSets", no_args_is_help=True)
preparation_app = typer.Typer(cls=SortedGroup, help="Prepare candidate outputs for Case labeling", no_args_is_help=True)
run_app = typer.Typer(cls=SortedGroup, help="Start and manage evaluation Runs", no_args_is_help=True)
execution_app = typer.Typer(cls=SortedGroup, help="Inspect Run Executions and results", no_args_is_help=True)
export_app = typer.Typer(cls=SortedGroup, help="Export Run results", no_args_is_help=True)
embedding_app = typer.Typer(cls=SortedGroup, help="Image embedding analyses for Executions", no_args_is_help=True)
evaluator_app = typer.Typer(cls=SortedGroup, help="Browse the evaluator catalog", no_args_is_help=True)
schema_app = typer.Typer(cls=SortedGroup, help="Read authoring JSON Schemas", no_args_is_help=True)
agent_app = typer.Typer(cls=SortedGroup, help="Read engine guidance for agents and MCP clients", no_args_is_help=True)

for _sub_app, _name in (
    (agent_app, "agent"),
    (binding_app, "binding"),
    (case_app, "case"),
    (dataset_app, "dataset"),
    (embedding_app, "embedding"),
    (evaluator_app, "evaluator"),
    (execution_app, "execution"),
    (export_app, "export"),
    (preparation_app, "preparation"),
    (run_app, "run"),
    (schema_app, "schema"),
    (spec_app, "spec"),
):
    workflow_eval_app.add_typer(_sub_app, name=_name)


# ---------------------------------------------------------------------------
# Shared option types
# ---------------------------------------------------------------------------

EvalOpt = Annotated[str, typer.Option("--eval", "-e", help="Eval id")]
RunOpt = Annotated[str, typer.Option("--run", "-r", help="Run id")]
ExecutionOpt = Annotated[str, typer.Option("--execution", "-x", help="Execution id")]
DatasetOpt = Annotated[str, typer.Option("--dataset", "-d", help="Eval Dataset id")]
RevisionOpt = Annotated[str, typer.Option("--revision", help="Current revision of the resource (sent as If-Match)")]
LimitOpt = Annotated[Optional[int], typer.Option("--limit", "-n", help="Page size")]
CursorOpt = Annotated[Optional[str], typer.Option("--cursor", help="Cursor from a previous page's nextCursor")]
BodyOpt = Annotated[
    Optional[str], typer.Option("--body", "-b", help="JSON request body: inline JSON, @file.json, file path, or -")
]
IdempotencyOpt = Annotated[
    Optional[str],
    typer.Option("--idempotency-key", help="UUID v4 to reuse when retrying (default: generated)"),
]
YesOpt = Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation prompt")]
SliceOpt = Annotated[Optional[List[str]], typer.Option("--slice", help="Filter by slice (repeatable)")]
SliceMatchOpt = Annotated[Optional[str], typer.Option("--slice-match", help="Slice match mode: any or all")]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _client(args: Any) -> Any:
    from roboflow.cli._resolver import resolve_ws_and_key
    from roboflow.core.workflow_evals import WorkflowEvals

    resolved = resolve_ws_and_key(args)
    if not resolved:
        return None
    workspace_url, api_key = resolved
    return WorkflowEvals(api_key, workspace_url)


def _error_hint(exc: Any) -> Optional[str]:
    status, code = exc.status_code, (exc.code or "")
    if code == "FEATURE_DISABLED":
        return "Workflow Evals is not enabled for this workspace."
    if code == "deletion_confirmation_required":
        return "Re-run the delete command to confirm, or pass --yes."
    if status == 428:
        return "Pass the resource's current --revision (shown by the matching get command)."
    if status == 412:
        return "The resource changed since you read it. Fetch it again and retry with the new --revision."
    if status == 401:
        return "Check your API key, or run 'roboflow auth login'."
    if status == 403:
        return (
            "The API key needs the workflow-evals:read/write/run/export scope for this operation "
            "and cannot be folder-scoped."
        )
    if status == 422:
        return "See error.details.diagnostics for the validation failures."
    return exc.hint


def _exit_code(exc: Any) -> int:
    if exc.status_code in (401, 403):
        return 2
    if exc.status_code == 404:
        return 3
    return 1


def _fail(args: Any, exc: Exception) -> None:
    from roboflow.adapters.workflowevalsapi import WorkflowEvalError
    from roboflow.cli._output import output_error

    if isinstance(exc, WorkflowEvalError):
        payload = exc.to_dict()
        payload.pop("hint", None)
        output_error(args, json.dumps({"error": payload}), hint=_error_hint(exc), exit_code=_exit_code(exc))
    else:
        output_error(args, str(exc))


def _invoke(
    args: Any,
    call: Callable[[Any], Any],
    text: Optional[Callable[[Any], Optional[str]]] = None,
    *,
    interactive: bool = False,
) -> None:
    """Resolve the client, run *call*, and print the result (or a structured error).

    ``interactive`` keeps stdout attached so confirmation prompts stay visible.
    """
    import contextlib

    from roboflow.cli._output import output, suppress_sdk_output

    client = _client(args)
    if client is None:
        return
    try:
        with contextlib.nullcontext() if interactive else suppress_sdk_output(args):
            result = call(client)
    except Exception as exc:  # noqa: BLE001 — every failure becomes a structured CLI error
        _fail(args, exc)
        return
    output(args, result, text=text(result) if text and not getattr(args, "json", False) else None)


def _load_json(args: Any, value: Optional[str], flag: str = "--body") -> Any:
    """Parse a JSON option: inline JSON, ``@path``, an existing file path, or ``-`` for stdin."""
    from roboflow.cli._output import output_error

    if value is None:
        return None
    try:
        if value == "-":
            return json.load(sys.stdin)
        path = value[1:] if value.startswith("@") else value
        if value.startswith("@") or (not value.lstrip().startswith(("{", "[")) and os.path.isfile(path)):
            with open(path, encoding="utf-8") as handle:
                return json.load(handle)
        return json.loads(value)
    except (OSError, ValueError) as exc:
        output_error(
            args,
            f"{flag} is not valid JSON: {exc}",
            hint=f"Pass inline JSON, @file.json, or - to read {flag} from stdin.",
        )
        return None


def _load_object(args: Any, value: Optional[str], flag: str = "--body") -> Dict[str, Any]:
    from roboflow.cli._output import output_error

    data = _load_json(args, value, flag)
    if data is None:
        return {}
    if not isinstance(data, dict):
        output_error(args, f"{flag} must be a JSON object.")
    return data


def _require(args: Any, body: Dict[str, Any], flag: str = "--body") -> Dict[str, Any]:
    from roboflow.cli._output import output_error

    if not body:
        output_error(args, f"{flag} is required.", hint="Run the command with --help for the expected JSON.")
    return body


def _merge(body: Dict[str, Any], **fields: Any) -> Dict[str, Any]:
    return {**body, **{key: value for key, value in fields.items() if value is not None}}


def _subject(
    args: Any,
    subject_json: Optional[str],
    workflow: Optional[str],
    workflow_version: Optional[str],
    subject_key: Optional[str],
) -> Dict[str, Any]:
    from roboflow.cli._output import output_error

    if subject_json:
        return _load_object(args, subject_json, "--subject")
    if not workflow:
        output_error(
            args,
            "A Workflow subject is required.",
            hint="Pass --workflow <workflow-id> (optionally --workflow-version) or --subject <json>.",
        )
    subject: Dict[str, Any] = {"kind": "savedWorkflow", "workflowId": workflow, "subjectKey": subject_key or workflow}
    if workflow_version:
        subject.update(kind="savedWorkflowVersion", workflowVersionId=workflow_version)
    return subject


def _pick(item: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = item.get(key)
        if value not in (None, ""):
            return value
    return ""


def _page_text(columns: Dict[str, tuple]) -> Callable[[Any], str]:
    """Render ``{"items": [...]}`` pages as a table; *columns* maps header -> candidate keys."""
    from roboflow.cli._table import format_table

    def render(page: Any) -> str:
        items = page.get("items", []) if isinstance(page, dict) else []
        rows = [{header: _pick(item, *keys) for header, keys in columns.items()} for item in items]
        table = format_table(rows, list(columns))
        if isinstance(page, dict) and page.get("nextCursor"):
            table += f"\n\nMore results: --cursor {page['nextCursor']}"
        return table

    return render


def _confirm_delete(args: Any, client: Any, request: Callable[[Optional[str]], Any], what: str) -> Any:
    """Run the two-step deletion protocol, asking for confirmation in between."""
    from roboflow.adapters.workflowevalsapi import WorkflowEvalError
    from roboflow.cli._output import confirm_destructive

    try:
        return request(None)
    except WorkflowEvalError as exc:
        key = exc.details.get("deletionKey") if exc.code == "deletion_confirmation_required" else None
        if not key:
            raise
        impact = exc.details.get("impact")
        if impact and not getattr(args, "yes", False) and not getattr(args, "json", False):
            print(f"Deletion impact: {json.dumps(impact, default=str)}", file=sys.stderr)
        if not confirm_destructive(args, f"Delete {what}? This cannot be undone."):
            sys.exit(0)  # confirm_destructive already reported the cancellation
        return request(key)


_EVAL_COLUMNS = {
    "ID": ("id",),
    "NAME": ("name",),
    "STATE": ("state",),
    "RUNS": ("runCount",),
    "LAST RUN": ("lastRunAt",),
}
_RUN_COLUMNS = {
    "ID": ("id",),
    "STATE": ("state",),
    "MODE": ("executionMode",),
    "CASES": ("selectedCaseCount",),
    "EXECUTIONS": ("executionCount",),
    "CREATED": ("createdAt",),
}
_NAMED_COLUMNS = {"ID": ("id",), "NAME": ("name",), "REVISION": ("revision",), "UPDATED": ("updatedAt",)}


# ---------------------------------------------------------------------------
# Evals
# ---------------------------------------------------------------------------


@workflow_eval_app.command("list")
def list_evals(
    ctx: typer.Context,
    limit: LimitOpt = None,
    cursor: CursorOpt = None,
    search: Annotated[Optional[str], typer.Option("--search", help="Filter by name")] = None,
    spec_summary: Annotated[bool, typer.Option("--spec-summary", help="Include a summary of each Spec")] = False,
) -> None:
    """List Evals in the workspace (newest first)."""
    args = ctx_to_args(ctx)
    _invoke(
        args,
        lambda c: c.list(limit=limit, cursor=cursor, search=search, include_spec_summary=spec_summary),
        _page_text(_EVAL_COLUMNS),
    )


@workflow_eval_app.command("get")
def get_eval(
    ctx: typer.Context,
    eval_id: Annotated[str, typer.Argument(help="Eval id (or readable URL with --by-url)")],
    by_url: Annotated[bool, typer.Option("--by-url", help="Treat the argument as the Eval's readable URL")] = False,
) -> None:
    """Show an Eval with its setup state, diagnostics and recent Runs."""
    args = ctx_to_args(ctx)
    _invoke(args, lambda c: c.get_by_url(eval_id) if by_url else c.get(eval_id))


@workflow_eval_app.command("create")
def create_eval(
    ctx: typer.Context,
    name: Annotated[Optional[str], typer.Option("--name", help="Eval name")] = None,
    description: Annotated[Optional[str], typer.Option("--description", help="Eval description")] = None,
    spec: Annotated[Optional[str], typer.Option("--spec", "-s", help="Spec id")] = None,
    dataset: Annotated[Optional[str], typer.Option("--dataset", "-d", help="Eval Dataset id")] = None,
    body: BodyOpt = None,
    idempotency_key: IdempotencyOpt = None,
) -> None:
    """Create an Eval (an empty draft when no options are given)."""
    args = ctx_to_args(ctx)
    payload = _merge(_load_object(args, body), name=name, description=description, specId=spec, evalDatasetId=dataset)
    _invoke(args, lambda c: c.create(payload, idempotency_key=idempotency_key))


@workflow_eval_app.command("update")
def update_eval(
    ctx: typer.Context,
    eval_id: Annotated[str, typer.Argument(help="Eval id")],
    revision: RevisionOpt,
    name: Annotated[Optional[str], typer.Option("--name", help="New name")] = None,
    description: Annotated[Optional[str], typer.Option("--description", help="New description")] = None,
    spec: Annotated[Optional[str], typer.Option("--spec", "-s", help="Attach this Spec id")] = None,
    dataset: Annotated[Optional[str], typer.Option("--dataset", "-d", help="Attach this Eval Dataset id")] = None,
    body: BodyOpt = None,
) -> None:
    """Update an Eval's metadata, Spec/Dataset, subjects, bindings or server configuration."""
    args = ctx_to_args(ctx)
    payload = _require(
        args,
        _merge(_load_object(args, body), name=name, description=description, specId=spec, evalDatasetId=dataset),
        "--body or a field option",
    )
    _invoke(args, lambda c: c.update(eval_id, payload, revision=revision))


@workflow_eval_app.command("delete")
def delete_eval(
    ctx: typer.Context,
    eval_id: Annotated[str, typer.Argument(help="Eval id")],
    yes: YesOpt = False,
) -> None:
    """Delete an Eval with its Runs, results and exclusively owned artifacts."""
    args = ctx_to_args(ctx, yes=yes)
    _invoke(
        args,
        lambda c: _confirm_delete(args, c, lambda key: c.delete(eval_id, deletion_key=key), f"Eval {eval_id}"),
        interactive=True,
    )


@workflow_eval_app.command("capabilities")
def capabilities(ctx: typer.Context) -> None:
    """List the public operations and what this credential may do."""
    args = ctx_to_args(ctx)
    _invoke(args, lambda c: c.capabilities())


@workflow_eval_app.command("openapi")
def openapi(ctx: typer.Context) -> None:
    """Print the Workflow Evals OpenAPI 3.1 contract."""
    args = ctx_to_args(ctx)
    _invoke(args, lambda c: c.openapi())


@workflow_eval_app.command("operation")
def operation(
    ctx: typer.Context,
    operation_id: Annotated[str, typer.Argument(help="Operation id from a deletion response")],
) -> None:
    """Show the status of a deletion operation."""
    args = ctx_to_args(ctx)
    _invoke(args, lambda c: c.get_operation(operation_id))


@workflow_eval_app.command("compare")
def compare(
    ctx: typer.Context,
    eval_id: EvalOpt,
    execution: Annotated[
        Optional[List[str]], typer.Option("--execution", "-x", help="Completed Execution id (repeat 2+ times)")
    ] = None,
    body: BodyOpt = None,
) -> None:
    """Compare two or more completed Executions (per-check and per-Case deltas)."""
    args = ctx_to_args(ctx)
    payload = _require(args, _merge(_load_object(args, body), executionIds=execution or None), "--execution")
    _invoke(args, lambda c: c.compare(eval_id, payload))


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------


@spec_app.command("list")
def list_specs(ctx: typer.Context, limit: LimitOpt = None, cursor: CursorOpt = None) -> None:
    """List Specs in the workspace."""
    args = ctx_to_args(ctx)
    _invoke(args, lambda c: c.list_specs(limit=limit, cursor=cursor), _page_text(_NAMED_COLUMNS))


@spec_app.command("get")
def get_spec(ctx: typer.Context, spec_id: Annotated[str, typer.Argument(help="Spec id")]) -> None:
    """Show the current live Spec."""
    args = ctx_to_args(ctx)
    _invoke(args, lambda c: c.get_spec(spec_id))


@spec_app.command("create")
def create_spec(
    ctx: typer.Context,
    body: Annotated[
        str, typer.Option("--body", "-b", help="JSON {name?, description?, spec}: inline, @file.json, path, or -")
    ],
    name: Annotated[Optional[str], typer.Option("--name", help="Spec name")] = None,
    description: Annotated[Optional[str], typer.Option("--description", help="Spec description")] = None,
    idempotency_key: IdempotencyOpt = None,
) -> None:
    """Validate, compile and save a Spec."""
    args = ctx_to_args(ctx)
    payload = _merge(_require(args, _load_object(args, body)), name=name, description=description)
    _invoke(args, lambda c: c.create_spec(payload, idempotency_key=idempotency_key))


@spec_app.command("update")
def update_spec(
    ctx: typer.Context,
    spec_id: Annotated[str, typer.Argument(help="Spec id")],
    revision: RevisionOpt,
    name: Annotated[Optional[str], typer.Option("--name", help="New name")] = None,
    description: Annotated[Optional[str], typer.Option("--description", help="New description")] = None,
    body: BodyOpt = None,
) -> None:
    """Replace the live Spec's policy and/or metadata."""
    args = ctx_to_args(ctx)
    payload = _require(
        args, _merge(_load_object(args, body), name=name, description=description), "--body or a field option"
    )
    _invoke(args, lambda c: c.update_spec(spec_id, payload, revision=revision))


@spec_app.command("validate")
def validate_spec(
    ctx: typer.Context,
    body: Annotated[str, typer.Option("--body", "-b", help="Spec JSON (or {spec}): inline, @file.json, path, or -")],
) -> None:
    """Validate an unsaved Spec and show diagnostics and ground-truth requirements."""
    args = ctx_to_args(ctx)
    spec = _require(args, _load_object(args, body))
    _invoke(args, lambda c: c.validate_spec(spec))


@spec_app.command("compatible-datasets")
def spec_compatible_datasets(
    ctx: typer.Context,
    spec_id: Annotated[str, typer.Argument(help="Spec id")],
    limit: LimitOpt = None,
    cursor: CursorOpt = None,
) -> None:
    """List Eval Datasets that satisfy this Spec."""
    args = ctx_to_args(ctx)
    _invoke(
        args,
        lambda c: c.list_compatible_datasets(spec_id, limit=limit, cursor=cursor),
        _page_text(_NAMED_COLUMNS),
    )


@spec_app.command("ai-draft")
def ai_draft(
    ctx: typer.Context,
    eval_id: EvalOpt,
    instructions: Annotated[
        Optional[str], typer.Option("--instructions", "-i", help="Evaluation intent or requested improvements")
    ] = None,
    idempotency_key: IdempotencyOpt = None,
) -> None:
    """Draft a Spec and bindings with AI for an Eval's Subject (consumes AI credits; not saved)."""
    args = ctx_to_args(ctx)
    show_progress = not args.json and not args.quiet

    def on_event(event: Dict[str, Any]) -> None:
        if not show_progress:
            return
        data = event.get("data")
        if event["event"] == "status" and isinstance(data, dict) and data.get("message"):
            print(f"… {data['message']}", file=sys.stderr)

    _invoke(args, lambda c: c.ai_draft(eval_id, instructions, idempotency_key=idempotency_key, on_event=on_event))


# ---------------------------------------------------------------------------
# Eval Datasets
# ---------------------------------------------------------------------------


@dataset_app.command("list")
def list_datasets(ctx: typer.Context, limit: LimitOpt = None, cursor: CursorOpt = None) -> None:
    """List Eval Datasets in the workspace."""
    args = ctx_to_args(ctx)
    columns = {**_NAMED_COLUMNS, "CASES": ("caseCount",)}
    _invoke(args, lambda c: c.list_datasets(limit=limit, cursor=cursor), _page_text(columns))


@dataset_app.command("get")
def get_dataset(ctx: typer.Context, dataset_id: Annotated[str, typer.Argument(help="Eval Dataset id")]) -> None:
    """Show an Eval Dataset and its derived summary."""
    args = ctx_to_args(ctx)
    _invoke(args, lambda c: c.get_dataset(dataset_id))


@dataset_app.command("create")
def create_dataset(
    ctx: typer.Context,
    body: Annotated[
        str,
        typer.Option(
            "--body", "-b", help="JSON {name, description?, inputs, groundTruthContract}: inline, @file, path, or -"
        ),
    ],
    name: Annotated[Optional[str], typer.Option("--name", help="Dataset name")] = None,
    description: Annotated[Optional[str], typer.Option("--description", help="Dataset description")] = None,
    idempotency_key: IdempotencyOpt = None,
) -> None:
    """Create an empty Eval Dataset with declared input and ground-truth contracts."""
    args = ctx_to_args(ctx)
    payload = _merge(_require(args, _load_object(args, body)), name=name, description=description)
    _invoke(args, lambda c: c.create_dataset(payload, idempotency_key=idempotency_key))


@dataset_app.command("update")
def update_dataset(
    ctx: typer.Context,
    dataset_id: Annotated[str, typer.Argument(help="Eval Dataset id")],
    revision: RevisionOpt,
    name: Annotated[Optional[str], typer.Option("--name", help="New name")] = None,
    description: Annotated[Optional[str], typer.Option("--description", help="New description")] = None,
) -> None:
    """Update an Eval Dataset's name or description."""
    args = ctx_to_args(ctx)
    payload = _require(args, _merge({}, name=name, description=description), "--name or --description")
    _invoke(args, lambda c: c.update_dataset(dataset_id, payload, revision=revision))


@dataset_app.command("update-contract")
def update_dataset_contract(
    ctx: typer.Context,
    dataset_id: Annotated[str, typer.Argument(help="Eval Dataset id")],
    revision: RevisionOpt,
    body: Annotated[
        str, typer.Option("--body", "-b", help="JSON {inputs?, groundTruthContract?}: inline, @file, path, or -")
    ],
) -> None:
    """Replace the declared input and/or ground-truth contracts (may rewrite Cases in the background)."""
    args = ctx_to_args(ctx)
    payload = _require(args, _load_object(args, body))
    _invoke(args, lambda c: c.mutate_dataset_contract(dataset_id, payload, revision=revision))


@dataset_app.command("compatible-specs")
def dataset_compatible_specs(
    ctx: typer.Context,
    dataset_id: Annotated[str, typer.Argument(help="Eval Dataset id")],
    limit: LimitOpt = None,
    cursor: CursorOpt = None,
) -> None:
    """List Specs this Eval Dataset satisfies."""
    args = ctx_to_args(ctx)
    _invoke(
        args,
        lambda c: c.list_compatible_specs(dataset_id, limit=limit, cursor=cursor),
        _page_text(_NAMED_COLUMNS),
    )


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


@case_app.command("list")
def list_cases(
    ctx: typer.Context,
    dataset_id: DatasetOpt,
    limit: LimitOpt = None,
    cursor: CursorOpt = None,
    slice_: SliceOpt = None,
    slice_match: SliceMatchOpt = None,
    ground_truth: Annotated[
        Optional[str], typer.Option("--ground-truth", help="Filter by ground truth: complete or incomplete")
    ] = None,
) -> None:
    """List Cases in an Eval Dataset."""
    args = ctx_to_args(ctx)
    columns = {"ID": ("id",), "NAME": ("name",), "GROUND TRUTH": ("groundTruthState",), "SLICES": ("slices",)}
    _invoke(
        args,
        lambda c: c.list_cases(
            dataset_id,
            limit=limit,
            cursor=cursor,
            slices=slice_,
            slice_match=slice_match,
            ground_truth_completeness=ground_truth,
        ),
        _page_text(columns),
    )


@case_app.command("get")
def get_case(
    ctx: typer.Context, case_id: Annotated[str, typer.Argument(help="Case id")], dataset_id: DatasetOpt
) -> None:
    """Show one Case with short-lived read URLs for its assets."""
    args = ctx_to_args(ctx)
    _invoke(args, lambda c: c.get_case(dataset_id, case_id))


@case_app.command("add")
def add_case(
    ctx: typer.Context,
    dataset_id: DatasetOpt,
    body: Annotated[
        str,
        typer.Option("--body", "-b", help="JSON {name, inputs, groundTruth, slices?, ...}: inline, @file, path, or -"),
    ],
    name: Annotated[Optional[str], typer.Option("--name", help="Case name")] = None,
    idempotency_key: IdempotencyOpt = None,
) -> None:
    """Add one Case to an Eval Dataset."""
    args = ctx_to_args(ctx)
    payload = _merge(_require(args, _load_object(args, body)), name=name)
    _invoke(args, lambda c: c.add_case(dataset_id, payload, idempotency_key=idempotency_key))


@case_app.command("replace")
def replace_case(
    ctx: typer.Context,
    case_id: Annotated[str, typer.Argument(help="Case id")],
    dataset_id: DatasetOpt,
    revision: RevisionOpt,
    body: Annotated[
        str, typer.Option("--body", "-b", help="JSON {inputs, groundTruth, name?, slices?}: inline, @file, path, or -")
    ],
) -> None:
    """Replace one Case atomically."""
    args = ctx_to_args(ctx)
    payload = _require(args, _load_object(args, body))
    _invoke(args, lambda c: c.replace_case(dataset_id, case_id, payload, revision=revision))


@case_app.command("delete")
def delete_case(
    ctx: typer.Context,
    case_id: Annotated[str, typer.Argument(help="Case id")],
    dataset_id: DatasetOpt,
    revision: RevisionOpt,
    yes: YesOpt = False,
) -> None:
    """Remove a Case from the live Dataset (Runs that froze it are unaffected)."""
    from roboflow.cli._output import confirm_destructive

    args = ctx_to_args(ctx, yes=yes)
    if not confirm_destructive(args, f"Delete Case {case_id}?"):
        return

    def call(c: Any) -> Dict[str, Any]:
        c.delete_case(dataset_id, case_id, revision=revision)
        return {"deleted": True, "caseId": case_id}

    _invoke(args, call, lambda _: f"Deleted Case {case_id}.")


@case_app.command("import")
def import_cases(
    ctx: typer.Context,
    dataset_id: DatasetOpt,
    input_field: Annotated[
        Optional[str], typer.Option("--input-field", help="Dataset image input that receives each item")
    ] = None,
    source: Annotated[
        Optional[List[str]], typer.Option("--source", help="Platform image Source id (repeatable)")
    ] = None,
    from_dataset: Annotated[
        Optional[List[str]],
        typer.Option("--from-dataset", help="Platform Dataset id, optionally id:train|valid|test (repeatable)"),
    ] = None,
    artifact: Annotated[
        Optional[List[str]], typer.Option("--artifact", help="Uploaded Case asset artifact id (repeatable)")
    ] = None,
    slice_: Annotated[Optional[List[str]], typer.Option("--slice", help="Slice for every imported Case")] = None,
    body: BodyOpt = None,
    idempotency_key: IdempotencyOpt = None,
) -> None:
    """Import images as incomplete Cases in the background."""
    args = ctx_to_args(ctx)
    items: List[Dict[str, Any]] = [{"kind": "source", "sourceId": value} for value in source or []]
    for value in from_dataset or []:
        dataset, _, split = value.partition(":")
        items.append({"kind": "dataset", "datasetId": dataset, **({"split": split} if split else {})})
    items += [{"kind": "artifact", "artifactId": value} for value in artifact or []]
    payload = _merge(_load_object(args, body), inputField=input_field, slices=slice_ or None)
    if items:
        payload["items"] = payload.get("items", []) + items
    payload = _require(args, payload, "--source/--from-dataset/--artifact or --body")
    _invoke(args, lambda c: c.import_cases(dataset_id, payload, idempotency_key=idempotency_key))


@case_app.command("import-status")
def import_status(
    ctx: typer.Context,
    intake_id: Annotated[str, typer.Argument(help="Intake id from `case import`")],
    dataset_id: DatasetOpt,
) -> None:
    """Show a Case import's status and progress."""
    args = ctx_to_args(ctx)
    _invoke(args, lambda c: c.get_case_import(dataset_id, intake_id))


@case_app.command("upload")
def upload_case_asset(
    ctx: typer.Context,
    path: Annotated[str, typer.Argument(help="Local image file (JPEG, PNG, WebP or GIF)")],
    dataset_id: DatasetOpt,
    content_type: Annotated[
        Optional[str], typer.Option("--content-type", help="MIME type (default: inferred from the extension)")
    ] = None,
    role: Annotated[Optional[str], typer.Option("--role", help="Optional asset role")] = None,
) -> None:
    """Upload an image as a Case asset and print its artifactId."""
    args = ctx_to_args(ctx)
    if not os.path.isfile(path):
        from roboflow.cli._output import output_error

        output_error(args, f"File not found: {path}", exit_code=3)
    _invoke(
        args,
        lambda c: c.upload_case_asset(dataset_id, path, content_type=content_type, role=role),
        lambda asset: f"Uploaded {path} → artifactId {asset.get('artifactId')}",
    )


# ---------------------------------------------------------------------------
# Bindings
# ---------------------------------------------------------------------------

_SubjectOpt = Annotated[Optional[str], typer.Option("--subject", help="WorkflowSubject JSON (overrides --workflow)")]
_WorkflowOpt = Annotated[Optional[str], typer.Option("--workflow", help="Saved Workflow id")]
_WorkflowVersionOpt = Annotated[Optional[str], typer.Option("--workflow-version", help="Saved WorkflowVersion id")]
_SubjectKeyOpt = Annotated[Optional[str], typer.Option("--subject-key", help="Subject key (default: workflow id)")]


@binding_app.command("suggest")
def suggest_bindings(
    ctx: typer.Context,
    spec_id: Annotated[str, typer.Option("--spec", "-s", help="Spec id")],
    dataset_id: DatasetOpt,
    workflow: _WorkflowOpt = None,
    workflow_version: _WorkflowVersionOpt = None,
    subject_key: _SubjectKeyOpt = None,
    subject: _SubjectOpt = None,
) -> None:
    """Suggest a BindingSet that maps a Workflow to the Spec and Dataset."""
    args = ctx_to_args(ctx)
    resolved = _subject(args, subject, workflow, workflow_version, subject_key)
    _invoke(args, lambda c: c.suggest_bindings(spec_id, dataset_id, resolved))


@binding_app.command("validate")
def validate_bindings(
    ctx: typer.Context,
    spec_id: Annotated[str, typer.Option("--spec", "-s", help="Spec id")],
    dataset_id: DatasetOpt,
    binding_set: Annotated[str, typer.Option("--binding-set", help="BindingSet JSON: inline, @file.json, path, or -")],
    workflow: _WorkflowOpt = None,
    workflow_version: _WorkflowVersionOpt = None,
    subject_key: _SubjectKeyOpt = None,
    subject: _SubjectOpt = None,
) -> None:
    """Validate a BindingSet against the Spec, Dataset and Workflow."""
    args = ctx_to_args(ctx)
    resolved = _subject(args, subject, workflow, workflow_version, subject_key)
    bindings = _require(args, _load_object(args, binding_set, "--binding-set"), "--binding-set")
    _invoke(args, lambda c: c.validate_bindings(spec_id, dataset_id, resolved, bindings))


# ---------------------------------------------------------------------------
# Case preparations
# ---------------------------------------------------------------------------


@preparation_app.command("start")
def start_preparation(
    ctx: typer.Context,
    eval_id: EvalOpt,
    runtime: Annotated[str, typer.Option("--runtime", help="serverless or dedicated")] = "serverless",
    url: Annotated[Optional[str], typer.Option("--url", help="Dedicated deployment base URL")] = None,
    case: Annotated[
        Optional[List[str]], typer.Option("--case", help="Prepare only these Case ids (repeatable)")
    ] = None,
    case_import: Annotated[
        Optional[str], typer.Option("--case-import", help="Follow a running Case import's async task id")
    ] = None,
    body: BodyOpt = None,
    idempotency_key: IdempotencyOpt = None,
) -> None:
    """Run the Eval's Workflow over Cases to produce candidate outputs for labeling."""
    args = ctx_to_args(ctx)
    payload = _load_object(args, body)
    payload.setdefault("runtime", _merge({"kind": runtime}, url=url))
    if case:
        payload["caseSelection"] = {"kind": "explicit", "caseIds": case}
    payload = _merge(payload, caseImportId=case_import)
    _invoke(args, lambda c: c.start_case_preparation(eval_id, payload, idempotency_key=idempotency_key))


@preparation_app.command("current")
def current_preparation(ctx: typer.Context, eval_id: EvalOpt) -> None:
    """Show the latest preparation for the Eval's current setup."""
    args = ctx_to_args(ctx)
    _invoke(
        args,
        lambda c: c.get_current_case_preparation(eval_id),
        lambda result: None if result else "No current case preparation.",
    )


@preparation_app.command("get")
def get_preparation(
    ctx: typer.Context, preparation_id: Annotated[str, typer.Argument(help="Preparation id")], eval_id: EvalOpt
) -> None:
    """Show a preparation's state and progress."""
    args = ctx_to_args(ctx)
    _invoke(args, lambda c: c.get_case_preparation(eval_id, preparation_id))


@preparation_app.command("resume")
def resume_preparation(
    ctx: typer.Context, preparation_id: Annotated[str, typer.Argument(help="Preparation id")], eval_id: EvalOpt
) -> None:
    """Resume a preparation, requeueing only failed Cases."""
    args = ctx_to_args(ctx)
    _invoke(args, lambda c: c.resume_case_preparation(eval_id, preparation_id))


@preparation_app.command("cases")
def preparation_cases(
    ctx: typer.Context,
    preparation_id: Annotated[str, typer.Argument(help="Preparation id")],
    eval_id: EvalOpt,
    limit: LimitOpt = None,
    cursor: CursorOpt = None,
) -> None:
    """List per-Case candidate outputs of a preparation."""
    args = ctx_to_args(ctx)
    _invoke(args, lambda c: c.list_case_preparation_cases(eval_id, preparation_id, limit=limit, cursor=cursor))


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

_WaitOpt = Annotated[bool, typer.Option("--wait", help="Poll until the Run reaches a terminal state")]
_TimeoutOpt = Annotated[float, typer.Option("--timeout", help="Seconds to wait before giving up")]
_IntervalOpt = Annotated[float, typer.Option("--interval", help="Seconds between polls")]


def _run_id(admission: Dict[str, Any]) -> Optional[str]:
    run = admission.get("run")
    return run.get("id") if isinstance(run, dict) else admission.get("id")


def _then_wait(args: Any, eval_id: str, timeout: float, interval: float) -> Callable[[Any, Dict[str, Any]], Any]:
    def wait(client: Any, admission: Dict[str, Any]) -> Any:
        run_id = _run_id(admission)
        if not run_id:
            return admission
        if not args.json and not args.quiet:
            print(f"Waiting for Run {run_id}…", file=sys.stderr)
        return client.wait_for_run(eval_id, run_id, timeout=timeout, interval=interval)

    return wait


@run_app.command("list")
def list_runs(
    ctx: typer.Context,
    eval_id: EvalOpt,
    limit: LimitOpt = None,
    cursor: CursorOpt = None,
    check_count: Annotated[bool, typer.Option("--check-count", help="Include each Run's frozen Check count")] = False,
) -> None:
    """List an Eval's Runs (newest first)."""
    args = ctx_to_args(ctx)
    _invoke(
        args,
        lambda c: c.list_runs(eval_id, limit=limit, cursor=cursor, include_check_count=check_count),
        _page_text(_RUN_COLUMNS),
    )


@run_app.command("get")
def get_run(
    ctx: typer.Context,
    run_id: Annotated[str, typer.Argument(help="Run id")],
    eval_id: EvalOpt,
    wait: _WaitOpt = False,
    timeout: _TimeoutOpt = 3600,
    interval: _IntervalOpt = 5,
) -> None:
    """Show a Run's lifecycle, frozen setup and result availability."""
    args = ctx_to_args(ctx)
    if wait:
        _invoke(args, lambda c: c.wait_for_run(eval_id, run_id, timeout=timeout, interval=interval))
    else:
        _invoke(args, lambda c: c.get_run(eval_id, run_id))


@run_app.command("start")
def start_run(
    ctx: typer.Context,
    eval_id: EvalOpt,
    body: Annotated[
        str,
        typer.Option(
            "--body", "-b", help="JSON {executions: [{subject, bindingSet}], caseSelection?, ...}: inline, @file, or -"
        ),
    ],
    wait: _WaitOpt = False,
    timeout: _TimeoutOpt = 3600,
    interval: _IntervalOpt = 5,
    idempotency_key: IdempotencyOpt = None,
) -> None:
    """Start a grouped Run of one or more Workflows over the Eval's Dataset (may consume credits)."""
    args = ctx_to_args(ctx)
    payload = _require(args, _load_object(args, body))
    then = _then_wait(args, eval_id, timeout, interval)

    def call(c: Any) -> Any:
        admission = c.start_run(eval_id, payload, idempotency_key=idempotency_key)
        return then(c, admission) if wait else admission

    _invoke(args, call)


@run_app.command("cancel")
def cancel_run(ctx: typer.Context, run_id: Annotated[str, typer.Argument(help="Run id")], eval_id: EvalOpt) -> None:
    """Request cancellation of a queued or running Run."""
    args = ctx_to_args(ctx)
    _invoke(args, lambda c: c.cancel_run(eval_id, run_id))


@run_app.command("retry")
def retry_run(
    ctx: typer.Context,
    run_id: Annotated[str, typer.Argument(help="Run id")],
    eval_id: EvalOpt,
    idempotency_key: IdempotencyOpt = None,
) -> None:
    """Retry missing or operationally failed Cases in the same Run."""
    args = ctx_to_args(ctx)
    _invoke(args, lambda c: c.retry_run(eval_id, run_id, idempotency_key=idempotency_key))


@run_app.command("replay")
def replay_run(
    ctx: typer.Context,
    run_id: Annotated[str, typer.Argument(help="Source Run id")],
    eval_id: EvalOpt,
    body: BodyOpt = None,
    idempotency_key: IdempotencyOpt = None,
) -> None:
    """Create a new Run from a previous one, optionally replacing Subjects and bindings."""
    args = ctx_to_args(ctx)
    payload = _load_object(args, body)
    _invoke(args, lambda c: c.replay_run(eval_id, run_id, payload, idempotency_key=idempotency_key))


@run_app.command("rescore")
def rescore_run(
    ctx: typer.Context,
    run_id: Annotated[str, typer.Argument(help="Source Run id")],
    eval_id: EvalOpt,
    body: Annotated[
        str, typer.Option("--body", "-b", help="JSON {specId, executions: [...]}: inline, @file, path, or -")
    ],
    idempotency_key: IdempotencyOpt = None,
) -> None:
    """Rescore a Run's retained captures with the Eval's current saved Spec."""
    args = ctx_to_args(ctx)
    payload = _require(args, _load_object(args, body))
    _invoke(args, lambda c: c.rescore_run(eval_id, run_id, payload, idempotency_key=idempotency_key))


@run_app.command("delete")
def delete_run(
    ctx: typer.Context,
    run_id: Annotated[str, typer.Argument(help="Run id")],
    eval_id: EvalOpt,
    yes: YesOpt = False,
) -> None:
    """Delete a Run with its Executions, results and artifacts."""
    args = ctx_to_args(ctx, yes=yes)
    _invoke(
        args,
        lambda c: _confirm_delete(
            args, c, lambda key: c.delete_run(eval_id, run_id, deletion_key=key), f"Run {run_id}"
        ),
        interactive=True,
    )


@run_app.command("config")
def run_config(ctx: typer.Context, run_id: Annotated[str, typer.Argument(help="Run id")], eval_id: EvalOpt) -> None:
    """Show a Run's frozen shared configuration."""
    args = ctx_to_args(ctx)
    _invoke(args, lambda c: c.get_run_configuration(eval_id, run_id))


# ---------------------------------------------------------------------------
# Executions and results
# ---------------------------------------------------------------------------

_ExecutionArg = Annotated[str, typer.Argument(help="Execution id")]


@execution_app.command("list")
def list_executions(ctx: typer.Context, eval_id: EvalOpt, run_id: RunOpt) -> None:
    """List the Executions of a Run."""
    args = ctx_to_args(ctx)
    columns = {"ID": ("id",), "SUBJECT": ("subjectKey", "name"), "STATE": ("state",)}
    _invoke(args, lambda c: c.list_executions(eval_id, run_id), _page_text(columns))


@execution_app.command("get")
def get_execution(ctx: typer.Context, execution_id: _ExecutionArg, eval_id: EvalOpt, run_id: RunOpt) -> None:
    """Show one Execution's lifecycle and result availability."""
    args = ctx_to_args(ctx)
    _invoke(args, lambda c: c.get_execution(eval_id, run_id, execution_id))


@execution_app.command("config")
def execution_config(ctx: typer.Context, execution_id: _ExecutionArg, eval_id: EvalOpt, run_id: RunOpt) -> None:
    """Show an Execution's frozen Subject, BindingSet and linked plan."""
    args = ctx_to_args(ctx)
    _invoke(args, lambda c: c.get_execution_configuration(eval_id, run_id, execution_id))


@execution_app.command("input-cases")
def execution_input_cases(
    ctx: typer.Context,
    execution_id: _ExecutionArg,
    eval_id: EvalOpt,
    run_id: RunOpt,
    limit: LimitOpt = None,
    cursor: CursorOpt = None,
) -> None:
    """List frozen Case inputs and checkpoints for a locally executed Run."""
    args = ctx_to_args(ctx)
    _invoke(args, lambda c: c.list_execution_input_cases(eval_id, run_id, execution_id, limit=limit, cursor=cursor))


@execution_app.command("capture")
def submit_capture(
    ctx: typer.Context,
    execution_id: _ExecutionArg,
    eval_id: EvalOpt,
    run_id: RunOpt,
    case_id: Annotated[str, typer.Option("--case", "-c", help="Case id")],
    body: Annotated[
        str,
        typer.Option(
            "--body", "-b", help="JSON {attemptCount, shardIndex, output, runtime?}: inline, @file, path, or -"
        ),
    ],
) -> None:
    """Submit a locally captured Workflow output for server-side scoring."""
    args = ctx_to_args(ctx)
    payload = _require(args, _load_object(args, body))
    _invoke(args, lambda c: c.submit_capture(eval_id, run_id, execution_id, case_id, payload))


@execution_app.command("overview")
def results_overview(
    ctx: typer.Context,
    execution_id: _ExecutionArg,
    eval_id: EvalOpt,
    run_id: RunOpt,
    slice_: SliceOpt = None,
    slice_match: SliceMatchOpt = None,
) -> None:
    """Show aggregate Check performance and runtime for an Execution."""
    args = ctx_to_args(ctx)
    _invoke(
        args, lambda c: c.get_results_overview(eval_id, run_id, execution_id, slices=slice_, slice_match=slice_match)
    )


@execution_app.command("results")
def list_case_results(
    ctx: typer.Context,
    execution_id: _ExecutionArg,
    eval_id: EvalOpt,
    run_id: RunOpt,
    limit: LimitOpt = None,
    cursor: CursorOpt = None,
    slice_: SliceOpt = None,
    slice_match: SliceMatchOpt = None,
    state: Annotated[
        Optional[str], typer.Option("--state", help="running, completed, failed, cancelled or skipped")
    ] = None,
    case_id: Annotated[Optional[str], typer.Option("--case", "-c", help="Only this Case id")] = None,
    check_id: Annotated[Optional[str], typer.Option("--check", help="Filter by Check id")] = None,
    judgment: Annotated[Optional[str], typer.Option("--judgment", help="Filter by judgment (pass/fail)")] = None,
    failed_check: Annotated[
        Optional[bool], typer.Option("--failed-check/--no-failed-check", help="Only Cases with a failed Check")
    ] = None,
    order: Annotated[Optional[str], typer.Option("--order", help="Sort by case or latency")] = None,
) -> None:
    """List per-Case results of an Execution."""
    args = ctx_to_args(ctx)
    columns = {"CASE": ("caseId", "id"), "STATE": ("state",), "LATENCY MS": ("latencyMs",)}
    _invoke(
        args,
        lambda c: c.list_case_results(
            eval_id,
            run_id,
            execution_id,
            limit=limit,
            cursor=cursor,
            slices=slice_,
            slice_match=slice_match,
            order=order,
            state=state,
            case_id=case_id,
            check_id=check_id,
            judgment=judgment,
            failed_check=failed_check,
        ),
        _page_text(columns),
    )


@execution_app.command("result")
def get_case_result(
    ctx: typer.Context,
    execution_id: _ExecutionArg,
    eval_id: EvalOpt,
    run_id: RunOpt,
    case_id: Annotated[str, typer.Option("--case", "-c", help="Case id")],
) -> None:
    """Show one Case's detailed result in an Execution."""
    args = ctx_to_args(ctx)
    _invoke(args, lambda c: c.get_case_result(eval_id, run_id, execution_id, case_id))


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


@export_app.command("start")
def start_export(
    ctx: typer.Context,
    eval_id: EvalOpt,
    run_id: RunOpt,
    export_format: Annotated[str, typer.Option("--format", "-f", help="csv, json or xlsx")] = "json",
    execution: Annotated[
        Optional[List[str]], typer.Option("--execution", "-x", help="Limit to these Execution ids (repeatable)")
    ] = None,
    body: BodyOpt = None,
    wait: Annotated[bool, typer.Option("--wait", help="Poll until the export finishes")] = False,
    timeout: _TimeoutOpt = 1800,
    idempotency_key: IdempotencyOpt = None,
) -> None:
    """Export a stable Run's results."""
    args = ctx_to_args(ctx)
    payload = _merge(_load_object(args, body), format=export_format, executionIds=execution or None)

    def call(c: Any) -> Any:
        admission = c.start_export(eval_id, run_id, payload, idempotency_key=idempotency_key)
        if not wait or not admission.get("asyncTaskId"):
            return admission
        return c.wait_for_export(eval_id, run_id, admission["asyncTaskId"], timeout=timeout)

    _invoke(args, call)


@export_app.command("status")
def export_status(
    ctx: typer.Context,
    task_id: Annotated[str, typer.Argument(help="Export asyncTaskId")],
    eval_id: EvalOpt,
    run_id: RunOpt,
) -> None:
    """Show an export job's state and result."""
    args = ctx_to_args(ctx)
    _invoke(args, lambda c: c.get_export_job(eval_id, run_id, task_id))


@export_app.command("download-url")
def export_download_url(
    ctx: typer.Context,
    artifact_id: Annotated[str, typer.Argument(help="Export artifact id")],
    eval_id: EvalOpt,
    run_id: RunOpt,
) -> None:
    """Mint a fresh short-lived download URL for a completed export."""
    args = ctx_to_args(ctx)
    _invoke(args, lambda c: c.get_export_download(eval_id, run_id, artifact_id))


# ---------------------------------------------------------------------------
# Embedding analyses
# ---------------------------------------------------------------------------


@embedding_app.command("start")
def start_embeddings(
    ctx: typer.Context,
    eval_id: EvalOpt,
    run_id: RunOpt,
    execution_id: ExecutionOpt,
    provider: Annotated[str, typer.Option("--provider", "-p", help="clip, dinov2, dinov3 or siglip2")],
    input_key: Annotated[Optional[str], typer.Option("--input-key", help="Image input to embed")] = None,
    retry: Annotated[bool, typer.Option("--retry", help="Retry a failed or cancelled job")] = False,
    idempotency_key: IdempotencyOpt = None,
) -> None:
    """Generate (or reuse) image embeddings for an Execution's Case inputs."""
    args = ctx_to_args(ctx)
    _invoke(
        args,
        lambda c: c.start_embeddings(
            eval_id,
            run_id,
            execution_id,
            provider,
            input_key=input_key,
            retry=retry or None,
            idempotency_key=idempotency_key,
        ),
    )


@embedding_app.command("get")
def get_embeddings(
    ctx: typer.Context,
    eval_id: EvalOpt,
    run_id: RunOpt,
    execution_id: ExecutionOpt,
    provider: Annotated[Optional[str], typer.Option("--provider", "-p", help="Discover the job for a provider")] = None,
    task_id: Annotated[Optional[str], typer.Option("--task", help="Read a specific job attempt")] = None,
) -> None:
    """Show an embedding analysis job and its projection result."""
    args = ctx_to_args(ctx)
    _invoke(args, lambda c: c.get_embeddings(eval_id, run_id, execution_id, provider=provider, async_task_id=task_id))


@embedding_app.command("cancel")
def cancel_embeddings(
    ctx: typer.Context,
    task_id: Annotated[str, typer.Argument(help="Embedding job asyncTaskId")],
    eval_id: EvalOpt,
    run_id: RunOpt,
    execution_id: ExecutionOpt,
) -> None:
    """Cancel an embedding analysis job."""
    args = ctx_to_args(ctx)
    _invoke(args, lambda c: c.cancel_embeddings(eval_id, run_id, execution_id, task_id))


# ---------------------------------------------------------------------------
# Engine catalog and agent guidance
# ---------------------------------------------------------------------------


@evaluator_app.command("list")
def list_evaluators(
    ctx: typer.Context,
    semantic_type: Annotated[Optional[str], typer.Option("--semantic-type", help="Filter by semantic type")] = None,
    family: Annotated[Optional[str], typer.Option("--family", help="Filter by evaluator family")] = None,
) -> None:
    """List the installed engine's evaluators."""
    args = ctx_to_args(ctx)
    columns = {"TYPE": ("type",), "SEMANTIC TYPE": ("semanticType",), "FAMILY": ("family",)}
    _invoke(args, lambda c: c.list_evaluators(semantic_type=semantic_type, family=family), _page_text(columns))


@evaluator_app.command("get")
def get_evaluator(ctx: typer.Context, evaluator_type: Annotated[str, typer.Argument(help="Evaluator type")]) -> None:
    """Show one evaluator and its configuration schema."""
    args = ctx_to_args(ctx)
    _invoke(args, lambda c: c.get_evaluator(evaluator_type))


@schema_app.command("get")
def get_schema(
    ctx: typer.Context,
    schema_name: Annotated[
        str, typer.Argument(help="spec, eval-dataset, case, binding-set, subject or evaluator-config")
    ],
) -> None:
    """Print an authoring JSON Schema."""
    args = ctx_to_args(ctx)
    _invoke(args, lambda c: c.get_schema(schema_name))


_RESOURCE_COLUMNS = {"ID": ("id",), "PATH": ("path",)}


@agent_app.command("manifest")
def agent_manifest(ctx: typer.Context) -> None:
    """Show the engine's agent manifest (entry skill, skills, bundles, assets)."""
    args = ctx_to_args(ctx)
    _invoke(args, lambda c: c.agent_manifest())


@agent_app.command("skills")
def agent_skills(ctx: typer.Context) -> None:
    """List agent skills."""
    args = ctx_to_args(ctx)
    _invoke(args, lambda c: c.list_agent_skills(), _page_text(_RESOURCE_COLUMNS))


@agent_app.command("skill")
def agent_skill(
    ctx: typer.Context, resource_id: Annotated[str, typer.Argument(help="Skill id, e.g. skill:create-eval")]
) -> None:
    """Print an agent skill's Markdown."""
    args = ctx_to_args(ctx)
    _invoke(args, lambda c: c.get_agent_skill(resource_id), lambda r: r.get("content"))


@agent_app.command("bundles")
def agent_bundles(ctx: typer.Context) -> None:
    """List agent bundles."""
    args = ctx_to_args(ctx)
    _invoke(args, lambda c: c.list_agent_bundles(), _page_text(_RESOURCE_COLUMNS))


@agent_app.command("bundle")
def agent_bundle(
    ctx: typer.Context, resource_id: Annotated[str, typer.Argument(help="Bundle id, e.g. bundle:create-eval")]
) -> None:
    """Show a parsed agent bundle."""
    args = ctx_to_args(ctx)
    _invoke(args, lambda c: c.get_agent_bundle(resource_id))


@agent_app.command("asset")
def agent_asset(
    ctx: typer.Context, resource_id: Annotated[str, typer.Argument(help="Asset id from a bundle's assets")]
) -> None:
    """Print a registered bundle asset's original content."""
    args = ctx_to_args(ctx)
    _invoke(args, lambda c: c.get_agent_asset(resource_id), lambda r: r.get("content"))
