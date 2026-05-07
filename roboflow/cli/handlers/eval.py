"""Model evaluation commands.

Wraps the public ``/{workspace}/model-evals`` REST surface — list runs in a
workspace and pull each panel (mAP, confidence sweep, per-class table,
confusion matrix, vector clusters, per-image stats, recommendations).

The eval-id is opaque (the human in the UI navigates by URL); commands take
it as a positional argument so it composes well with ``--json | jq``.
"""

from __future__ import annotations

from typing import Annotated, Optional

import typer

from roboflow.cli._compat import SortedGroup, ctx_to_args

eval_app = typer.Typer(cls=SortedGroup, help="Inspect model evaluation runs", no_args_is_help=True)


# ---------------------------------------------------------------------------
# Command surface (Typer)
# ---------------------------------------------------------------------------


@eval_app.command("list")
def list_evals_cmd(
    ctx: typer.Context,
    project: Annotated[Optional[str], typer.Option("-p", "--project", help="Filter by project slug or id")] = None,
    version: Annotated[Optional[str], typer.Option("-v", "--version", help="Filter by version id")] = None,
    model: Annotated[Optional[str], typer.Option("-m", "--model", help="Filter by model id")] = None,
    status: Annotated[
        Optional[str], typer.Option("-s", "--status", help="Filter by status (running/done/failed)")
    ] = None,
    limit: Annotated[Optional[int], typer.Option("-n", "--limit", help="Max results (default 50, max 200)")] = None,
) -> None:
    """List model evaluations in the workspace."""
    args = ctx_to_args(ctx, project=project, version=version, model=model, status=status, limit=limit)
    _list_evals(args)


@eval_app.command("get")
def get_eval_cmd(
    ctx: typer.Context,
    eval_id: Annotated[str, typer.Argument(help="Eval id (from `roboflow eval list`)")],
) -> None:
    """Show a single eval's metadata and summary metrics."""
    args = ctx_to_args(ctx, eval_id=eval_id)
    _get_eval(args)


@eval_app.command("map-results")
def map_results_cmd(
    ctx: typer.Context,
    eval_id: Annotated[str, typer.Argument(help="Eval id")],
) -> None:
    """Show per-split mAP results (mAP50, mAP50-95, mAP75, by object size, per class)."""
    args = ctx_to_args(ctx, eval_id=eval_id)
    _map_results(args)


@eval_app.command("confidence-sweep")
def confidence_sweep_cmd(
    ctx: typer.Context,
    eval_id: Annotated[str, typer.Argument(help="Eval id")],
) -> None:
    """Show the confidence-threshold sweep (precision/recall/F1) for the test split."""
    args = ctx_to_args(ctx, eval_id=eval_id)
    _confidence_sweep(args)


@eval_app.command("performance-by-class")
def performance_by_class_cmd(
    ctx: typer.Context,
    eval_id: Annotated[str, typer.Argument(help="Eval id")],
    split: Annotated[
        Optional[str],
        typer.Option("-s", "--split", help="Split: train, valid, or test (default test). 'all' is rejected."),
    ] = None,
) -> None:
    """Show per-class precision / recall / F1 / mAP for the chosen split."""
    args = ctx_to_args(ctx, eval_id=eval_id, split=split)
    _performance_by_class(args)


@eval_app.command("confusion-matrix")
def confusion_matrix_cmd(
    ctx: typer.Context,
    eval_id: Annotated[str, typer.Argument(help="Eval id")],
    split: Annotated[
        Optional[str], typer.Option("-s", "--split", help="Split: train, valid, test, or all (default test)")
    ] = None,
    confidence: Annotated[
        Optional[int],
        typer.Option("-c", "--confidence", help="Integer confidence threshold (0-100)"),
    ] = None,
) -> None:
    """Show the confusion matrix for *split* at *confidence*."""
    args = ctx_to_args(ctx, eval_id=eval_id, split=split, confidence=confidence)
    _confusion_matrix(args)


@eval_app.command("vector-analysis")
def vector_analysis_cmd(
    ctx: typer.Context,
    eval_id: Annotated[str, typer.Argument(help="Eval id")],
    confidence: Annotated[
        Optional[int],
        typer.Option("-c", "--confidence", help="Integer confidence threshold (0-100)"),
    ] = None,
) -> None:
    """Show embedding-cluster diagnostics (per-cluster sample images + metrics)."""
    args = ctx_to_args(ctx, eval_id=eval_id, confidence=confidence)
    _vector_analysis(args)


@eval_app.command("image-predictions")
def image_predictions_cmd(
    ctx: typer.Context,
    eval_id: Annotated[str, typer.Argument(help="Eval id")],
    split: Annotated[
        Optional[str], typer.Option("-s", "--split", help="Split: train, valid, test, or all (default test)")
    ] = None,
    confidence: Annotated[
        Optional[int],
        typer.Option("-c", "--confidence", help="Integer confidence threshold (0-100)"),
    ] = None,
    limit: Annotated[
        Optional[int],
        typer.Option("-n", "--limit", help="Page size (default 200, max 1000)"),
    ] = None,
    offset: Annotated[
        Optional[int],
        typer.Option("-o", "--offset", help="Pagination offset"),
    ] = None,
) -> None:
    """Show paginated per-image stats (TP/FP/FN, augmentations, cluster id)."""
    args = ctx_to_args(ctx, eval_id=eval_id, split=split, confidence=confidence, limit=limit, offset=offset)
    _image_predictions(args)


@eval_app.command("recommendations")
def recommendations_cmd(
    ctx: typer.Context,
    eval_id: Annotated[str, typer.Argument(help="Eval id")],
) -> None:
    """Show server-generated suggestions for improving the model."""
    args = ctx_to_args(ctx, eval_id=eval_id)
    _recommendations(args)


# ---------------------------------------------------------------------------
# Business logic
# ---------------------------------------------------------------------------


def _resolve(args):  # noqa: ANN001
    from roboflow.cli._resolver import resolve_ws_and_key

    return resolve_ws_and_key(args)


def _eval_error_exit_code(exc: Exception) -> int:
    """Map a model-eval error to the canonical CLI exit code.

    1 = general; 2 = auth; 3 = not found; 4 = conflict (eval not done);
    5 = invalid argument (bad split / confidence). Keeping these distinct
    lets shell scripts and AI agents react to specific failure modes
    without parsing message strings.
    """
    from roboflow.adapters import rfapi

    if isinstance(exc, rfapi.ModelEvalNotFoundError):
        return 3
    if isinstance(exc, rfapi.ModelEvalNotDoneError):
        return 4
    if isinstance(exc, (rfapi.InvalidSplitError, rfapi.InvalidConfidenceError)):
        return 5
    return 1


def _hint_for(exc: Exception) -> Optional[str]:
    """Per-error actionable hint shown alongside the message in non-JSON mode."""
    from roboflow.adapters import rfapi

    if isinstance(exc, rfapi.ModelEvalNotFoundError):
        return "Run 'roboflow eval list' to see eval ids in this workspace."
    if isinstance(exc, rfapi.ModelEvalNotDoneError):
        return "Wait for the eval to finish (status='done') before reading panel data."
    if isinstance(exc, rfapi.InvalidSplitError):
        return "Use one of: train, valid, test (or 'all' where supported)."
    if isinstance(exc, rfapi.InvalidConfidenceError):
        return "Pass an integer between 0 and 100."
    return None


def _list_evals(args):  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error
    from roboflow.cli._table import format_table

    resolved = _resolve(args)
    if not resolved:
        return
    workspace_url, api_key = resolved

    try:
        result = rfapi.list_model_evals(
            api_key,
            workspace_url,
            project=args.project,
            version=args.version,
            model=args.model,
            status=args.status,
            limit=args.limit,
        )
    except Exception as exc:
        output_error(args, str(exc), hint=_hint_for(exc), exit_code=_eval_error_exit_code(exc))
        return

    evals = result.get("evals", [])
    rows = [
        {
            # Prefer DNA's `evalId`; tolerate legacy `id` from older server versions.
            "id": e.get("evalId", e.get("id", "")),
            "status": e.get("status", ""),
            "project": e.get("projectId", ""),
            "version": e.get("versionId", ""),
            "model": e.get("modelId", "") or "",
            "created": e.get("createdAt", ""),
        }
        for e in evals
    ]
    table = format_table(
        rows,
        columns=["id", "status", "project", "version", "model", "created"],
        headers=["ID", "STATUS", "PROJECT", "VERSION", "MODEL", "CREATED"],
    )
    output(args, evals, text=table)


def _get_eval(args):  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error

    resolved = _resolve(args)
    if not resolved:
        return
    workspace_url, api_key = resolved

    try:
        info = rfapi.get_model_eval(api_key, workspace_url, args.eval_id)
    except Exception as exc:
        output_error(args, str(exc), hint=_hint_for(exc), exit_code=_eval_error_exit_code(exc))
        return

    lines = [
        # Prefer DNA's `evalId`; tolerate legacy `id`.
        f"Eval: {info.get('evalId', info.get('id', args.eval_id))}",
        f"  Status:  {info.get('status', '')}",
        f"  Project: {info.get('projectId', '')}",
        f"  Version: {info.get('versionId', '')}",
        f"  Model:   {info.get('modelId', '') or '(none)'}",
        f"  Created: {info.get('createdAt', '')}",
    ]
    config = info.get("config") or {}
    if config:
        lines.append(f"  Config:  overlap={config.get('overlap')} iouThreshold={config.get('iouThreshold')}")
    summary = info.get("summary") or {}
    if summary:
        lines.append(
            f"  Summary: mAP={summary.get('mAP')} precision={summary.get('precision')} recall={summary.get('recall')}"
        )
    output(args, info, text="\n".join(lines))


def _emit_dict(args, payload, *, header: Optional[str] = None) -> None:  # noqa: ANN001
    """Default text rendering for panel commands: pretty-printed JSON.

    Each panel has a deeply nested per-eval shape that doesn't tabulate
    well in the general case (per-class tables exist, but vector clusters
    and recommendations don't). For agent ergonomics we lean on --json,
    and for humans we just pretty-print so they can pipe to jq or eyeball.
    """
    import json as _json

    from roboflow.cli._output import output

    text = _json.dumps(payload, indent=2, default=str)
    if header:
        text = f"{header}\n{text}"
    output(args, payload, text=text)


def _map_results(args):  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output_error

    resolved = _resolve(args)
    if not resolved:
        return
    workspace_url, api_key = resolved

    try:
        data = rfapi.get_model_eval_map_results(api_key, workspace_url, args.eval_id)
    except Exception as exc:
        output_error(args, str(exc), hint=_hint_for(exc), exit_code=_eval_error_exit_code(exc))
        return
    _emit_dict(args, data)


def _confidence_sweep(args):  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output_error

    resolved = _resolve(args)
    if not resolved:
        return
    workspace_url, api_key = resolved

    try:
        data = rfapi.get_model_eval_confidence_sweep(api_key, workspace_url, args.eval_id)
    except Exception as exc:
        output_error(args, str(exc), hint=_hint_for(exc), exit_code=_eval_error_exit_code(exc))
        return
    _emit_dict(args, data)


def _performance_by_class(args):  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error
    from roboflow.cli._table import format_table

    resolved = _resolve(args)
    if not resolved:
        return
    workspace_url, api_key = resolved

    try:
        data = rfapi.get_model_eval_performance_by_class(api_key, workspace_url, args.eval_id, split=args.split)
    except Exception as exc:
        output_error(args, str(exc), hint=_hint_for(exc), exit_code=_eval_error_exit_code(exc))
        return

    classes = data.get("classes", [])
    rows = []
    for c in classes:
        rows.append(
            {
                "class": c.get("className", ""),
                "map50": _fmt_float(c.get("map50")),
                "map50_95": _fmt_float(c.get("map50_95")),
                "map75": _fmt_float(c.get("map75")),
                "precision": _fmt_float(c.get("precision")),
                "recall": _fmt_float(c.get("recall")),
                "f1": _fmt_float(c.get("f1")),
                "opt_thresh": _fmt_float(c.get("optimalThreshold")),
            }
        )
    table = format_table(
        rows,
        columns=["class", "map50", "map50_95", "map75", "precision", "recall", "f1", "opt_thresh"],
        headers=["CLASS", "mAP50", "mAP50-95", "mAP75", "P", "R", "F1", "OPT_THR"],
    )
    header = f"Split: {data.get('split', args.split or 'test')}"
    output(args, data, text=f"{header}\n{table}")


def _fmt_float(value):
    """Format a float to 4 decimal places for table output; pass through ``None`` as ''."""
    if value is None:
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _confusion_matrix(args):  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output_error

    resolved = _resolve(args)
    if not resolved:
        return
    workspace_url, api_key = resolved

    try:
        data = rfapi.get_model_eval_confusion_matrix(
            api_key,
            workspace_url,
            args.eval_id,
            split=args.split,
            confidence=args.confidence,
        )
    except Exception as exc:
        output_error(args, str(exc), hint=_hint_for(exc), exit_code=_eval_error_exit_code(exc))
        return

    header = (
        f"Split: {data.get('split', args.split or 'test')}  "
        f"Confidence: {data.get('confidenceThreshold', args.confidence or 'default')}"
    )
    _emit_dict(args, data, header=header)


def _vector_analysis(args):  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output_error

    resolved = _resolve(args)
    if not resolved:
        return
    workspace_url, api_key = resolved

    try:
        data = rfapi.get_model_eval_vector_analysis(api_key, workspace_url, args.eval_id, confidence=args.confidence)
    except Exception as exc:
        output_error(args, str(exc), hint=_hint_for(exc), exit_code=_eval_error_exit_code(exc))
        return
    _emit_dict(args, data)


def _image_predictions(args):  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error
    from roboflow.cli._table import format_table

    resolved = _resolve(args)
    if not resolved:
        return
    workspace_url, api_key = resolved

    try:
        data = rfapi.get_model_eval_image_predictions(
            api_key,
            workspace_url,
            args.eval_id,
            split=args.split,
            confidence=args.confidence,
            limit=args.limit,
            offset=args.offset,
        )
    except Exception as exc:
        output_error(args, str(exc), hint=_hint_for(exc), exit_code=_eval_error_exit_code(exc))
        return

    images = data.get("images", [])
    rows = []
    for img in images:
        stats = img.get("stats") or {}
        rows.append(
            {
                "image": img.get("imageName", img.get("imageId", "")),
                "split": img.get("split", ""),
                "tp": stats.get("tp", ""),
                "fp": stats.get("fp", ""),
                "fn": stats.get("fn", ""),
                "cluster": img.get("cluster", ""),
            }
        )
    table = format_table(
        rows,
        columns=["image", "split", "tp", "fp", "fn", "cluster"],
        headers=["IMAGE", "SPLIT", "TP", "FP", "FN", "CLUSTER"],
    )
    header = (
        f"Split: {data.get('split', args.split or 'test')}  "
        f"Confidence: {data.get('confidenceThreshold', args.confidence or 'default')}  "
        f"Total: {data.get('totalImages', len(images))}  "
        f"Offset: {data.get('offset', args.offset or 0)}  "
        f"Limit: {data.get('limit', args.limit or 200)}"
    )
    output(args, data, text=f"{header}\n{table}")


def _recommendations(args):  # noqa: ANN001
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output_error

    resolved = _resolve(args)
    if not resolved:
        return
    workspace_url, api_key = resolved

    try:
        data = rfapi.get_model_eval_recommendations(api_key, workspace_url, args.eval_id)
    except Exception as exc:
        output_error(args, str(exc), hint=_hint_for(exc), exit_code=_eval_error_exit_code(exc))
        return
    _emit_dict(args, data)
