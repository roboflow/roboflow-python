"""Hosted auto-label commands: list models, preview, start and track jobs."""

from __future__ import annotations

import json
from typing import Annotated, Any, Callable, Optional

import typer

from roboflow.cli._compat import SortedGroup, ctx_to_args

autolabel_app = typer.Typer(cls=SortedGroup, help="Hosted auto-label jobs", no_args_is_help=True)


@autolabel_app.command("models")
def models(ctx: typer.Context) -> None:
    """List the foundation models available for auto-labeling in this workspace."""
    _models(ctx_to_args(ctx))


@autolabel_app.command("preview")
def preview(
    ctx: typer.Context,
    project: Annotated[str, typer.Option("-p", "--project", help="Project ID")],
    model: Annotated[str, typer.Option("-m", "--model", help="Foundation model ID from 'autolabel models'")],
    image: Annotated[str, typer.Option("--image", help="Sample image: HTTPS URL or local file path")],
    classes: Annotated[
        Optional[list[str]],
        typer.Option("--class", help="Class name to detect; repeat for multiple classes"),
    ] = None,
    ontology: Annotated[
        Optional[str],
        typer.Option("--ontology", help='JSON mapping of class name to prompt, e.g. \'{"cat": "a cat"}\''),
    ] = None,
    confidence: Annotated[
        Optional[float], typer.Option("--confidence", help="Detection threshold from 0.0 to 1.0 (sam3 only)")
    ] = None,
) -> None:
    """Preview one image with a foundation model. Free: no job is created."""
    args = ctx_to_args(ctx, project=project)
    resolved_ontology = _parse_ontology(args, ontology, classes)
    if resolved_ontology is _INVALID:
        return
    from roboflow.util.autolabel_utils import image_payload

    _project_command(
        args,
        lambda key, workspace, proj: _rfapi().preview_autolabel(
            key,
            workspace,
            proj,
            model_type=model,
            image=image_payload(image),
            ontology=resolved_ontology,
            confidence_threshold=confidence,
        ),
    )


@autolabel_app.command("start")
def start(
    ctx: typer.Context,
    project: Annotated[str, typer.Option("-p", "--project", help="Project ID")],
    batch_id: Annotated[str, typer.Option("--batch-id", help="Source batch ID containing the images to label")],
    model: Annotated[
        str,
        typer.Option(
            "-m",
            "--model",
            help="Foundation model ID from 'autolabel models', or a Roboflow model ID with --model-type roboflow",
        ),
    ],
    model_type: Annotated[
        str,
        typer.Option("--model-type", help="'foundational' (hosted foundation model) or 'roboflow' (trained model)"),
    ] = "foundational",
    classes: Annotated[
        Optional[list[str]],
        typer.Option("--class", help="Class name to label; repeat for multiple classes"),
    ] = None,
    ontology: Annotated[
        Optional[str],
        typer.Option("--ontology", help='JSON mapping of class name to prompt, e.g. \'{"cat": "a cat"}\''),
    ] = None,
    num_images: Annotated[
        Optional[int], typer.Option("--num-images", help="Number of images to label (default: whole batch)")
    ] = None,
    confidence: Annotated[
        Optional[float], typer.Option("--confidence", help="Confidence threshold applied to every class")
    ] = None,
    confidence_thresholds: Annotated[
        Optional[str],
        typer.Option("--confidence-thresholds", help="JSON per-class thresholds, e.g. '{\"cat\": 0.5}'"),
    ] = None,
    no_nms: Annotated[bool, typer.Option("--no-nms", help="Disable non-max suppression")] = False,
    reviewer: Annotated[
        Optional[str], typer.Option("--reviewer", help="Reviewer email for the resulting annotation job")
    ] = None,
    model_options: Annotated[
        Optional[str],
        typer.Option("--model-options", help='JSON model options, e.g. \'{"outputFormat": "polygon"}\''),
    ] = None,
) -> None:
    """Start a hosted auto-label job over a batch of images."""
    args = ctx_to_args(ctx, project=project)
    resolved_ontology = _parse_ontology(args, ontology, classes)
    if resolved_ontology is _INVALID:
        return
    resolved_thresholds = _parse_json_option(args, "--confidence-thresholds", confidence_thresholds)
    resolved_options = _parse_json_option(args, "--model-options", model_options)
    if resolved_thresholds is _INVALID or resolved_options is _INVALID:
        return
    from roboflow.util.autolabel_utils import resolve_model

    def start_job(key: str, workspace: str, proj: str) -> Any:
        wire_model_type, wire_options = resolve_model(model, model_type, resolved_options)
        return _rfapi().start_autolabel_job(
            key,
            workspace,
            proj,
            batch_id=batch_id,
            model_type=wire_model_type,
            ontology=resolved_ontology,
            num_images_to_label=num_images,
            default_confidence=confidence,
            confidence_thresholds=resolved_thresholds,
            run_nms=False if no_nms else None,
            reviewer_email=reviewer,
            model_options=wire_options,
        )

    _project_command(args, start_job)


@autolabel_app.command("job")
def job(
    ctx: typer.Context,
    job_id: Annotated[str, typer.Argument(help="Auto-label job ID returned by 'autolabel start'")],
) -> None:
    """Get status and per-subjob progress for an auto-label job."""
    args = ctx_to_args(ctx)
    _workspace_command(args, lambda key, workspace: _rfapi().get_autolabel_job(key, workspace, job_id))


# ---------------------------------------------------------------------------
# Business logic
# ---------------------------------------------------------------------------

_INVALID = object()


def _rfapi():
    from roboflow.adapters import rfapi

    return rfapi


def _parse_json_option(args: Any, flag: str, raw: Optional[str]) -> Any:
    from roboflow.cli._output import output_error

    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except ValueError:
        output_error(args, f"{flag} must be valid JSON.")
        return _INVALID
    if not isinstance(value, dict):
        output_error(args, f"{flag} must be a JSON object.")
        return _INVALID
    return value


def _parse_ontology(args: Any, ontology: Optional[str], classes: Optional[list[str]]) -> Any:
    """Build the ontology from --ontology JSON (takes precedence) or repeated --class."""
    if ontology is not None:
        return _parse_json_option(args, "--ontology", ontology)
    if classes:
        return {name: name for name in classes}
    return None


def _resolve_workspace(args: Any) -> tuple[Optional[str], Optional[str]]:
    from roboflow.cli._output import output_error
    from roboflow.cli._resolver import resolve_default_workspace
    from roboflow.config import load_roboflow_api_key

    workspace_url = args.workspace or resolve_default_workspace(api_key=args.api_key)
    if not workspace_url:
        output_error(args, "No workspace specified.", hint="Use --workspace or run 'roboflow auth login'.")
        return None, None
    api_key = args.api_key or load_roboflow_api_key(workspace_url)
    if not api_key:
        output_error(args, "No API key found.", hint="Set ROBOFLOW_API_KEY or run 'roboflow auth login'.", exit_code=2)
        return None, None
    return api_key, workspace_url


def _resolve_project(args: Any) -> tuple[Optional[str], Optional[str], Optional[str]]:
    from roboflow.cli._output import output_error
    from roboflow.cli._resolver import resolve_resource
    from roboflow.config import load_roboflow_api_key

    try:
        workspace, project, _version = resolve_resource(args.project, workspace_override=args.workspace)
    except ValueError as exc:
        output_error(args, str(exc))
        return None, None, None
    api_key = args.api_key or load_roboflow_api_key(workspace)
    if not api_key:
        output_error(args, "No API key found.", hint="Set ROBOFLOW_API_KEY or run 'roboflow auth login'.", exit_code=2)
        return None, None, None
    return api_key, workspace, project


def _run(args: Any, operation: Callable[[], Any], text: Optional[Callable[[Any], str]] = None) -> None:
    from roboflow.cli._output import output, output_api_error, output_error

    try:
        data = operation()
    except _rfapi().RoboflowError as exc:
        output_api_error(args, exc)
        return
    except ValueError as exc:
        output_error(args, str(exc))
        return
    output(args, data, text=text(data) if text else None)


def _workspace_command(args: Any, operation: Callable[[str, str], Any]) -> None:
    api_key, workspace_url = _resolve_workspace(args)
    if not workspace_url:
        return
    _run(args, lambda: operation(api_key, workspace_url))


def _project_command(args: Any, operation: Callable[[str, str, str], Any]) -> None:
    api_key, workspace, project = _resolve_project(args)
    if not project:
        return
    _run(args, lambda: operation(api_key, workspace, project))


def _models(args: Any) -> None:
    from roboflow.cli._table import format_table

    def table(data: Any) -> str:
        rows = [
            {
                "id": model.get("id", ""),
                "name": model.get("name", ""),
                "available": "yes" if model.get("available", True) else "no",
                "default": "yes" if model.get("isDefault") else "",
                "credits": model.get("creditsPerImage", ""),
                "ontology": model.get("ontologyFormat", ""),
            }
            for model in data.get("models", [])
        ]
        return format_table(
            rows,
            columns=["id", "name", "available", "default", "credits", "ontology"],
            headers=["ID", "NAME", "AVAILABLE", "DEFAULT", "CREDITS/IMAGE", "ONTOLOGY"],
        )

    api_key, workspace_url = _resolve_workspace(args)
    if not workspace_url:
        return
    _run(args, lambda: _rfapi().list_autolabel_models(api_key, workspace_url), text=table)
