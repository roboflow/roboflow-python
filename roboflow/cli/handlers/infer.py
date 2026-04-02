"""Infer command: run inference on an image."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the top-level ``infer`` command."""
    infer_parser = subparsers.add_parser("infer", help="Run inference on an image")
    infer_parser.add_argument(
        "file",
        help="Path to an image file",
    )
    infer_parser.add_argument(
        "-m",
        "--model",
        dest="model",
        required=True,
        help="Model ID (project/version, e.g. my-project/3)",
    )
    infer_parser.add_argument(
        "-c",
        "--confidence",
        dest="confidence",
        type=float,
        default=0.5,
        help="Confidence threshold 0.0-1.0 (default: 0.5)",
    )
    infer_parser.add_argument(
        "-o",
        "--overlap",
        dest="overlap",
        type=float,
        default=0.5,
        help="Overlap threshold 0.0-1.0 (default: 0.5)",
    )
    infer_parser.add_argument(
        "-t",
        "--type",
        dest="type",
        default=None,
        choices=[
            "object-detection",
            "classification",
            "instance-segmentation",
            "semantic-segmentation",
            "keypoint-detection",
        ],
        help="Model type (auto-detected if not specified)",
    )
    infer_parser.set_defaults(func=_infer)


def _infer(args: argparse.Namespace) -> None:
    from roboflow.adapters import rfapi
    from roboflow.cli._output import output, output_error
    from roboflow.cli._resolver import resolve_resource
    from roboflow.config import load_roboflow_api_key

    try:
        workspace_url, project_slug, version = resolve_resource(args.model, workspace_override=args.workspace)
    except ValueError as exc:
        output_error(args, str(exc))
        return

    api_key = args.api_key or load_roboflow_api_key(workspace_url)
    if not api_key:
        output_error(args, "No API key found.", hint="Set ROBOFLOW_API_KEY or run 'roboflow auth login'.", exit_code=2)
        return

    project_type = args.type
    if not project_type:
        try:
            dataset_json = rfapi.get_project(api_key, workspace_url, project_slug)
            project_type = dataset_json["project"]["type"]
        except (rfapi.RoboflowError, KeyError) as exc:
            output_error(args, f"Could not determine project type: {exc}", hint="Use -t/--type to specify.")
            return

    # Lazy imports of model classes
    from roboflow.models.classification import ClassificationModel
    from roboflow.models.instance_segmentation import InstanceSegmentationModel
    from roboflow.models.keypoint_detection import KeypointDetectionModel
    from roboflow.models.object_detection import ObjectDetectionModel
    from roboflow.models.semantic_segmentation import SemanticSegmentationModel

    model_class_map = {
        "object-detection": ObjectDetectionModel,
        "classification": ClassificationModel,
        "instance-segmentation": InstanceSegmentationModel,
        "semantic-segmentation": SemanticSegmentationModel,
        "keypoint-detection": KeypointDetectionModel,
    }

    model_cls = model_class_map.get(project_type)
    if model_cls is None:
        output_error(args, f"Unsupported project type: {project_type}")
        return

    if version is not None:
        project_url = f"{workspace_url}/{project_slug}/{version}"
    else:
        project_url = f"{workspace_url}/{project_slug}"

    model = model_cls(api_key, project_url)

    kwargs = {}
    if args.confidence is not None and project_type in [
        "object-detection",
        "instance-segmentation",
        "semantic-segmentation",
    ]:
        kwargs["confidence"] = int(args.confidence * 100)
    if args.overlap is not None and project_type == "object-detection":
        kwargs["overlap"] = int(args.overlap * 100)

    try:
        group = model.predict(args.file, **kwargs)
    except Exception as exc:
        output_error(args, f"Inference failed: {exc}")
        return

    # Serialize predictions for JSON output
    if getattr(args, "json", False):
        predictions = []
        for pred in group:
            if hasattr(pred, "json"):
                predictions.append(pred.json())
            elif hasattr(pred, "__dict__"):
                predictions.append(pred.__dict__)
            else:
                predictions.append(str(pred))
        output(args, predictions)
    else:
        output(args, None, text=str(group))
