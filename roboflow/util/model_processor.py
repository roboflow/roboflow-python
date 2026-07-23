"""Packaging of custom model weights for Roboflow upload.

The public, non-interactive entry point is :func:`package_custom_weights`. It
only builds the upload archive; it never prompts, prints, or uploads, so it is
safe to call from servers and other headless environments (for example the
Roboflow MCP server)::

    from roboflow.util.model_processor import package_custom_weights

    bundle = package_custom_weights("yolov8n", "runs/detect/train")
    try:
        ...  # upload bundle.archive_path
    finally:
        bundle.cleanup()

Expected, user-correctable failures raise :class:`ModelPackagingError`
subclasses; anything else escaping these helpers is a bug.

The legacy :func:`process` entry point and the ``Version.deploy`` /
``Workspace.deploy_model`` flows wrap the packaging step with
:func:`package_custom_weights_interactive`, which preserves the historical
print-and-confirm CLI behavior.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml

from roboflow.config import (
    DISABLE_CLASS_SORTING,
    TASK_CLS,
    TASK_DET,
    TASK_OBB,
    TASK_POSE,
    TASK_SEG,
    TASK_SEM,
    TYPE_CLASSICATION,
    TYPE_INSTANCE_SEGMENTATION,
    TYPE_KEYPOINT_DETECTION,
    TYPE_OBJECT_DETECTION,
    TYPE_SEMANTIC_SEGMENTATION,
)
from roboflow.util.versions import get_wrong_dependencies_versions, normalize_yolo_model_type

SUPPORTED_MODELS = (
    "yolov5",
    "yolov7",
    "yolov7-seg",
    "yolov8",
    "yolov9",
    "yolov10",
    "yolov11",
    "yolov12",
    "yolo26",
    "yolonas",
    "paligemma",
    "paligemma2",
    "florence-2",
    "rfdetr",
)

SUPPORTED_HUGGINGFACE_TYPES = (
    "florence-2-base",
    "florence-2-large",
    "paligemma-3b-pt-224",
    "paligemma-3b-pt-448",
    "paligemma-3b-pt-896",
    "paligemma2-3b-pt-224",
    "paligemma2-3b-pt-448",
    "paligemma2-3b-pt-896",
    "paligemma2-3b-pt-224-peft",
    "paligemma2-3b-pt-448-peft",
    "paligemma2-3b-pt-896-peft",
)

# Minimum rf-detr release shipping ``RFDETR.export_for_roboflow`` (used to rebuild
# an upload bundle from a raw PyTorch-Lightning checkpoint).
RFDETR_MIN_VERSION = "1.8.0"

# rf-detr model_type -> RFDETR subclass name. Single source of truth for both the
# supported-type check and the ``from_checkpoint`` fallback (used when a raw
# checkpoint lacks the metadata rf-detr needs to infer its own class).
_RFDETR_MODEL_TYPE_TO_CLASS = {
    # Detection
    "rfdetr-base": "RFDETRBase",
    "rfdetr-nano": "RFDETRNano",
    "rfdetr-small": "RFDETRSmall",
    "rfdetr-medium": "RFDETRMedium",
    "rfdetr-large": "RFDETRLarge",
    "rfdetr-xlarge": "RFDETRXLarge",
    "rfdetr-2xlarge": "RFDETR2XLarge",
    # Segmentation
    "rfdetr-seg-nano": "RFDETRSegNano",
    "rfdetr-seg-small": "RFDETRSegSmall",
    "rfdetr-seg-medium": "RFDETRSegMedium",
    "rfdetr-seg-large": "RFDETRSegLarge",
    "rfdetr-seg-xlarge": "RFDETRSegXLarge",
    "rfdetr-seg-2xlarge": "RFDETRSeg2XLarge",
    # Keypoint detection
    "rfdetr-keypoint-preview": "RFDETRKeypointPreview",
}

SUPPORTED_RFDETR_TYPES = tuple(_RFDETR_MODEL_TYPE_TO_CLASS)

DEFAULT_WEIGHTS_FILENAME = "weights/best.pt"

# YOLO families Roboflow rejects without a size suffix (e.g. `yolov8` must be
# `yolov8n`/`yolov8s`/...). Legacy yolov5/7/9 go through the opt.yaml path and are
# intentionally excluded.
ULTRALYTICS_YOLO_FAMILIES = ("yolov8", "yolov10", "yolov11", "yolov12", "yolo26")

# Canonical (depth_multiple, width_multiple) -> size letter for the classic YOLO
# scaling (v5/v8/v9/v10). Newer families (v11+) instead store an explicit ``scale``
# letter in the model yaml, which is read first.
YOLO_DEPTH_WIDTH_TO_SIZE = {
    (0.33, 0.25): "n",
    (0.33, 0.50): "s",
    (0.67, 0.75): "m",
    (0.67, 1.00): "b",
    (1.00, 1.00): "l",
    (1.00, 1.25): "x",
}

# Position-encoding grid size (DINOv2 tokens per side) each *known* RF-DETR variant
# is built with, mirroring rfdetr/config.py. Roboflow reconstructs the architecture
# from the model_type at the variant's default resolution, so a checkpoint trained at
# that default must match or state_dict loading fails on the backbone
# position_embeddings. Variants absent here (e.g. detection xlarge/2xlarge, which have
# no standard config) are not grid-checked. A checkpoint trained at a custom
# resolution may not match any entry; that case warns rather than blocks.
RFDETR_POSITIONAL_ENCODING_SIZE = {
    "rfdetr-nano": 24,
    "rfdetr-small": 32,
    "rfdetr-medium": 36,
    "rfdetr-base": 37,
    "rfdetr-large": 44,
    "rfdetr-seg-nano": 26,
    "rfdetr-seg-small": 32,
    "rfdetr-seg-medium": 36,
    "rfdetr-seg-large": 42,
    "rfdetr-seg-xlarge": 52,
    "rfdetr-seg-2xlarge": 64,
    # Keypoint (576 / patch_size 12 = 48)
    "rfdetr-keypoint-preview": 48,
}


class ModelPackagingError(Exception):
    """Custom weights could not be packaged for a user-correctable reason.

    Consumers can treat any instance of this class as an expected input problem
    (bad model_type, missing files, mismatched metadata, ...) and surface the
    message to the user. Exceptions that are not ModelPackagingError indicate
    bugs and are deliberately not wrapped.
    """


class UnsupportedModelError(ModelPackagingError, ValueError):
    """The model_type is not supported for custom weights upload."""


class TaskMismatchError(ModelPackagingError, ValueError):
    """The model_type's task conflicts with the checkpoint or the project type."""


class MissingFileError(ModelPackagingError, FileNotFoundError):
    """A file required for packaging was not found."""


class MissingDependencyError(ModelPackagingError, RuntimeError):
    """A Python package required to package these weights is not installed."""


class DependencyMismatchError(ModelPackagingError, RuntimeError):
    """An installed dependency version differs from the recommended one.

    Retry with ``allow_dependency_mismatch=True`` to package anyway.
    """

    retry_flag = "allow_dependency_mismatch"

    def __init__(self, message: str, *, dependency: str, required: str, installed: str):
        super().__init__(message)
        self.dependency = dependency
        self.required = required
        self.installed = installed


class SizeMismatchError(ModelPackagingError, ValueError):
    """The declared model size/variant conflicts with the checkpoint architecture.

    Retry with ``allow_size_mismatch=True`` to package the requested model_type
    as-is.
    """

    retry_flag = "allow_size_mismatch"

    def __init__(self, message: str, *, requested: str, detected: str | None = None):
        super().__init__(message)
        self.requested = requested
        self.detected = detected


@dataclass(frozen=True)
class ModelUploadBundle:
    """Packaged archive ready to upload through the Roboflow API.

    ``model_type`` is the resolved type (it may differ from the requested one,
    e.g. ``yolov8`` filled in as ``yolov8n`` from the checkpoint architecture).
    ``owns_build_dir`` is True when :func:`package_custom_weights` created a
    temporary build directory; call :meth:`cleanup` once the archive has been
    consumed.
    """

    archive_path: Path
    build_dir: Path
    model_type: str
    warnings: tuple[str, ...] = ()
    owns_build_dir: bool = False

    @property
    def size_bytes(self) -> int:
        return self.archive_path.stat().st_size

    def cleanup(self) -> None:
        """Remove the build directory if this bundle created it (no-op otherwise)."""
        if self.owns_build_dir:
            shutil.rmtree(self.build_dir, ignore_errors=True)


def task_of_model_type(model_type: str) -> str:
    """Canonical task for a deploy model_type string.

    Non-detect tasks double as the model_type suffix token
    (e.g. 'yolov11-seg' -> TASK_SEG). Plain 'yolov11' / 'rfdetr-base' -> TASK_DET.

    Keypoint/pose models may spell the token as either 'pose' (Ultralytics) or
    'keypoint' (rf-detr, e.g. 'rfdetr-keypoint-preview'); both map to TASK_POSE.
    """
    s = model_type.lower()
    if "keypoint" in s:
        return TASK_POSE
    for task in (TASK_SEM, TASK_SEG, TASK_POSE, TASK_CLS, TASK_OBB):
        if task in s:
            return task
    return TASK_DET


def _checkpoint_args_as_dict(raw_args: Any) -> dict[str, Any]:
    """Coerce a checkpoint's ``args`` (dict, argparse.Namespace, or None) to a dict.

    A corrupt checkpoint may store ``args`` as a scalar or list; those have no
    meaningful attributes, so treat them as no-args rather than letting
    ``vars()`` raise a bare ``TypeError`` outside the ModelPackagingError contract.
    """
    if isinstance(raw_args, dict):
        return raw_args
    if hasattr(raw_args, "__dict__"):
        return dict(vars(raw_args))
    return {}


def _resolve_within_source(source_dir: Path, filename: str) -> Path:
    """Resolve ``filename`` against ``source_dir``, refusing to escape it.

    ``filename`` is documented as relative to ``model_path`` and is forwarded
    verbatim by hosted callers (the MCP server). An absolute path or ``..``
    segments would let a caller read weights from outside the model directory,
    so reject both instead of silently packaging a file the caller never
    intended. ``source_dir`` is already resolved; resolving the join collapses
    ``..`` and follows symlinks before the containment check.

    Absoluteness is tested under both OS conventions, not just the host's: the
    hosted MCP forwards caller-supplied paths, so a POSIX absolute like
    ``/etc/passwd`` must be rejected even when packaging happens to run on
    Windows (where ``Path.is_absolute()`` alone would miss it, and vice versa).
    """
    if PurePosixPath(filename).is_absolute() or PureWindowsPath(filename).is_absolute():
        raise ModelPackagingError(f"filename '{filename}' must be a path relative to model_path, not an absolute path.")
    resolved = (source_dir / filename).resolve()
    if resolved != source_dir and source_dir not in resolved.parents:
        raise ModelPackagingError(
            f"filename '{filename}' resolves outside model_path '{source_dir}'. "
            "It must point to a checkpoint inside the model directory."
        )
    # '' / '.' resolve to model_path itself and a subdirectory passes the
    # containment check; all would reach torch.load(<dir>) and leak a raw
    # IsADirectoryError outside the ModelPackagingError contract.
    if resolved.is_dir():
        raise ModelPackagingError(f"filename '{filename}' must point to a checkpoint file, not a directory.")
    return resolved


def validate_model_type_for_project(model_type: str, project_type: str, project_id: str) -> None:
    """Raise TaskMismatchError if model_type's task doesn't match the Roboflow project type."""
    expected = {
        TYPE_OBJECT_DETECTION: TASK_DET,
        TYPE_INSTANCE_SEGMENTATION: TASK_SEG,
        TYPE_SEMANTIC_SEGMENTATION: TASK_SEM,
        TYPE_KEYPOINT_DETECTION: TASK_POSE,
        TYPE_CLASSICATION: TASK_CLS,
    }.get(project_type)
    if expected is None:
        return
    actual = task_of_model_type(model_type)
    if actual != expected:
        raise TaskMismatchError(
            f"Project '{project_id}' is type '{project_type}' (task '{expected}') "
            f"but model_type '{model_type}' implies task '{actual}'."
        )


def package_custom_weights(
    model_type: str,
    model_path: str,
    filename: str = DEFAULT_WEIGHTS_FILENAME,
    *,
    build_dir: str | Path | None = None,
    allow_dependency_mismatch: bool = False,
    allow_size_mismatch: bool = False,
) -> ModelUploadBundle:
    """Package locally trained custom weights into a Roboflow upload archive.

    This is the public packaging entry point. It is non-interactive and free of
    side effects on ``model_path``: it never prompts, prints, or writes into the
    source directory. Heavy dependencies (torch, ultralytics) are imported
    lazily, only for the model families that need them.

    Args:
        model_type: Roboflow model type (e.g. "yolov8n", "rfdetr-base").
        model_path: Directory containing the trained model artifacts.
        filename: Weights file path, relative to ``model_path``.
        build_dir: Directory to write intermediate artifacts and the final
            archive into. Defaults to a fresh temporary directory owned by the
            returned bundle; call ``bundle.cleanup()`` when done.
        allow_dependency_mismatch: Record a warning instead of raising
            DependencyMismatchError when an installed dependency version is not
            the recommended one.
        allow_size_mismatch: Record a warning instead of raising
            SizeMismatchError when the declared model size/variant conflicts
            with the checkpoint architecture.

    Returns:
        ModelUploadBundle with the archive path, the resolved model_type, and
        any warnings collected while packaging.

    Raises:
        ModelPackagingError: (or a subclass) for user-correctable problems.
    """
    normalized_model_type = normalize_yolo_model_type(model_type.strip())
    source_dir = Path(model_path).expanduser().resolve()
    if not source_dir.is_dir():
        raise MissingFileError(f"Model path '{model_path}' does not exist or is not a directory.")
    _resolve_within_source(source_dir, filename)

    owns_build_dir = build_dir is None
    if build_dir is None:
        build_path = Path(tempfile.mkdtemp(prefix="roboflow-package-"))
    else:
        build_path = Path(build_dir).expanduser().resolve()
        build_path.mkdir(parents=True, exist_ok=True)

    try:
        archive_path, resolved_model_type, warnings = _process_model(
            model_type=normalized_model_type,
            model_path=source_dir,
            filename=filename,
            build_dir=build_path,
            allow_dependency_mismatch=allow_dependency_mismatch,
            allow_size_mismatch=allow_size_mismatch,
        )
    except BaseException:
        if owns_build_dir:
            shutil.rmtree(build_path, ignore_errors=True)
        raise

    return ModelUploadBundle(
        archive_path=archive_path,
        build_dir=build_path,
        model_type=resolved_model_type,
        warnings=tuple(warnings),
        owns_build_dir=owns_build_dir,
    )


def package_custom_weights_interactive(
    model_type: str,
    model_path: str,
    filename: str = DEFAULT_WEIGHTS_FILENAME,
    *,
    build_dir: str | Path | None = None,
) -> ModelUploadBundle:
    """Package weights with the historical interactive SDK behavior.

    Used by ``Version.deploy`` and ``Workspace.deploy_model``: warnings are
    printed, and dependency/size mismatches ask for confirmation before
    retrying with the corresponding override. Declining re-raises the error.
    """
    allow_dependency_mismatch = False
    allow_size_mismatch = False
    while True:
        try:
            bundle = package_custom_weights(
                model_type,
                model_path,
                filename,
                build_dir=build_dir,
                allow_dependency_mismatch=allow_dependency_mismatch,
                allow_size_mismatch=allow_size_mismatch,
            )
        except (DependencyMismatchError, SizeMismatchError) as error:
            print(error)
            answer = input("Would you like to continue anyway? y/n: ")
            if answer.lower() != "y":
                raise
            if isinstance(error, DependencyMismatchError):
                allow_dependency_mismatch = True
            else:
                allow_size_mismatch = True
            continue
        for warning in bundle.warnings:
            print(warning)
        return bundle


def process(model_type: str, model_path: str, filename: str) -> tuple[str, str]:
    """Legacy packaging entry point, kept for backwards compatibility.

    Preserves the historical contract end to end: packages into ``model_path``
    (intermediate artifacts and the final archive land there), prints packaging
    warnings, asks for confirmation on dependency/size mismatches, and returns
    ``(archive_file_name, resolved_model_type)``. Headless code should call
    :func:`package_custom_weights` instead.
    """
    bundle = package_custom_weights_interactive(model_type, model_path, filename, build_dir=model_path)
    return bundle.archive_path.name, bundle.model_type


def _process_model(
    *,
    model_type: str,
    model_path: Path,
    filename: str,
    build_dir: Path,
    allow_dependency_mismatch: bool,
    allow_size_mismatch: bool,
) -> tuple[Path, str, list[str]]:
    if not model_type.startswith(SUPPORTED_MODELS):
        raise UnsupportedModelError(
            f"Model type '{model_type}' is not supported for custom weights upload. "
            f"It must start with a supported family: {', '.join(SUPPORTED_MODELS)}."
        )

    if model_type.startswith(("paligemma", "paligemma2", "florence-2")):
        return _process_huggingface(model_type, model_path, build_dir)
    if model_type.startswith("yolonas"):
        return _process_yolonas(model_type, model_path, filename, build_dir)
    if model_type.startswith("rfdetr"):
        return _process_rfdetr(model_type, model_path, filename, build_dir, allow_size_mismatch)
    return _process_yolo(
        model_type,
        model_path,
        filename,
        build_dir,
        allow_dependency_mismatch,
        allow_size_mismatch,
    )


def _import_required_module(module_name: str, install_hint: str) -> Any:
    try:
        return import_module(module_name)
    except ImportError as exc:
        raise MissingDependencyError(
            f"The '{module_name}' Python package is required to package these "
            f"custom weights. Install it with `{install_hint}`."
        ) from exc


def _check_dependency_version(
    *,
    dependency: str,
    operator: str,
    required_version: str,
    allow_mismatch: bool,
    warnings: list[str],
) -> None:
    mismatches = get_wrong_dependencies_versions([(dependency, operator, required_version)])
    if not mismatches:
        return
    _, _, _, installed = mismatches[0]
    message = (
        f"{dependency}{operator}{required_version} is recommended for this "
        f"upload, but {dependency} {installed} is installed."
    )
    if allow_mismatch:
        warnings.append(message)
        return
    raise DependencyMismatchError(
        f"{message} Retry with allow_dependency_mismatch=True to package with the "
        f"installed version, or `pip install {dependency}{operator}{required_version}`.",
        dependency=dependency,
        required=f"{dependency}{operator}{required_version}",
        installed=installed,
    )


def _detect_yolo_task(model_instance: Any) -> str | None:
    """Detect the training task of an Ultralytics model instance via its class name."""
    if model_instance is None:
        return None
    return {
        "DetectionModel": TASK_DET,
        "SegmentationModel": TASK_SEG,
        "SemanticSegmentationModel": TASK_SEM,
        "PoseModel": TASK_POSE,
        "ClassificationModel": TASK_CLS,
        "OBBModel": TASK_OBB,
    }.get(type(model_instance).__name__)


def _class_names_from_model_instance(model_instance: Any) -> list[str]:
    names = getattr(model_instance, "names", None)
    if isinstance(names, list):
        return names
    if isinstance(names, dict):
        # NOTE: When DISABLE_CLASS_SORTING is enabled, users are responsible for ensuring
        # their model's names dict has properly ordered/sequential keys. Non-sequential keys
        # may result in incorrect class-to-index mappings.
        if not DISABLE_CLASS_SORTING:
            return [name for _, name in sorted(names.items(), key=lambda item: item[0])]
        return [name for _, name in names.items()]
    raise ModelPackagingError("Could not extract class names from the model checkpoint.")


def _filtered_args(args: Any) -> dict[str, Any]:
    # A corrupt checkpoint may store args as a scalar/None; coerce via the shared
    # helper so it becomes {} rather than raising a raw TypeError from vars().
    return {k: v for k, v in _checkpoint_args_as_dict(args).items() if k in {"model", "imgsz", "batch"}}


def _load_checkpoint(torch_module: Any, checkpoint_path: Path, *, map_location: str | None = None) -> Any:
    kwargs: dict[str, Any] = {"weights_only": False}
    if map_location is not None:
        kwargs["map_location"] = map_location
    return torch_module.load(checkpoint_path, **kwargs)


def _legacy_yolo_args(opts: dict[str, Any], opt_path: Path) -> dict[str, Any]:
    """Return required legacy YOLO upload args from opt.yaml."""
    if "imgsz" in opts:
        image_size = opts["imgsz"]
    elif "img_size" in opts:
        image_size = opts["img_size"]
    else:
        raise ModelPackagingError(f"{opt_path} is missing required key 'imgsz' or 'img_size'.")
    if "batch_size" not in opts:
        raise ModelPackagingError(f"{opt_path} is missing required key 'batch_size'.")
    return {"imgsz": image_size, "batch": opts["batch_size"]}


def _infer_yolo_size(model_instance: Any) -> str | None:
    """Infer a YOLO size letter (n/s/m/l/x/...) from a loaded checkpoint.

    Prefers an explicit ``scale`` letter in the model yaml (set by newer
    Ultralytics), then maps the ``(depth_multiple, width_multiple)`` pair used by
    the classic scaling. Returns None when neither signal is present.
    """
    yaml_cfg = getattr(model_instance, "yaml", None) or {}
    scale = yaml_cfg.get("scale")
    if isinstance(scale, str) and len(scale) == 1 and scale.isalpha():
        return scale.lower()

    depth = yaml_cfg.get("depth_multiple")
    width = yaml_cfg.get("width_multiple")
    if isinstance(depth, (int, float)) and isinstance(width, (int, float)):
        for (ref_depth, ref_width), letter in YOLO_DEPTH_WIDTH_TO_SIZE.items():
            if abs(depth - ref_depth) < 1e-6 and abs(width - ref_width) < 1e-6:
                return letter
    return None


def _resolve_yolo_size(
    model_type: str,
    model_instance: Any,
    warnings: list[str],
    allow_mismatch: bool = False,
) -> str:
    """Fill in or check a YOLO model_type's size suffix against the checkpoint.

    Roboflow rejects bare family names (e.g. ``yolov8``) with an
    ``InvalidModelTypeException`` because it needs the model size, and a size that
    disagrees with the weights fails conversion. A *missing* size is inferred and
    filled in. A *supplied* size that conflicts with the inferred one raises so the
    caller can confirm — unless ``allow_mismatch`` is set, in which case the
    supplied size is packaged as-is with a warning. A user size is also kept when
    the size cannot be inferred. Returns the resolved model_type.
    """
    core = model_type.lower().split("-", 1)[0]
    family = next((f for f in ULTRALYTICS_YOLO_FAMILIES if core.startswith(f)), None)
    if family is None:
        return model_type

    inferred = _infer_yolo_size(model_instance)
    provided = core[len(family) :]
    task_suffix = model_type[len(core) :]

    if inferred is None:
        if not provided:
            if allow_mismatch:
                warnings.append(
                    f"Could not infer a size for '{model_type}' from the checkpoint; "
                    f"uploading the bare family name as requested. Roboflow may reject it "
                    f"if it requires an explicit size."
                )
                return model_type
            raise SizeMismatchError(
                f"model_type '{model_type}' is missing a size suffix and the size "
                f"could not be inferred from the checkpoint. Specify it explicitly, "
                f"e.g. '{family}n', '{family}s', '{family}m', '{family}l', '{family}x'.",
                requested=model_type,
            )
        return model_type

    if not provided:
        warnings.append(
            f"Inferred model size '{family}{inferred}' from the checkpoint "
            f"architecture (model_type was '{model_type}')."
        )
        return f"{family}{inferred}{task_suffix}"

    if provided == inferred:
        return model_type

    if allow_mismatch:
        warnings.append(
            f"model_type '{model_type}' declares size '{provided}', but the checkpoint "
            f"architecture is '{family}{inferred}'. Packaging as '{model_type}' as requested."
        )
        return model_type

    raise SizeMismatchError(
        f"You specified model_type '{model_type}' (size '{provided}'), but the "
        f"checkpoint architecture is '{family}{inferred}'. They don't match, so "
        f"Roboflow's weight conversion would fail. Upload as '{family}{inferred}"
        f"{task_suffix}', or set allow_size_mismatch=True to upload "
        f"'{model_type}' exactly as specified.",
        requested=model_type,
        detected=f"{family}{inferred}{task_suffix}",
    )


def _require_model_attr(model_instance: Any, attr: str, model_type: str) -> Any:
    """Return ``model_instance.<attr>`` or raise a ModelPackagingError.

    Roboflow's server-side conversion needs these fields; a stripped checkpoint
    missing one would otherwise raise a raw ``AttributeError`` outside the
    ModelPackagingError contract (an opaque 500 for hosted callers).
    """
    value = getattr(model_instance, attr, None)
    if value is None:
        raise ModelPackagingError(
            f"The {model_type} checkpoint's model is missing '{attr}'; it does not look "
            "like a complete Ultralytics training checkpoint. Re-export it from your training run."
        )
    return value


def _require_checkpoint_field(checkpoint: Any, key: str, model_type: str) -> Any:
    """Return ``checkpoint[key]`` or raise a ModelPackagingError (see _require_model_attr)."""
    if not isinstance(checkpoint, dict) or key not in checkpoint:
        raise ModelPackagingError(
            f"The {model_type} checkpoint is missing '{key}'; it does not look like a "
            "complete Ultralytics training checkpoint. Re-export it from your training run."
        )
    return checkpoint[key]


def _process_yolo(
    model_type: str,
    model_path: Path,
    filename: str,
    build_dir: Path,
    allow_dependency_mismatch: bool,
    allow_size_mismatch: bool,
) -> tuple[Path, str, list[str]]:
    warnings: list[str] = []
    torch = _import_required_module("torch", "pip install torch")
    ultralytics = None

    if "yolov8" in model_type:
        ultralytics = _import_required_module("ultralytics", "pip install ultralytics==8.0.196")
        _check_dependency_version(
            dependency="ultralytics",
            operator="==",
            required_version="8.0.196",
            allow_mismatch=allow_dependency_mismatch,
            warnings=warnings,
        )
    elif "yolov10" in model_type:
        ultralytics = _import_required_module("ultralytics", "pip install ultralytics")
    elif "yolov11" in model_type:
        ultralytics = _import_required_module("ultralytics", "pip install 'ultralytics>=8.3.0'")
        _check_dependency_version(
            dependency="ultralytics",
            operator=">=",
            required_version="8.3.0",
            allow_mismatch=allow_dependency_mismatch,
            warnings=warnings,
        )
    elif "yolov12" in model_type:
        ultralytics = _import_required_module(
            "ultralytics",
            "pip install git+https://github.com/sunsmarterjie/yolov12.git",
        )
        warnings.append(
            "YOLOv12 uploads must use the Ultralytics fork from "
            "https://github.com/sunsmarterjie/yolov12 or a Roboflow-trained model."
        )
        _check_dependency_version(
            dependency="ultralytics",
            operator="==",
            required_version="8.3.63",
            allow_mismatch=allow_dependency_mismatch,
            warnings=warnings,
        )
    elif "yolo26" in model_type:
        ultralytics = _import_required_module("ultralytics", "pip install ultralytics")

    checkpoint_path = model_path / filename
    if not checkpoint_path.exists():
        raise MissingFileError(f"Model weights file '{checkpoint_path}' was not found.")

    checkpoint = _load_checkpoint(torch, checkpoint_path)
    if not isinstance(checkpoint, dict):
        raise ModelPackagingError(f"Model weights file '{checkpoint_path}' is not a supported checkpoint dictionary.")
    model_instance = checkpoint.get("model") or checkpoint.get("ema")
    if model_instance is None:
        raise ModelPackagingError("Could not find a 'model' or 'ema' entry in the checkpoint.")

    model_type = _resolve_yolo_size(model_type, model_instance, warnings, allow_size_mismatch)

    detected_task = _detect_yolo_task(model_instance)
    if detected_task:
        existing_task = task_of_model_type(model_type)
        if existing_task == TASK_DET and detected_task != TASK_DET:
            model_type = f"{model_type}-{detected_task}"
        elif existing_task != detected_task:
            raise TaskMismatchError(
                f"model_type '{model_type}' implies task '{existing_task}' but the "
                f".pt file is a '{detected_task}' checkpoint. Use a matching model_type."
            )

    class_names = _class_names_from_model_instance(model_instance)
    if any(name in model_type for name in ULTRALYTICS_YOLO_FAMILIES):
        if ultralytics is None:
            ultralytics = _import_required_module("ultralytics", "pip install ultralytics")
        model_yaml = _require_model_attr(model_instance, "yaml", model_type)
        if (
            "-cls" in model_type
            or model_type.startswith("yolov10")
            or model_type.startswith("yolov11")
            or model_type.startswith("yolov12")
            or model_type.startswith("yolo26")
        ):
            if not isinstance(model_yaml, dict) or "nc" not in model_yaml:
                raise ModelPackagingError(
                    f"The {model_type} checkpoint's model config (model.yaml) is missing 'nc'; "
                    "it does not look like a complete Ultralytics training checkpoint."
                )
            nc = model_yaml["nc"]
            args = _require_checkpoint_field(checkpoint, "train_args", model_type)
        else:
            nc = _require_model_attr(model_instance, "nc", model_type)
            args = _require_model_attr(model_instance, "args", model_type)
        model_artifacts: dict[str, Any] = {
            "names": class_names,
            "yaml": model_yaml,
            "nc": nc,
            "args": _filtered_args(args),
            "ultralytics_version": ultralytics.__version__,
            "model_type": model_type,
        }
    else:
        # yolov5 / yolov7 / yolov9 read their upload args from opt.yaml
        opt_path = model_path / "opt.yaml"
        if not opt_path.exists():
            raise MissingFileError(f"You must provide an opt.yaml file at '{opt_path}' for {model_type} uploads.")
        with opt_path.open() as stream:
            opts = yaml.safe_load(stream) or {}
        model_artifacts = {
            "names": class_names,
            "nc": _require_model_attr(model_instance, "nc", model_type),
            "args": _legacy_yolo_args(opts, opt_path),
            "model_type": model_type,
        }
        if hasattr(model_instance, "yaml"):
            model_artifacts["yaml"] = model_instance.yaml

    (build_dir / "model_artifacts.json").write_text(json.dumps(model_artifacts))
    torch.save(model_instance.state_dict(), build_dir / "state_dict.pt")

    archive_path = build_dir / "roboflow_deploy.zip"
    _write_zip(
        archive_path,
        [
            (model_path / "results.csv", "results.csv", False),
            (model_path / "results.png", "results.png", False),
            (build_dir / "model_artifacts.json", "model_artifacts.json", True),
            (build_dir / "state_dict.pt", "state_dict.pt", True),
        ],
    )
    return archive_path, model_type, warnings


def _detect_rfdetr_task(checkpoint: Any) -> str | None:
    """Detect the training task of an rf-detr checkpoint.

    rf-detr supports weight upload for detection, instance segmentation, and
    keypoint detection. Modern checkpoints (rf-detr v1.7+) store the Python
    class name at `checkpoint["model_name"]` (e.g. 'RFDETRNano' vs
    'RFDETRSegNano' vs 'RFDETRKeypointPreview').

    The deploy bundle written by rf-detr's `export_for_roboflow` only serialises
    `{"model", "args"}` — it drops `model_name` — so detection must also work
    from `args`: keypoint checkpoints carry a non-empty `args.num_keypoints_per_class`,
    and detection/segmentation checkpoints carry `args.segmentation_head: bool`.
    """
    if not isinstance(checkpoint, dict):
        return None
    model_name = checkpoint.get("model_name")
    if isinstance(model_name, str):
        name = model_name.lower()
        if "keypoint" in name:
            return TASK_POSE
        return TASK_SEG if TASK_SEG in name else TASK_DET
    raw_args = checkpoint.get("args")
    if raw_args is None:
        return None
    args = _checkpoint_args_as_dict(raw_args)
    # Keypoint checkpoints carry num_keypoints_per_class; classify them as pose so it agrees
    # with task_of_model_type('rfdetr-keypoint-preview') == TASK_POSE and the upload proceeds.
    if args.get("num_keypoints_per_class"):
        return TASK_POSE
    segmentation_head = args.get("segmentation_head")
    if segmentation_head is True:
        return TASK_SEG
    if segmentation_head is False:
        return TASK_DET
    return None


def _rfdetr_checkpoint_pe_size(checkpoint: Any) -> int | None:
    """Return an RF-DETR checkpoint's position-encoding grid size (tokens per side).

    Prefers the explicit ``positional_encoding_size`` arg, then ``resolution //
    patch_size``, then derives it from the backbone ``position_embeddings`` tensor
    (``grid² + 1`` tokens). Returns None when the geometry cannot be determined.
    """
    if not isinstance(checkpoint, dict):
        return None
    args = _checkpoint_args_as_dict(checkpoint.get("args"))

    pe = args.get("positional_encoding_size")
    if isinstance(pe, int) and pe > 0:
        return pe
    resolution = args.get("resolution")
    patch_size = args.get("patch_size")
    if isinstance(resolution, int) and isinstance(patch_size, int) and patch_size > 0:
        return resolution // patch_size

    state_dict = checkpoint.get("model")
    if isinstance(state_dict, dict):
        for key, tensor in state_dict.items():
            if not key.endswith("position_embeddings"):
                continue
            shape = getattr(tensor, "shape", None)
            if shape is not None and len(shape) == 3:
                grid = math.isqrt(int(shape[1]) - 1)
                if grid > 0 and grid * grid == int(shape[1]) - 1:
                    return grid
    return None


def _resolve_rfdetr_variant(
    model_type: str,
    checkpoint: Any,
    warnings: list[str],
    allow_mismatch: bool = False,
) -> str:
    """Check an RF-DETR model_type's size variant against the checkpoint geometry.

    Roboflow rebuilds the architecture from ``model_type`` at the variant's default
    resolution before loading the weights, so a variant whose position-encoding grid
    differs from the checkpoint fails conversion with a ``position_embeddings`` size
    mismatch. Two cases:

    * The checkpoint's grid matches a *different* known variant (e.g. uploaded as
      ``rfdetr-seg-nano`` but the grid is ``rfdetr-seg-small``) — a high-confidence
      mislabel. Raise, naming the variant that fits, so the caller can confirm.
    * The grid matches *no* known variant — likely a custom training resolution. We
      cannot tell whether the backend supports it, so warn and proceed rather than
      block a possibly-valid upload.

    ``allow_mismatch`` always proceeds with the requested variant (with a warning).
    The detection-vs-segmentation task is handled separately and is not changed
    here. Returns the resolved model_type.
    """
    expected = RFDETR_POSITIONAL_ENCODING_SIZE.get(model_type)
    actual = _rfdetr_checkpoint_pe_size(checkpoint)
    if actual is None or expected is None or actual == expected:
        return model_type

    task = task_of_model_type(model_type)
    match = next(
        (
            name
            for name, grid in RFDETR_POSITIONAL_ENCODING_SIZE.items()
            if grid == actual and task_of_model_type(name) == task
        ),
        None,
    )

    if match is not None and not allow_mismatch:
        raise SizeMismatchError(
            f"You specified model_type '{model_type}' (a {expected}x{expected} "
            f"position-encoding grid), but the checkpoint was trained with "
            f"{actual}x{actual}, which matches '{match}'. They don't match, so "
            f"Roboflow's weight conversion would fail to load the backbone position "
            f"embeddings. Upload as '{match}', or set allow_size_mismatch=True to "
            f"upload '{model_type}' exactly as specified.",
            requested=model_type,
            detected=match,
        )

    if match is None:
        warnings.append(
            f"model_type '{model_type}' expects a {expected}x{expected} position-encoding "
            f"grid, but the checkpoint is {actual}x{actual} and matches no known RF-DETR "
            f"variant (it may use a custom training resolution). Packaging as "
            f"'{model_type}'; Roboflow's conversion may reject it if it rebuilds at the "
            f"variant's default resolution."
        )
    else:
        warnings.append(
            f"model_type '{model_type}' expects a {expected}x{expected} grid, but the "
            f"checkpoint is {actual}x{actual} (matches '{match}'). Packaging as "
            f"'{model_type}' as requested."
        )
    return model_type


def _find_rfdetr_checkpoint(model_path: Path, filename: str, warnings: list[str]) -> Path:
    """Locate the rf-detr checkpoint.

    An explicitly-requested ``filename`` (anything other than the default) must
    exist: falling back to a different checkpoint on a typo would silently
    package the wrong weights. Only the default path falls back to discovering
    the first top-level .pt/.pth file (sorted for determinism), preserving how
    rf-detr uploads located the checkpoint before ``filename`` was honored.
    """
    requested_file = model_path / filename
    if requested_file.exists():
        return requested_file

    if filename != DEFAULT_WEIGHTS_FILENAME:
        raise MissingFileError(
            f"RF-DETR weights file '{requested_file}' was not found. Set filename to the "
            f"checkpoint's exact .pt or .pth path relative to model_path."
        )

    discovered = sorted(path for path in model_path.iterdir() if path.is_file() and path.suffix in {".pt", ".pth"})
    if not discovered:
        raise MissingFileError(
            f"No .pt or .pth checkpoint found in '{model_path}' (and '{requested_file}' does not exist)."
        )
    if len(discovered) > 1:
        others = ", ".join(path.name for path in discovered)
        warnings.append(
            f"Weights file '{requested_file}' was not found and '{model_path}' holds multiple "
            f"checkpoints ({others}); packaging '{discovered[0].name}'. Set filename to pick a "
            "specific checkpoint if that is not the one you want."
        )
    else:
        warnings.append(
            f"Weights file '{requested_file}' was not found; using discovered checkpoint "
            f"'{discovered[0].name}' instead."
        )
    return discovered[0]


def _write_rfdetr_class_names(model_path: Path, build_dir: Path, checkpoint: Any) -> Path:
    class_names_path = model_path / "class_names.txt"
    if class_names_path.exists():
        class_names = class_names_path.read_text().splitlines()
    else:
        raw_args = checkpoint.get("args") if isinstance(checkpoint, dict) else None
        class_names = _checkpoint_args_as_dict(raw_args).get("class_names") or []
        if not class_names:
            raise MissingFileError(
                f"No class_names.txt file found in '{model_path}', and the RF-DETR "
                "checkpoint does not include args with class_names. This should only "
                "happen on rfdetr models trained before version 1.1.0. Create "
                "class_names.txt with one class per line or retrain with a newer "
                "rfdetr library."
            )

    if "background_class83422" not in class_names:
        class_names = ["background_class83422", *class_names]
    output_path = build_dir / "class_names.txt"
    output_path.write_text("\n".join(class_names) + "\n")
    return output_path


def _is_ptl_checkpoint(checkpoint: Any) -> bool:
    """True if the checkpoint is a raw PyTorch-Lightning rf-detr checkpoint dict."""
    return isinstance(checkpoint, dict) and "pytorch-lightning_version" in checkpoint


def _require_rfdetr() -> Any:
    """Import ``rfdetr`` and verify it ships the upload-bundle helpers.

    Raises :class:`MissingDependencyError` (a ModelPackagingError, so callers see
    an actionable 400 rather than an opaque server error) when ``rfdetr`` is
    missing or too old to export a Roboflow upload bundle.
    """
    try:
        import rfdetr
    except ImportError as exc:
        raise MissingDependencyError(
            "The 'rfdetr' package is required to package raw PyTorch-Lightning rf-detr "
            f"checkpoints. Install it with `pip install 'rfdetr>={RFDETR_MIN_VERSION}'`."
        ) from exc

    if not hasattr(rfdetr.RFDETR, "export_for_roboflow"):
        raise MissingDependencyError(
            "The installed 'rfdetr' is too old to package raw PyTorch-Lightning rf-detr "
            f"checkpoints. Upgrade it with `pip install --upgrade 'rfdetr>={RFDETR_MIN_VERSION}'`."
        )
    return rfdetr


def _process_rfdetr(
    model_type: str,
    model_path: Path,
    filename: str,
    build_dir: Path,
    allow_size_mismatch: bool,
) -> tuple[Path, str, list[str]]:
    if model_type not in SUPPORTED_RFDETR_TYPES:
        raise UnsupportedModelError(
            f"Model type '{model_type}' is not supported for RF-DETR upload. "
            f"Supported types are: {', '.join(SUPPORTED_RFDETR_TYPES)}."
        )
    torch = _import_required_module("torch", "pip install torch")
    warnings: list[str] = []

    checkpoint_path = _find_rfdetr_checkpoint(model_path, filename, warnings)
    checkpoint = _load_checkpoint(torch, checkpoint_path, map_location="cpu")

    # Task detection + mismatch runs for every checkpoint shape, so a checkpoint whose
    # task disagrees with model_type (e.g. a keypoint checkpoint uploaded as 'rfdetr-base')
    # is rejected instead of packaged under the wrong task.
    detected_task = _detect_rfdetr_task(checkpoint)
    if detected_task and detected_task != task_of_model_type(model_type):
        raise TaskMismatchError(
            f"model_type '{model_type}' implies task '{task_of_model_type(model_type)}', "
            f"but the checkpoint is a '{detected_task}' RF-DETR model. Use a matching model_type."
        )

    if _is_ptl_checkpoint(checkpoint):
        # Raw PyTorch-Lightning checkpoint: let rf-detr rebuild a proper upload
        # bundle (weights.pt with args.resolution + class_names.txt) into build_dir,
        # so the caller's model_path is never mutated.
        rfdetr = _require_rfdetr()
        try:
            model = rfdetr.RFDETR.from_checkpoint(str(checkpoint_path))
        except ValueError:
            # Checkpoint lacks model_name/pretrain_weights signals; fall back to the
            # already-validated model_type to pick the RFDETR subclass.
            model_cls = getattr(rfdetr, _RFDETR_MODEL_TYPE_TO_CLASS[model_type])
            model = model_cls(pretrain_weights=str(checkpoint_path))
        model.export_for_roboflow(str(build_dir))  # writes weights.pt + class_names.txt
    else:
        # Roboflow's server-side RF-DETR conversion reads checkpoint["args"] (the
        # class names, class count, and model config). A bare inference state_dict —
        # e.g. {"model": <weights>} with nothing else — would otherwise package and
        # upload fine, then fail conversion with an opaque KeyError: 'args'. Catch it
        # here so the caller gets an actionable error before uploading.
        if not isinstance(checkpoint, dict) or checkpoint.get("args") is None:
            raise ModelPackagingError(
                f"The RF-DETR checkpoint '{checkpoint_path.name}' is missing its 'args' "
                "metadata; it looks like a bare inference state_dict. Roboflow's weight "
                "conversion needs the full training checkpoint (args carries the class "
                "names, class count, and model config). Re-export the checkpoint from your "
                "training run, or download the deploy checkpoint from Roboflow."
            )

        model_type = _resolve_rfdetr_variant(model_type, checkpoint, warnings, allow_size_mismatch)

        weights_dest = build_dir / "weights.pt"
        # In the legacy deploy flow build_dir is model_path, so a checkpoint already
        # named weights.pt is its own destination; copying would raise SameFileError.
        if checkpoint_path.resolve() != weights_dest.resolve():
            shutil.copy(checkpoint_path, weights_dest)
        _write_rfdetr_class_names(model_path, build_dir, checkpoint)

    archive_path = build_dir / "roboflow_deploy.zip"
    _write_zip(
        archive_path,
        [
            (build_dir / "weights.pt", "weights.pt", True),
            (model_path / "results.csv", "results.csv", False),
            (model_path / "results.png", "results.png", False),
            (model_path / "model_artifacts.json", "model_artifacts.json", False),
            (build_dir / "class_names.txt", "class_names.txt", False),
        ],
    )
    return archive_path, model_type, warnings


def _process_huggingface(
    model_type: str,
    model_path: Path,
    build_dir: Path,
) -> tuple[Path, str, list[str]]:
    if model_type not in SUPPORTED_HUGGINGFACE_TYPES:
        raise UnsupportedModelError(
            f"Model type '{model_type}' is not supported for this type of upload. "
            f"Supported types are: {', '.join(SUPPORTED_HUGGINGFACE_TYPES)}."
        )

    model_files = [path for path in model_path.iterdir() if path.is_file()]
    safetensors_files = [path for path in model_files if path.suffix == ".safetensors"]
    npz_file = next((path for path in model_files if path.suffix == ".npz"), None)
    if safetensors_files:
        required = {
            "preprocessor_config.json",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "tokenizer.json",
        }
        missing = sorted(required - {path.name for path in model_files})
        if missing:
            raise MissingFileError(f"Missing files required for a PyTorch {model_type} upload: {', '.join(missing)}.")
        files_to_deploy = model_files
    elif npz_file is not None:
        files_to_deploy = [npz_file]
    else:
        raise MissingFileError(f"No .npz or .safetensors model file found in '{model_path}'.")

    archive_path = build_dir / "roboflow_deploy.tar"
    with tarfile.open(archive_path, "w") as tar:
        for path in files_to_deploy:
            tar.add(path, arcname=path.name)
    return archive_path, model_type, []


def _process_yolonas(
    model_type: str,
    model_path: Path,
    filename: str,
    build_dir: Path,
) -> tuple[Path, str, list[str]]:
    if model_type != "yolonas":
        raise UnsupportedModelError(
            f"Model type '{model_type}' is not supported for YOLO-NAS upload. The only "
            "supported YOLO-NAS type is 'yolonas'; the architecture size goes in opt.yaml "
            "as 'architecture: yolo_nas_s' (or _m / _l)."
        )
    torch = _import_required_module("torch", "pip install torch")
    weights_path = model_path / filename
    if not weights_path.exists():
        raise MissingFileError(f"Model weights file '{weights_path}' was not found.")

    checkpoint = _load_checkpoint(torch, weights_path, map_location="cpu")
    # A SuperGradients YOLO-NAS checkpoint carries processing_params.class_names.
    # A bare state_dict (e.g. torch.save(net.state_dict())) lacks it and would
    # otherwise raise a raw KeyError/TypeError instead of an actionable error.
    processing_params = checkpoint.get("processing_params") if isinstance(checkpoint, dict) else None
    class_names = processing_params.get("class_names") if isinstance(processing_params, dict) else None
    if not class_names:
        raise ModelPackagingError(
            f"The YOLO-NAS checkpoint '{weights_path.name}' is missing "
            "processing_params.class_names; it looks like a bare state_dict. Provide the "
            "full training checkpoint saved by SuperGradients."
        )
    opt_path = model_path / "opt.yaml"
    if not opt_path.exists():
        raise MissingFileError(
            f"You must create an opt.yaml file at '{opt_path}' of the format:\n"
            f"imgsz: <resolution of model>\n"
            f"batch_size: <batch size of inference model>\n"
            f"architecture: <one of [yolo_nas_s, yolo_nas_m, yolo_nas_l]. "
            f"s, m, l refer to small, medium, large architecture sizes, respectively>\n"
        )
    with opt_path.open() as stream:
        opts = yaml.safe_load(stream) or {}
    missing = [key for key in ("imgsz", "batch_size", "architecture") if key not in opts]
    if missing:
        raise ModelPackagingError(f"{opt_path} lacks required keys: {', '.join(missing)}.")

    model_artifacts = {
        "names": class_names,
        "nc": len(class_names),
        "args": {
            "imgsz": opts["imgsz"],
            "batch": opts["batch_size"],
            "architecture": opts["architecture"],
        },
        "model_type": model_type,
    }
    (build_dir / "model_artifacts.json").write_text(json.dumps(model_artifacts))
    shutil.copy(weights_path, build_dir / "state_dict.pt")

    archive_path = build_dir / "roboflow_deploy.zip"
    _write_zip(
        archive_path,
        [
            (model_path / "results.json", "results.json", False),
            (model_path / "results.png", "results.png", False),
            (build_dir / "model_artifacts.json", "model_artifacts.json", True),
            (build_dir / "state_dict.pt", "state_dict.pt", True),
        ],
    )
    return archive_path, model_type, []


def _write_zip(
    archive_path: Path,
    files: list[tuple[Path, str, bool]],
) -> None:
    with zipfile.ZipFile(archive_path, "w") as zip_file:
        for path, arcname, required in files:
            if path.exists():
                zip_file.write(path, arcname=arcname, compress_type=zipfile.ZIP_DEFLATED)
            elif required:
                raise MissingFileError(f"Required upload artifact '{path}' was not found.")


def get_classnames_txt_for_rfdetr(model_path: str, pt_file: str, checkpoint=None):
    """Legacy rf-detr class-names helper, kept for backwards compatibility.

    Writes (and mutates) ``class_names.txt`` inside ``model_path``. The packaging
    flow uses :func:`_write_rfdetr_class_names` instead, which leaves the source
    directory untouched.
    """
    class_names_path = os.path.join(model_path, "class_names.txt")
    if os.path.exists(class_names_path):
        maybe_prepend_dummy_class(class_names_path)
        return class_names_path

    if checkpoint is None:
        import torch

        checkpoint = torch.load(os.path.join(model_path, pt_file), map_location="cpu", weights_only=False)
    raw_args = checkpoint["args"]
    # args may be a plain dict in some checkpoints
    args = raw_args if isinstance(raw_args, dict) else vars(raw_args)
    if "class_names" in args:
        with open(class_names_path, "w") as f:
            for class_name in args["class_names"]:
                f.write(class_name + "\n")
        maybe_prepend_dummy_class(class_names_path)
        return class_names_path

    raise MissingFileError(
        f"No class_names.txt file found in model path {model_path}.\n"
        f"This should only happen on rfdetr models trained before version 1.1.0.\n"
        f"Please re-train your model with the latest version of the rfdetr library, or\n"
        f"please create a class_names.txt file in the model path with the class names\n"
        f"in new lines in the order of the classes in the model.\n"
    )


def maybe_prepend_dummy_class(class_name_file: str):
    with open(class_name_file) as f:
        class_names = f.readlines()

    dummy_class = "background_class83422\n"
    if dummy_class not in class_names:
        class_names.insert(0, dummy_class)
        with open(class_name_file, "w") as f:
            f.writelines(class_names)
