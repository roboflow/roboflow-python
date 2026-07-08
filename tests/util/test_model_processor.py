import json
import os
import sys
import tarfile
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from roboflow.config import TASK_CLS, TASK_DET, TASK_OBB, TASK_POSE, TASK_SEG, TASK_SEM
from roboflow.util import model_processor
from roboflow.util.model_processor import (
    _RFDETR_MODEL_TYPE_TO_CLASS,
    MissingFileError,
    ModelPackagingError,
    SizeMismatchError,
    TaskMismatchError,
    UnsupportedModelError,
    _checkpoint_args_as_dict,
    _detect_rfdetr_task,
    _detect_yolo_task,
    _filtered_args,
    _infer_yolo_size,
    _is_ptl_checkpoint,
    _legacy_yolo_args,
    _require_rfdetr,
    _resolve_rfdetr_variant,
    _resolve_yolo_size,
    _rfdetr_checkpoint_pe_size,
    _write_rfdetr_class_names,
    get_classnames_txt_for_rfdetr,
    package_custom_weights,
    package_custom_weights_interactive,
    process,
    task_of_model_type,
    validate_model_type_for_project,
)


class _FakeModel:
    """Stand-in for an Ultralytics model_instance; only __class__.__name__ matters."""


def _make_fake(name: str):
    return type(name, (_FakeModel,), {})()


class TaskOfModelTypeTest(unittest.TestCase):
    def test_detect_defaults(self):
        self.assertEqual(task_of_model_type("yolov11"), TASK_DET)
        self.assertEqual(task_of_model_type("rfdetr-base"), TASK_DET)
        self.assertEqual(task_of_model_type("rfdetr-medium"), TASK_DET)
        self.assertEqual(task_of_model_type("yolov8"), TASK_DET)

    def test_segment(self):
        self.assertEqual(task_of_model_type("yolov11-seg"), TASK_SEG)
        self.assertEqual(task_of_model_type("rfdetr-seg-medium"), TASK_SEG)
        self.assertEqual(task_of_model_type("yolov7-seg"), TASK_SEG)

    def test_pose(self):
        self.assertEqual(task_of_model_type("yolov11-pose"), TASK_POSE)

    def test_classify(self):
        self.assertEqual(task_of_model_type("yolov11-cls"), TASK_CLS)

    def test_semantic(self):
        self.assertEqual(task_of_model_type("yolo26-sem"), TASK_SEM)

    def test_obb(self):
        self.assertEqual(task_of_model_type("yolov11-obb"), TASK_OBB)


class DetectYoloTaskTest(unittest.TestCase):
    def test_ultralytics_class_names(self):
        cases = {
            "SegmentationModel": TASK_SEG,
            "PoseModel": TASK_POSE,
            "ClassificationModel": TASK_CLS,
            "OBBModel": TASK_OBB,
            "DetectionModel": TASK_DET,
        }
        for cls_name, expected in cases.items():
            self.assertEqual(_detect_yolo_task(_make_fake(cls_name)), expected, cls_name)

    def test_semantic_segmentation_model(self):
        self.assertEqual(_detect_yolo_task(_make_fake("SemanticSegmentationModel")), TASK_SEM)

    def test_unrecognized_returns_none(self):
        self.assertIsNone(_detect_yolo_task(_make_fake("SomeOtherModel")))
        self.assertIsNone(_detect_yolo_task(None))


class DetectRfdetrTaskTest(unittest.TestCase):
    def test_segmentation_model_names(self):
        for name in ("RFDETRSegNano", "RFDETRSegSmall", "RFDETRSegMedium", "RFDETRSegLarge"):
            self.assertEqual(_detect_rfdetr_task({"model_name": name}), TASK_SEG, name)

    def test_detection_model_names(self):
        for name in ("RFDETRNano", "RFDETRSmall", "RFDETRMedium", "RFDETRLarge", "RFDETRXLarge"):
            self.assertEqual(_detect_rfdetr_task({"model_name": name}), TASK_DET, name)

    def test_keypoint_model_name_returns_pose(self):
        # Keypoint checkpoints are unsupported; classifying them as pose lets the
        # model_type task check reject them instead of uploading them as detection.
        self.assertEqual(_detect_rfdetr_task({"model_name": "RFDETRKeypointPreview"}), TASK_POSE)

    def test_segmentation_head_fallback(self):
        # Roboflow-hosted rf-detr .pt downloads lack `model_name` but always carry
        # `args.segmentation_head`. Cover both namespace and dict shapes.
        self.assertEqual(_detect_rfdetr_task({"args": SimpleNamespace(segmentation_head=True)}), TASK_SEG)
        self.assertEqual(_detect_rfdetr_task({"args": SimpleNamespace(segmentation_head=False)}), TASK_DET)
        self.assertEqual(_detect_rfdetr_task({"args": {"segmentation_head": True}}), TASK_SEG)
        self.assertEqual(_detect_rfdetr_task({"args": {"segmentation_head": False}}), TASK_DET)

    def test_model_name_preferred_over_args(self):
        # When both are present, model_name wins (matches rf-detr's loader).
        ckpt = {"model_name": "RFDETRNano", "args": SimpleNamespace(segmentation_head=True)}
        self.assertEqual(_detect_rfdetr_task(ckpt), TASK_DET)

    def test_unrecognized_returns_none(self):
        self.assertIsNone(_detect_rfdetr_task({}))
        self.assertIsNone(_detect_rfdetr_task({"model_name": None}))
        self.assertIsNone(_detect_rfdetr_task({"args": SimpleNamespace(other=1)}))

    def test_scalar_args_do_not_raise_typeerror(self):
        # A corrupt checkpoint storing args as a bare scalar must not escape the
        # ModelPackagingError contract with a raw vars() TypeError.
        self.assertIsNone(_detect_rfdetr_task({"args": 640}))


class CheckpointArgsAsDictTest(unittest.TestCase):
    def test_coerces_dict_namespace_none_and_scalar(self):
        self.assertEqual(_checkpoint_args_as_dict({"a": 1}), {"a": 1})
        self.assertEqual(_checkpoint_args_as_dict(SimpleNamespace(a=1)), {"a": 1})
        self.assertEqual(_checkpoint_args_as_dict(None), {})
        self.assertEqual(_checkpoint_args_as_dict(640), {})
        self.assertEqual(_checkpoint_args_as_dict(["not", "a", "dict"]), {})


class FilteredArgsTest(unittest.TestCase):
    def test_keeps_only_upload_keys_from_dict_or_namespace(self):
        self.assertEqual(
            _filtered_args({"model": "m", "imgsz": 640, "batch": 8, "lr0": 0.01}),
            {"model": "m", "imgsz": 640, "batch": 8},
        )
        self.assertEqual(_filtered_args(SimpleNamespace(imgsz=320, batch=4, extra=1)), {"imgsz": 320, "batch": 4})

    def test_scalar_or_none_args_do_not_raise(self):
        # A corrupt .args must coerce to {} instead of raising a raw TypeError.
        self.assertEqual(_filtered_args(None), {})
        self.assertEqual(_filtered_args(640), {})


class GetClassnamesTxtForRfdetrTest(unittest.TestCase):
    def _classnames(self, args):
        with tempfile.TemporaryDirectory() as model_path:
            get_classnames_txt_for_rfdetr(model_path, "weights.pt", checkpoint={"args": args})
            with open(os.path.join(model_path, "class_names.txt")) as f:
                return f.read().splitlines()

    def test_dict_args(self):
        self.assertEqual(self._classnames({"class_names": ["cat", "dog"]}), ["background_class83422", "cat", "dog"])

    def test_namespace_args(self):
        self.assertEqual(
            self._classnames(SimpleNamespace(class_names=["cat", "dog"])),
            ["background_class83422", "cat", "dog"],
        )


class ValidateModelTypeForProjectTest(unittest.TestCase):
    def test_rejects_detection_for_classification(self):
        with self.assertRaises(TaskMismatchError) as ctx:
            validate_model_type_for_project("yolov8", "classification", "widgets")
        self.assertIn("classification", str(ctx.exception))
        self.assertIn("task 'cls'", str(ctx.exception))

    def test_task_mismatch_is_a_value_error(self):
        # Callers that caught the historical ValueError keep working.
        with self.assertRaises(ValueError):
            validate_model_type_for_project("yolov8", "classification", "widgets")

    def test_unknown_project_type_is_ignored(self):
        validate_model_type_for_project("yolov8", "some-new-type", "widgets")


class LegacyYoloArgsTest(unittest.TestCase):
    def test_reports_missing_batch_size(self):
        with self.assertRaises(ModelPackagingError) as ctx:
            _legacy_yolo_args({"imgsz": 640}, Path("opt.yaml"))
        self.assertIn("batch_size", str(ctx.exception))

    def test_reports_missing_image_size(self):
        with self.assertRaises(ModelPackagingError) as ctx:
            _legacy_yolo_args({"batch_size": 8}, Path("opt.yaml"))
        self.assertIn("imgsz", str(ctx.exception))

    def test_accepts_either_image_size_key(self):
        self.assertEqual(_legacy_yolo_args({"imgsz": 640, "batch_size": 8}, Path("x")), {"imgsz": 640, "batch": 8})
        self.assertEqual(_legacy_yolo_args({"img_size": 416, "batch_size": 4}, Path("x")), {"imgsz": 416, "batch": 4})


class _FakeYoloModel:
    def __init__(self, yaml):
        self.yaml = yaml


class InferYoloSizeTest(unittest.TestCase):
    def test_from_depth_width_multiples(self):
        model = _FakeYoloModel({"depth_multiple": 0.33, "width_multiple": 0.25})
        self.assertEqual(_infer_yolo_size(model), "n")

    def test_explicit_scale_letter_wins(self):
        model = _FakeYoloModel({"scale": "m", "depth_multiple": 0.67, "width_multiple": 0.75})
        self.assertEqual(_infer_yolo_size(model), "m")

    def test_unknown_returns_none(self):
        self.assertIsNone(_infer_yolo_size(_FakeYoloModel({})))


class ResolveYoloSizeTest(unittest.TestCase):
    def test_fills_bare_family_from_architecture(self):
        warnings: list = []
        model = _FakeYoloModel({"depth_multiple": 0.33, "width_multiple": 0.25})
        self.assertEqual(_resolve_yolo_size("yolov8", model, warnings), "yolov8n")
        self.assertTrue(warnings and "Inferred model size 'yolov8n'" in warnings[0])

    def test_preserves_task_suffix_when_filling(self):
        model = _FakeYoloModel({"depth_multiple": 0.33, "width_multiple": 0.50})
        self.assertEqual(_resolve_yolo_size("yolov8-seg", model, []), "yolov8s-seg")

    def test_raises_when_size_cannot_be_inferred(self):
        with self.assertRaises(SizeMismatchError) as ctx:
            _resolve_yolo_size("yolov8", _FakeYoloModel({}), [])
        self.assertIn("could not be inferred", str(ctx.exception))
        self.assertIn("yolov8n", str(ctx.exception))

    def test_raises_on_declared_size_conflict(self):
        model = _FakeYoloModel({"depth_multiple": 0.33, "width_multiple": 0.25})
        with self.assertRaises(SizeMismatchError) as ctx:
            _resolve_yolo_size("yolov8m", model, [])
        self.assertIn("yolov8n", str(ctx.exception))
        self.assertIn("allow_size_mismatch=True", str(ctx.exception))
        self.assertEqual(ctx.exception.requested, "yolov8m")
        self.assertEqual(ctx.exception.detected, "yolov8n")

    def test_allow_mismatch_keeps_declared_size(self):
        warnings: list = []
        model = _FakeYoloModel({"depth_multiple": 0.33, "width_multiple": 0.25})
        self.assertEqual(_resolve_yolo_size("yolov8m", model, warnings, allow_mismatch=True), "yolov8m")
        self.assertTrue(warnings and "as requested" in warnings[0])

    def test_keeps_user_size_when_not_inferable(self):
        warnings: list = []
        self.assertEqual(_resolve_yolo_size("yolov8m", _FakeYoloModel({}), warnings), "yolov8m")
        self.assertEqual(warnings, [])

    def test_keeps_matching_sized_type_without_warning(self):
        warnings: list = []
        model = _FakeYoloModel({"depth_multiple": 0.33, "width_multiple": 0.25})
        self.assertEqual(_resolve_yolo_size("yolov8n", model, warnings), "yolov8n")
        self.assertEqual(warnings, [])

    def test_bare_family_uninferable_raises_by_default(self):
        with self.assertRaises(SizeMismatchError):
            _resolve_yolo_size("yolov10", _FakeYoloModel({}), [])

    def test_bare_family_uninferable_proceeds_under_allow_mismatch(self):
        # The interactive retry sets allow_mismatch=True after the user confirms;
        # this branch must then converge (return) rather than raise forever.
        warnings: list = []
        self.assertEqual(
            _resolve_yolo_size("yolov10", _FakeYoloModel({}), warnings, allow_mismatch=True),
            "yolov10",
        )
        self.assertTrue(warnings and "bare family name" in warnings[0])


class ResolveRfdetrVariantTest(unittest.TestCase):
    def test_raises_on_size_conflict_naming_the_fit(self):
        checkpoint = {"args": {"resolution": 384, "patch_size": 12}}
        with self.assertRaises(SizeMismatchError) as ctx:
            _resolve_rfdetr_variant("rfdetr-seg-nano", checkpoint, [])
        self.assertIn("rfdetr-seg-small", str(ctx.exception))
        self.assertIn("32x32", str(ctx.exception))
        self.assertIn("allow_size_mismatch=True", str(ctx.exception))

    def test_allow_mismatch_keeps_requested_variant(self):
        warnings: list = []
        checkpoint = {"args": {"resolution": 384, "patch_size": 12}}
        resolved = _resolve_rfdetr_variant("rfdetr-seg-nano", checkpoint, warnings, allow_mismatch=True)
        self.assertEqual(resolved, "rfdetr-seg-nano")
        self.assertTrue(warnings and "as requested" in warnings[0])

    def test_keeps_matching_grid_without_warning(self):
        warnings: list = []
        checkpoint = {"args": {"positional_encoding_size": 32}}
        self.assertEqual(_resolve_rfdetr_variant("rfdetr-seg-small", checkpoint, warnings), "rfdetr-seg-small")
        self.assertEqual(warnings, [])

    def test_warns_but_allows_custom_resolution(self):
        warnings: list = []
        resolved = _resolve_rfdetr_variant("rfdetr-seg-nano", {"args": {"positional_encoding_size": 99}}, warnings)
        self.assertEqual(resolved, "rfdetr-seg-nano")
        self.assertTrue(warnings and "matches no known RF-DETR variant" in warnings[0])

    def test_pe_size_derived_from_position_embeddings_tensor(self):
        class FakeTensor:
            shape = (1, 1025, 384)

        checkpoint = {"model": {"backbone.0.encoder.encoder.embeddings.position_embeddings": FakeTensor()}}
        self.assertEqual(_rfdetr_checkpoint_pe_size(checkpoint), 32)


class WriteRfdetrClassNamesTest(unittest.TestCase):
    def test_fails_cleanly_without_checkpoint_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(MissingFileError) as ctx:
                _write_rfdetr_class_names(Path(tmp), Path(tmp), checkpoint={})
        self.assertIn("does not include args with class_names", str(ctx.exception))

    def test_does_not_mutate_existing_class_names_file(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as build:
            source = Path(tmp) / "class_names.txt"
            source.write_text("cat\ndog\n")
            output = _write_rfdetr_class_names(Path(tmp), Path(build), checkpoint={})
            self.assertEqual(source.read_text(), "cat\ndog\n")
            self.assertEqual(
                output.read_text().splitlines(),
                ["background_class83422", "cat", "dog"],
            )


def _fake_torch(load_result, calls=None):
    module = types.ModuleType("torch")

    def load(path, **kwargs):
        if calls is not None:
            calls.append((Path(path), kwargs))
        return load_result

    def save(obj, path):
        Path(path).write_bytes(b"fake-state-dict")

    module.load = load
    module.save = save
    return module


def _import_patch(modules):
    def _import(module_name, install_hint):
        return modules[module_name]

    return mock.patch.object(model_processor, "_import_required_module", side_effect=_import)


def _write_yolonas_inputs(model_dir: Path):
    weights = model_dir / "weights" / "best.pt"
    weights.parent.mkdir()
    weights.write_bytes(b"checkpoint")
    (model_dir / "opt.yaml").write_text("imgsz: 640\nbatch_size: 8\narchitecture: yolo_nas_s\n")


class PackageCustomWeightsTest(unittest.TestCase):
    """Contract tests for the public non-interactive helper."""

    def _package_yolonas(self, model_dir: Path, **kwargs):
        calls: list = []
        torch = _fake_torch({"processing_params": {"class_names": ["widget"]}}, calls)
        with _import_patch({"torch": torch}):
            bundle = package_custom_weights("yolonas", str(model_dir), **kwargs)
        return bundle, calls

    def test_never_prompts_or_exits(self):
        prompt_guard = mock.patch(
            "builtins.input", side_effect=AssertionError("package_custom_weights must not prompt")
        )
        exit_guard = mock.patch.object(sys, "exit", side_effect=AssertionError("package_custom_weights must not exit"))
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            _write_yolonas_inputs(model_dir)
            with prompt_guard, exit_guard:
                bundle, calls = self._package_yolonas(model_dir)
            try:
                self.assertTrue(bundle.archive_path.exists())
                self.assertEqual(calls[0][1], {"weights_only": False, "map_location": "cpu"})
            finally:
                bundle.cleanup()

    def test_does_not_write_into_model_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            _write_yolonas_inputs(model_dir)
            before = sorted(path for path in model_dir.rglob("*"))
            bundle, _ = self._package_yolonas(model_dir)
            try:
                self.assertEqual(sorted(path for path in model_dir.rglob("*")), before)
                self.assertNotEqual(bundle.build_dir, model_dir)
                self.assertTrue(bundle.owns_build_dir)
            finally:
                bundle.cleanup()
            self.assertFalse(bundle.build_dir.exists())

    def test_explicit_build_dir_is_used_and_not_cleaned_up(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as build:
            model_dir = Path(tmp)
            _write_yolonas_inputs(model_dir)
            bundle, _ = self._package_yolonas(model_dir, build_dir=build)
            self.assertEqual(bundle.build_dir, Path(build).resolve())
            self.assertFalse(bundle.owns_build_dir)
            bundle.cleanup()
            self.assertTrue(bundle.archive_path.exists())

    def test_owned_build_dir_is_removed_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as fake_build:
            with mock.patch.object(model_processor.tempfile, "mkdtemp", return_value=fake_build):
                with self.assertRaises(UnsupportedModelError):
                    package_custom_weights("not-a-model", tmp)
            self.assertFalse(Path(fake_build).exists())

    def test_missing_model_path_raises_missing_file(self):
        with self.assertRaises(MissingFileError):
            package_custom_weights("yolonas", "/nonexistent/path/for/test")

    def test_family_must_be_a_prefix_not_a_substring(self):
        # 'foo-yolov8n' merely contains a family token; the backend would
        # reject it after upload, so the gate must reject it up front.
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(UnsupportedModelError):
                package_custom_weights("foo-yolov8n", tmp)

    def test_yolonas_requires_exact_model_type(self):
        # Only 'yolonas' is valid; a suffixed typo like 'yolonas-foo' passes the
        # family prefix gate but must be rejected before upload, not by the backend.
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(UnsupportedModelError):
                package_custom_weights("yolonas-foo", tmp)

    def test_rfdetr_falls_back_to_discovered_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "other.pt").write_bytes(b"checkpoint")
            torch = _fake_torch({"args": {"class_names": ["widget"]}})
            with _import_patch({"torch": torch}):
                bundle = package_custom_weights("rfdetr-base", str(model_dir))
            try:
                self.assertTrue(any("other.pt" in warning for warning in bundle.warnings))
                with zipfile.ZipFile(bundle.archive_path) as archive:
                    self.assertIn("weights.pt", archive.namelist())
                    self.assertIn("class_names.txt", archive.namelist())
            finally:
                bundle.cleanup()

    def test_rfdetr_without_any_checkpoint_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            torch = _fake_torch({})
            with _import_patch({"torch": torch}):
                with self.assertRaises(MissingFileError):
                    package_custom_weights("rfdetr-base", tmp)

    def test_rfdetr_bare_state_dict_without_args_fails_before_upload(self):
        # A stripped inference checkpoint ({"model": state_dict} with no "args")
        # would package fine but fail Roboflow's server-side conversion with an
        # opaque KeyError: 'args'. Packaging must reject it up front.
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "weights.pt").write_bytes(b"checkpoint")
            torch = _fake_torch({"model": {"backbone.weight": object()}})
            with _import_patch({"torch": torch}):
                with self.assertRaises(ModelPackagingError) as ctx:
                    package_custom_weights("rfdetr-base", str(model_dir), filename="weights.pt")
        self.assertIn("args", str(ctx.exception))
        self.assertIn("state_dict", str(ctx.exception))

    def test_yolonas_bare_state_dict_without_class_names_fails_before_upload(self):
        # A bare YOLO-NAS state_dict lacks processing_params.class_names; it must
        # raise an actionable error instead of a raw KeyError/TypeError.
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "weights").mkdir()
            (model_dir / "weights" / "best.pt").write_bytes(b"checkpoint")
            torch = _fake_torch({"backbone.weight": object()})  # bare state_dict, no processing_params
            with _import_patch({"torch": torch}):
                with self.assertRaises(ModelPackagingError) as ctx:
                    package_custom_weights("yolonas", str(model_dir))
        self.assertIn("class_names", str(ctx.exception))
        self.assertIn("state_dict", str(ctx.exception))

    def test_rfdetr_explicit_missing_filename_does_not_fall_back(self):
        # A typo'd explicit filename must fail loudly, not silently upload a
        # different checkpoint that happens to be in the directory.
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "some_other_checkpoint.pth").write_bytes(b"checkpoint")
            torch = _fake_torch({"args": {"class_names": ["widget"]}})
            with _import_patch({"torch": torch}):
                with self.assertRaises(MissingFileError):
                    package_custom_weights("rfdetr-base", str(model_dir), filename="checkpoint_epoch50.pth")

    def test_rfdetr_legacy_deploy_layout_does_not_self_copy(self):
        # Legacy deploy passes build_dir=model_path with a top-level weights.pt;
        # copying weights.pt onto itself would raise shutil.SameFileError.
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "weights.pt").write_bytes(b"checkpoint")
            torch = _fake_torch({"args": {"class_names": ["widget"]}})
            with _import_patch({"torch": torch}):
                bundle = package_custom_weights(
                    "rfdetr-base", str(model_dir), filename="weights.pt", build_dir=model_dir
                )
            with zipfile.ZipFile(bundle.archive_path) as archive:
                self.assertIn("weights.pt", archive.namelist())

    def test_yolov8_full_flow_builds_artifacts(self):
        checkpoint_names = {1: "dog", 0: "cat"}

        class DetectionModel:
            names = checkpoint_names
            nc = 2
            yaml = {"nc": 2, "depth_multiple": 0.33, "width_multiple": 0.25}
            args = {"model": "yolov8n.yaml", "imgsz": 640, "batch": 16, "lr0": 0.01}

            def state_dict(self):
                return {"weight": b"w"}

        fake_ultralytics = types.ModuleType("ultralytics")
        fake_ultralytics.__version__ = "8.0.196"
        fake_torch = _fake_torch({"model": DetectionModel()})

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            weights = model_dir / "weights" / "best.pt"
            weights.parent.mkdir()
            weights.write_bytes(b"checkpoint")

            with (
                _import_patch({"torch": fake_torch, "ultralytics": fake_ultralytics}),
                mock.patch.dict(sys.modules, {"ultralytics": fake_ultralytics}),
            ):
                bundle = package_custom_weights("yolov8", str(model_dir))
            try:
                self.assertEqual(bundle.model_type, "yolov8n")
                self.assertTrue(any("Inferred model size 'yolov8n'" in warning for warning in bundle.warnings))
                with zipfile.ZipFile(bundle.archive_path) as archive:
                    artifacts = json.loads(archive.read("model_artifacts.json"))
                    self.assertIn("state_dict.pt", archive.namelist())
                self.assertEqual(artifacts["names"], ["cat", "dog"])
                self.assertEqual(artifacts["model_type"], "yolov8n")
                self.assertEqual(artifacts["ultralytics_version"], "8.0.196")
                self.assertEqual(artifacts["args"], {"model": "yolov8n.yaml", "imgsz": 640, "batch": 16})
            finally:
                bundle.cleanup()

    def test_rejects_absolute_filename(self):
        # A hosted caller (the MCP server) forwards filename verbatim; an absolute
        # path — POSIX or Windows-style, regardless of the packaging host's OS —
        # must be rejected rather than reading weights from outside model_path.
        with tempfile.TemporaryDirectory() as tmp:
            for bad in ("/etc/passwd", "C:\\Windows\\System32\\config"):
                with self.assertRaises(ModelPackagingError) as ctx:
                    package_custom_weights("yolonas", tmp, filename=bad)
                self.assertIn("absolute", str(ctx.exception))

    def test_rejects_filename_escaping_model_path(self):
        # '..' segments must not let the caller escape the model directory.
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            model_dir.mkdir()
            (Path(tmp) / "secret.pt").write_bytes(b"outside")
            with self.assertRaises(ModelPackagingError) as ctx:
                package_custom_weights("yolonas", str(model_dir), filename="../secret.pt")
        self.assertIn("outside model_path", str(ctx.exception))

    def _attempt_ultralytics_yolo(self, model_type, checkpoint):
        fake_ultralytics = types.ModuleType("ultralytics")
        fake_ultralytics.__version__ = "8.3.0"
        fake_torch = _fake_torch(checkpoint)
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            weights = model_dir / "weights" / "best.pt"
            weights.parent.mkdir()
            weights.write_bytes(b"checkpoint")
            with (
                _import_patch({"torch": fake_torch, "ultralytics": fake_ultralytics}),
                mock.patch.dict(sys.modules, {"ultralytics": fake_ultralytics}),
            ):
                # allow_dependency_mismatch keeps a version mismatch from masking the
                # completeness error under test.
                return package_custom_weights(model_type, str(model_dir), allow_dependency_mismatch=True)

    def test_yolov8_missing_nc_attr_raises_packaging_error(self):
        # A stripped Ultralytics checkpoint that lacks model.nc must raise a
        # ModelPackagingError, not a raw AttributeError (an opaque 500 in the MCP).
        class DetectionModel:
            names = {0: "cat"}
            yaml = {"nc": 1, "depth_multiple": 0.33, "width_multiple": 0.25}
            args = {"imgsz": 640}

            def state_dict(self):
                return {}

        with self.assertRaises(ModelPackagingError) as ctx:
            self._attempt_ultralytics_yolo("yolov8n", {"model": DetectionModel()})
        self.assertIn("nc", str(ctx.exception))

    def test_yolov8_missing_args_attr_raises_packaging_error(self):
        class DetectionModel:
            names = {0: "cat"}
            nc = 1
            yaml = {"nc": 1, "depth_multiple": 0.33, "width_multiple": 0.25}

            def state_dict(self):
                return {}

        with self.assertRaises(ModelPackagingError) as ctx:
            self._attempt_ultralytics_yolo("yolov8n", {"model": DetectionModel()})
        self.assertIn("args", str(ctx.exception))

    def test_yolov8_missing_yaml_attr_raises_packaging_error(self):
        class DetectionModel:
            names = {0: "cat"}
            nc = 1
            args = {"imgsz": 640}

            def state_dict(self):
                return {}

        with self.assertRaises(ModelPackagingError) as ctx:
            self._attempt_ultralytics_yolo("yolov8n", {"model": DetectionModel()})
        self.assertIn("yaml", str(ctx.exception))

    def test_yolov11_missing_train_args_raises_packaging_error(self):
        # yolov10/11/12/26 read args from checkpoint["train_args"]; a checkpoint
        # without it must fail with the ModelPackagingError contract.
        class DetectionModel:
            names = {0: "cat"}
            yaml = {"nc": 1, "scale": "n"}

            def state_dict(self):
                return {}

        with self.assertRaises(ModelPackagingError) as ctx:
            self._attempt_ultralytics_yolo("yolov11n", {"model": DetectionModel()})
        self.assertIn("train_args", str(ctx.exception))

    def test_yolov11_missing_nc_in_model_yaml_raises_packaging_error(self):
        class DetectionModel:
            names = {0: "cat"}
            yaml = {"scale": "n"}

            def state_dict(self):
                return {}

        with self.assertRaises(ModelPackagingError) as ctx:
            self._attempt_ultralytics_yolo("yolov11n", {"model": DetectionModel(), "train_args": {"imgsz": 640}})
        self.assertIn("nc", str(ctx.exception))

    def test_rfdetr_multiple_discovered_checkpoints_warns_which_used(self):
        # With no explicit filename and several checkpoints present, discovery is
        # ambiguous; the warning must name the candidates and which one was used.
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "alpha.pt").write_bytes(b"checkpoint")
            (model_dir / "beta.pth").write_bytes(b"checkpoint")
            torch = _fake_torch({"args": {"class_names": ["widget"]}})
            with _import_patch({"torch": torch}):
                bundle = package_custom_weights("rfdetr-base", str(model_dir))
            try:
                warning = "\n".join(bundle.warnings)
                self.assertIn("multiple", warning)
                self.assertIn("alpha.pt", warning)
                self.assertIn("beta.pth", warning)
            finally:
                bundle.cleanup()


class ProcessHuggingfaceTest(unittest.TestCase):
    """Packaging for HuggingFace-backed models (paligemma / florence-2)."""

    def test_missing_companion_files_raise_missing_file(self):
        # A safetensors checkpoint without its tokenizer/preprocessor companions
        # now raises MissingFileError instead of prompting for the files.
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "model.safetensors").write_bytes(b"weights")
            with mock.patch("builtins.input", side_effect=AssertionError("must not prompt")):
                with self.assertRaises(MissingFileError) as ctx:
                    package_custom_weights("florence-2-base", str(model_dir))
        message = str(ctx.exception)
        self.assertIn("tokenizer.json", message)
        self.assertIn("preprocessor_config.json", message)

    def test_no_model_file_raises_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "readme.txt").write_text("no weights here")
            with self.assertRaises(MissingFileError) as ctx:
                package_custom_weights("paligemma-3b-pt-224", str(model_dir))
        self.assertIn("safetensors", str(ctx.exception))

    def test_npz_checkpoint_packages_into_tar(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "model.npz").write_bytes(b"npz-bytes")
            bundle = package_custom_weights("paligemma-3b-pt-224", str(model_dir))
            try:
                self.assertTrue(bundle.archive_path.name.endswith(".tar"))
                with tarfile.open(bundle.archive_path) as tar:
                    self.assertIn("model.npz", tar.getnames())
            finally:
                bundle.cleanup()

    def test_unsupported_huggingface_type_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(UnsupportedModelError):
                package_custom_weights("florence-2-tiny", str(tmp))


class ProcessCompatTest(unittest.TestCase):
    """The legacy process() entry point keeps its historical contract."""

    def test_packages_into_model_path_and_returns_tuple(self):
        calls: list = []
        torch = _fake_torch({"processing_params": {"class_names": ["widget"]}}, calls)
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            _write_yolonas_inputs(model_dir)
            with _import_patch({"torch": torch}):
                zip_file_name, model_type = process("yolonas", str(model_dir), "weights/best.pt")

            self.assertEqual(zip_file_name, "roboflow_deploy.zip")
            self.assertEqual(model_type, "yolonas")
            # Historical side effects: artifacts and archive land in model_path.
            self.assertTrue((model_dir / "roboflow_deploy.zip").exists())
            self.assertTrue((model_dir / "model_artifacts.json").exists())
            self.assertTrue((model_dir / "state_dict.pt").exists())

    def test_prompts_and_retries_on_mismatch_like_before(self):
        error = model_processor.DependencyMismatchError(
            "wrong ultralytics",
            dependency="ultralytics",
            required="ultralytics==8.0.196",
            installed="8.3.0",
        )
        bundle = model_processor.ModelUploadBundle(
            archive_path=Path("roboflow_deploy.zip"),
            build_dir=Path("."),
            model_type="yolov8n",
        )
        outcomes = [error, bundle]

        def fake_package(*args, **kwargs):
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            self.assertTrue(kwargs["allow_dependency_mismatch"])
            return outcome

        with (
            mock.patch.object(model_processor, "package_custom_weights", side_effect=fake_package),
            mock.patch("builtins.input", return_value="y"),
            mock.patch("builtins.print"),
        ):
            zip_file_name, model_type = process("yolov8m", "/models", "weights/best.pt")

        self.assertEqual(zip_file_name, "roboflow_deploy.zip")
        self.assertEqual(model_type, "yolov8n")


class PackageCustomWeightsInteractiveTest(unittest.TestCase):
    def _bundle(self):
        return model_processor.ModelUploadBundle(
            archive_path=Path("roboflow_deploy.zip"),
            build_dir=Path("."),
            model_type="yolov8n",
            warnings=("some warning",),
        )

    def test_retries_with_size_override_on_confirmation(self):
        error = SizeMismatchError("size conflict", requested="yolov8m", detected="yolov8n")
        outcomes = [error, self._bundle()]

        def fake_package(*args, **kwargs):
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            self.assertTrue(kwargs["allow_size_mismatch"])
            return outcome

        with (
            mock.patch.object(model_processor, "package_custom_weights", side_effect=fake_package),
            mock.patch("builtins.input", return_value="y"),
            mock.patch("builtins.print"),
        ):
            bundle = package_custom_weights_interactive("yolov8m", "/models")
        self.assertEqual(bundle.model_type, "yolov8n")

    def test_reraises_when_user_declines(self):
        error = SizeMismatchError("size conflict", requested="yolov8m")
        with (
            mock.patch.object(model_processor, "package_custom_weights", side_effect=error),
            mock.patch("builtins.input", return_value="n"),
            mock.patch("builtins.print"),
        ):
            with self.assertRaises(SizeMismatchError):
                package_custom_weights_interactive("yolov8m", "/models")


class RfdetrModelTypeToClassTest(unittest.TestCase):
    def test_representative_mappings(self):
        self.assertEqual(_RFDETR_MODEL_TYPE_TO_CLASS["rfdetr-seg-medium"], "RFDETRSegMedium")
        self.assertEqual(_RFDETR_MODEL_TYPE_TO_CLASS["rfdetr-base"], "RFDETRBase")

    def test_keys_are_rfdetr_types_and_values_are_class_names(self):
        for model_type, class_name in _RFDETR_MODEL_TYPE_TO_CLASS.items():
            self.assertTrue(model_type.startswith("rfdetr-"), model_type)
            self.assertTrue(class_name.startswith("RFDETR"), class_name)
            # Segmentation types map to Seg classes (and detection types must not).
            self.assertEqual("seg" in model_type, "Seg" in class_name, model_type)


class IsPtlCheckpointTest(unittest.TestCase):
    def test_true_when_lightning_version_present(self):
        self.assertTrue(_is_ptl_checkpoint({"pytorch-lightning_version": "2.1.0", "args": {}}))

    def test_false_for_plain_checkpoint(self):
        self.assertFalse(_is_ptl_checkpoint({"args": {}, "model": {}}))

    def test_false_for_non_dict(self):
        self.assertFalse(_is_ptl_checkpoint(None))
        self.assertFalse(_is_ptl_checkpoint(SimpleNamespace(**{"pytorch-lightning_version": "2.1.0"})))


class _StubBundleModel:
    """Stub rf-detr model whose export_for_roboflow writes a dummy bundle on disk."""

    def __init__(self, class_names=("cat", "dog")):
        self.class_names = list(class_names)

    def export_for_roboflow(self, output_dir):
        (Path(output_dir) / "weights.pt").write_bytes(b"rebuilt-weights")
        (Path(output_dir) / "class_names.txt").write_text("\n".join(self.class_names) + "\n")


def _make_fake_rfdetr(*, from_checkpoint_raises=False, capabilities=True):
    """Build a fake ``rfdetr`` module for injection via sys.modules."""
    stub_model = _StubBundleModel()
    calls = {"from_checkpoint": 0, "fallback_constructed": 0, "constructor_kwargs": None}

    class _RFDETR:
        @staticmethod
        def from_checkpoint(path):
            calls["from_checkpoint"] += 1
            if from_checkpoint_raises:
                raise ValueError("cannot infer model class")
            return stub_model

    class _SizedModel(_StubBundleModel):
        def __init__(self, *, pretrain_weights):
            super().__init__()
            calls["fallback_constructed"] += 1
            calls["constructor_kwargs"] = {"pretrain_weights": pretrain_weights}

    module = SimpleNamespace()
    module.RFDETR = _RFDETR
    # The fallback resolves the subclass by name via _RFDETR_MODEL_TYPE_TO_CLASS,
    # e.g. "rfdetr-seg-medium" -> getattr(rfdetr, "RFDETRSegMedium").
    module.RFDETRSegMedium = _SizedModel
    if capabilities:
        _RFDETR.export_for_roboflow = _StubBundleModel.export_for_roboflow  # capability marker
    module._calls = calls
    return module


class RequireRfdetrTest(unittest.TestCase):
    def test_raises_when_not_installed(self):
        with mock.patch.dict(sys.modules, {"rfdetr": None}):
            with self.assertRaises(ModelPackagingError) as ctx:
                _require_rfdetr()
        self.assertIn("pip install", str(ctx.exception).lower())
        self.assertIn("rfdetr", str(ctx.exception).lower())

    def test_raises_when_capability_missing(self):
        fake = SimpleNamespace(RFDETR=type("RFDETR", (), {}))
        with mock.patch.dict(sys.modules, {"rfdetr": fake}):
            with self.assertRaises(ModelPackagingError) as ctx:
                _require_rfdetr()
        self.assertIn("upgrade", str(ctx.exception).lower())

    def test_returns_module_when_capable(self):
        fake = _make_fake_rfdetr()
        with mock.patch.dict(sys.modules, {"rfdetr": fake}):
            self.assertIs(_require_rfdetr(), fake)


class PackageRfdetrPtlTest(unittest.TestCase):
    """PyTorch-Lightning rf-detr checkpoints are rebuilt via rfdetr into build_dir."""

    def _package(self, model_type, fake_rfdetr, *, segmentation_head=False):
        with tempfile.TemporaryDirectory() as model_dir:
            (Path(model_dir) / "checkpoint_best_ema.pth").write_bytes(b"raw-ptl")
            ckpt = {
                "pytorch-lightning_version": "2.1.0",
                "args": {"segmentation_head": segmentation_head, "class_names": ["cat", "dog"]},
            }
            torch = _fake_torch(ckpt)
            with _import_patch({"torch": torch}), mock.patch.dict(sys.modules, {"rfdetr": fake_rfdetr}):
                bundle = package_custom_weights(model_type, model_dir, filename="checkpoint_best_ema.pth")
            try:
                with zipfile.ZipFile(bundle.archive_path) as archive:
                    names = archive.namelist()
            finally:
                bundle.cleanup()
            return bundle, names

    def test_from_checkpoint_success_produces_bundle(self):
        fake = _make_fake_rfdetr()
        bundle, names = self._package("rfdetr-base", fake)
        self.assertEqual(bundle.model_type, "rfdetr-base")
        self.assertEqual(fake._calls["from_checkpoint"], 1)
        self.assertEqual(fake._calls["fallback_constructed"], 0)
        self.assertIn("weights.pt", names)
        self.assertIn("class_names.txt", names)

    def test_from_checkpoint_valueerror_falls_back_to_model_type(self):
        fake = _make_fake_rfdetr(from_checkpoint_raises=True)
        bundle, names = self._package("rfdetr-seg-medium", fake, segmentation_head=True)
        self.assertEqual(bundle.model_type, "rfdetr-seg-medium")
        self.assertEqual(fake._calls["from_checkpoint"], 1)
        self.assertEqual(fake._calls["fallback_constructed"], 1)
        self.assertIn("weights.pt", names)

    def test_ptl_path_raises_when_rfdetr_absent(self):
        with tempfile.TemporaryDirectory() as model_dir:
            (Path(model_dir) / "checkpoint_best_ema.pth").write_bytes(b"raw-ptl")
            ckpt = {"pytorch-lightning_version": "2.1.0", "args": {"segmentation_head": False}}
            torch = _fake_torch(ckpt)
            with _import_patch({"torch": torch}), mock.patch.dict(sys.modules, {"rfdetr": None}):
                with self.assertRaises(ModelPackagingError):
                    package_custom_weights("rfdetr-base", model_dir, filename="checkpoint_best_ema.pth")

    def test_keypoint_checkpoint_rejected_before_export(self):
        # A keypoint checkpoint is rejected by the task check before any rfdetr use.
        with tempfile.TemporaryDirectory() as model_dir:
            (Path(model_dir) / "checkpoint_best_ema.pth").write_bytes(b"raw-ptl")
            ckpt = {"pytorch-lightning_version": "2.1.0", "model_name": "RFDETRKeypointPreview"}
            torch = _fake_torch(ckpt)
            with _import_patch({"torch": torch}):
                with self.assertRaises(TaskMismatchError):
                    package_custom_weights("rfdetr-base", model_dir, filename="checkpoint_best_ema.pth")


if __name__ == "__main__":
    unittest.main()
