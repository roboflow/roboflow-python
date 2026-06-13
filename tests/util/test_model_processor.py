import os
import sys
import tempfile
import unittest
import zipfile
from types import SimpleNamespace
from unittest import mock

try:
    # torch is an optional, lazily-imported SDK dependency; absent in CI. Tests that
    # round-trip a real checkpoint through `_process_rfdetr` are skipped without it.
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from roboflow.config import TASK_CLS, TASK_DET, TASK_OBB, TASK_POSE, TASK_SEG, TASK_SEM
from roboflow.util.model_processor import (
    _RFDETR_MODEL_TYPE_TO_CLASS,
    _detect_rfdetr_task,
    _detect_yolo_task,
    _is_ptl_checkpoint,
    _process_rfdetr,
    _require_rfdetr,
    get_classnames_txt_for_rfdetr,
    task_of_model_type,
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


class RfdetrModelTypeToClassTest(unittest.TestCase):
    def test_representative_mappings(self):
        self.assertEqual(_RFDETR_MODEL_TYPE_TO_CLASS["rfdetr-seg-medium"], "RFDETRSegMedium")
        self.assertEqual(_RFDETR_MODEL_TYPE_TO_CLASS["rfdetr-base"], "RFDETRBase")

    def test_keys_are_rfdetr_types_and_values_are_class_names(self):
        for model_type, class_name in _RFDETR_MODEL_TYPE_TO_CLASS.items():
            self.assertTrue(model_type.startswith("rfdetr-"), model_type)
            self.assertTrue(class_name.startswith("RFDETR"), class_name)
        # Segmentation types must map to Seg classes (and detection types must not).
        for model_type, class_name in _RFDETR_MODEL_TYPE_TO_CLASS.items():
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
        torch.save({"dummy": True}, os.path.join(output_dir, "weights.pt"))
        with open(os.path.join(output_dir, "class_names.txt"), "w") as f:
            for name in self.class_names:
                f.write(name + "\n")


def _make_fake_rfdetr(*, from_checkpoint_raises=False, capabilities=True):
    """Build a fake `rfdetr` module for injection via sys.modules."""

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
    # The SDK fallback resolves the subclass by name via _RFDETR_MODEL_TYPE_TO_CLASS,
    # e.g. "rfdetr-seg-medium" -> getattr(rfdetr, "RFDETRSegMedium").
    module.RFDETRSegMedium = _SizedModel

    if capabilities:
        _RFDETR.export_for_roboflow = _StubBundleModel.export_for_roboflow  # capability marker on class

    module._calls = calls
    return module


class RequireRfdetrTest(unittest.TestCase):
    def test_raises_when_not_installed(self):
        with mock.patch.dict(sys.modules, {"rfdetr": None}):
            with self.assertRaises(RuntimeError) as ctx:
                _require_rfdetr()
        self.assertIn("pip install", str(ctx.exception).lower())
        self.assertIn("rfdetr", str(ctx.exception).lower())

    def test_raises_when_capability_missing(self):
        # rfdetr present but RFDETR lacks export_for_roboflow (too old)
        fake = SimpleNamespace(RFDETR=type("RFDETR", (), {}))
        with mock.patch.dict(sys.modules, {"rfdetr": fake}):
            with self.assertRaises(RuntimeError) as ctx:
                _require_rfdetr()
        self.assertIn("upgrade", str(ctx.exception).lower())

    def test_returns_module_when_capable(self):
        fake = _make_fake_rfdetr()
        with mock.patch.dict(sys.modules, {"rfdetr": fake}):
            self.assertIs(_require_rfdetr(), fake)


@unittest.skipUnless(_HAS_TORCH, "requires torch")
class ProcessRfdetrPtlTest(unittest.TestCase):
    def _write_ptl_checkpoint(self, model_path, *, segmentation_head=False, class_names=("cat", "dog")):
        checkpoint = {
            "pytorch-lightning_version": "2.1.0",
            "args": {"segmentation_head": segmentation_head, "class_names": list(class_names)},
        }
        torch.save(checkpoint, os.path.join(model_path, "checkpoint_best_ema.pth"))

    def test_from_checkpoint_success_produces_bundle(self):
        fake = _make_fake_rfdetr()
        with tempfile.TemporaryDirectory() as model_path:
            self._write_ptl_checkpoint(model_path)
            with mock.patch.dict(sys.modules, {"rfdetr": fake}):
                zip_name, model_type = _process_rfdetr("rfdetr-base", model_path, "checkpoint_best_ema.pth")
            self.assertEqual(model_type, "rfdetr-base")
            self.assertEqual(fake._calls["from_checkpoint"], 1)
            self.assertEqual(fake._calls["fallback_constructed"], 0)
            with zipfile.ZipFile(os.path.join(model_path, zip_name)) as z:
                self.assertIn("weights.pt", z.namelist())

    def test_from_checkpoint_valueerror_falls_back_to_model_type(self):
        fake = _make_fake_rfdetr(from_checkpoint_raises=True)
        with tempfile.TemporaryDirectory() as model_path:
            self._write_ptl_checkpoint(model_path, segmentation_head=True)
            pth = os.path.join(model_path, "checkpoint_best_ema.pth")
            with mock.patch.dict(sys.modules, {"rfdetr": fake}):
                zip_name, model_type = _process_rfdetr("rfdetr-seg-medium", model_path, "checkpoint_best_ema.pth")
            self.assertEqual(model_type, "rfdetr-seg-medium")
            self.assertEqual(fake._calls["from_checkpoint"], 1)
            self.assertEqual(fake._calls["fallback_constructed"], 1)
            self.assertEqual(fake._calls["constructor_kwargs"], {"pretrain_weights": pth})
            with zipfile.ZipFile(os.path.join(model_path, zip_name)) as z:
                self.assertIn("weights.pt", z.namelist())

    def test_ptl_path_raises_when_rfdetr_absent(self):
        with tempfile.TemporaryDirectory() as model_path:
            self._write_ptl_checkpoint(model_path)
            with mock.patch.dict(sys.modules, {"rfdetr": None}):
                with self.assertRaises(RuntimeError):
                    _process_rfdetr("rfdetr-base", model_path, "checkpoint_best_ema.pth")


@unittest.skipUnless(_HAS_TORCH, "requires torch")
class ProcessRfdetrLegacyTest(unittest.TestCase):
    def _write_legacy_checkpoint(self, model_path):
        checkpoint = {"args": {"segmentation_head": False, "class_names": ["cat", "dog"]}}
        torch.save(checkpoint, os.path.join(model_path, "weights.pt"))

    def test_legacy_path_produces_bundle_without_importing_rfdetr(self):
        with tempfile.TemporaryDirectory() as model_path:
            self._write_legacy_checkpoint(model_path)
            # Make any attempt to import rfdetr fail loudly.
            with mock.patch.dict(sys.modules, {"rfdetr": None}):
                zip_name, model_type = _process_rfdetr("rfdetr-base", model_path, "weights.pt")
            self.assertEqual(model_type, "rfdetr-base")
            with zipfile.ZipFile(os.path.join(model_path, zip_name)) as z:
                names = z.namelist()
            self.assertIn("weights.pt", names)
            self.assertIn("class_names.txt", names)


if __name__ == "__main__":
    unittest.main()
