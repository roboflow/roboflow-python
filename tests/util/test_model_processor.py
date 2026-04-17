import unittest
from types import SimpleNamespace

from roboflow.util.model_processor import _detect_task_from_pt, task_of_model_type


class _FakeModel:
    """Stand-in for an Ultralytics model_instance; only __class__.__name__ matters."""


def _make_fake(name: str):
    return type(name, (_FakeModel,), {})()


class TaskOfModelTypeTest(unittest.TestCase):
    def test_detect_defaults(self):
        self.assertEqual(task_of_model_type("yolov11"), "detect")
        self.assertEqual(task_of_model_type("rfdetr-base"), "detect")
        self.assertEqual(task_of_model_type("rfdetr-medium"), "detect")
        self.assertEqual(task_of_model_type("yolov8"), "detect")

    def test_segment(self):
        self.assertEqual(task_of_model_type("yolov11-seg"), "segment")
        self.assertEqual(task_of_model_type("rfdetr-seg-medium"), "segment")
        self.assertEqual(task_of_model_type("yolov7-seg"), "segment")

    def test_pose(self):
        self.assertEqual(task_of_model_type("yolov11-pose"), "pose")
        # Future-proof for rfdetr-pose.
        self.assertEqual(task_of_model_type("rfdetr-pose-medium"), "pose")

    def test_classify(self):
        self.assertEqual(task_of_model_type("yolov11-cls"), "classify")

    def test_obb(self):
        self.assertEqual(task_of_model_type("yolov11-obb"), "obb")
        # Future-proof for rfdetr-obb.
        self.assertEqual(task_of_model_type("rfdetr-obb-large"), "obb")


class DetectTaskFromPtTest(unittest.TestCase):
    def test_ultralytics_class_names(self):
        cases = {
            "SegmentationModel": "segment",
            "PoseModel": "pose",
            "ClassificationModel": "classify",
            "OBBModel": "obb",
            "DetectionModel": "detect",
        }
        for cls_name, expected in cases.items():
            instance = _make_fake(cls_name)
            self.assertEqual(_detect_task_from_pt({}, instance), expected, cls_name)

    def test_rfdetr_segmentation_head_true(self):
        checkpoint = {"args": SimpleNamespace(segmentation_head=True)}
        self.assertEqual(_detect_task_from_pt(checkpoint, None), "segment")

    def test_rfdetr_segmentation_head_false(self):
        checkpoint = {"args": SimpleNamespace(segmentation_head=False)}
        self.assertEqual(_detect_task_from_pt(checkpoint, None), "detect")

    def test_future_args_task_string(self):
        # Future-proof: if rfdetr (or any framework) adds args.task.
        checkpoint = {"args": SimpleNamespace(task="pose")}
        self.assertEqual(_detect_task_from_pt(checkpoint, None), "pose")

    def test_args_task_dict(self):
        checkpoint = {"args": {"task": "segment"}}
        self.assertEqual(_detect_task_from_pt(checkpoint, None), "segment")

    def test_unrecognized_returns_none(self):
        self.assertIsNone(_detect_task_from_pt({}, None))
        self.assertIsNone(_detect_task_from_pt({"args": SimpleNamespace(other=1)}, None))
        self.assertIsNone(_detect_task_from_pt({}, _make_fake("Model")))


if __name__ == "__main__":
    unittest.main()
