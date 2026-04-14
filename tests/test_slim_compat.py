"""Tests for slim install compatibility.

Verifies that the package can be imported and lightweight features work
even when heavy dependencies (PIL, opencv, numpy, matplotlib) are missing.

In a full install, these tests verify the guards don't break normal behavior.
In a slim install, they verify graceful degradation.
"""

import unittest


class TestSlimImport(unittest.TestCase):
    """Verify that importing the package always succeeds."""

    def test_import_roboflow(self):
        import roboflow

        self.assertIsNotNone(roboflow.__version__)

    def test_import_vision_events_adapter(self):
        from roboflow.adapters import vision_events_api

        self.assertTrue(callable(vision_events_api.write_event))
        self.assertTrue(callable(vision_events_api.write_batch))
        self.assertTrue(callable(vision_events_api.query))
        self.assertTrue(callable(vision_events_api.list_use_cases))
        self.assertTrue(callable(vision_events_api.get_custom_metadata_schema))
        self.assertTrue(callable(vision_events_api.upload_image))

    def test_import_config(self):
        from roboflow.config import API_URL

        self.assertIsInstance(API_URL, str)

    def test_import_rfapi(self):
        from roboflow.adapters.rfapi import RoboflowError

        self.assertTrue(issubclass(RoboflowError, Exception))

    def test_import_cli(self):
        from roboflow.cli import app

        self.assertIsNotNone(app)


class TestSlimGracefulDegradation(unittest.TestCase):
    """Verify that heavy features fail with clear errors when deps are missing.

    These tests only exercise the error path when PIL/opencv are absent.
    In a full install they verify the guard exists but doesn't fire.
    """

    def test_workspace_and_project_attributes_exist(self):
        """Workspace and Project are either real classes or None sentinels."""
        import roboflow

        # In full install these are classes; in slim they're None
        ws = roboflow.Workspace
        proj = roboflow.Project
        self.assertTrue(ws is None or callable(ws))
        self.assertTrue(proj is None or callable(proj))

    def test_roboflow_workspace_guard(self):
        """If Workspace is None (slim), calling workspace() raises ImportError."""
        import roboflow

        if roboflow.Workspace is not None:
            self.skipTest("Full install, Workspace is available")

        rf = roboflow.Roboflow.__new__(roboflow.Roboflow)
        rf.api_key = "test"
        rf.current_workspace = "test"
        rf.model_format = "yolov8"

        with self.assertRaises(ImportError) as ctx:
            rf.workspace()
        self.assertIn("pip install roboflow", str(ctx.exception))

    def test_roboflow_project_guard(self):
        """If Project is None (slim), calling project() raises ImportError."""
        import roboflow

        if roboflow.Project is not None:
            self.skipTest("Full install, Project is available")

        rf = roboflow.Roboflow.__new__(roboflow.Roboflow)
        rf.api_key = "test"
        rf.current_workspace = "test"

        with self.assertRaises(ImportError) as ctx:
            rf.project("test-project")
        self.assertIn("pip install roboflow", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
