"""A project id that resolves to nothing must raise, not return a Project (#302)."""

import responses

from roboflow.adapters.rfapi import ProjectNotFoundError, RoboflowError
from roboflow.config import API_URL
from tests import ROBOFLOW_API_KEY, WORKSPACE_NAME, RoboflowTest

MISSING = "my_project"

# What the API answers for a missing, trashed or inaccessible project.
NOT_FOUND_BODY = {
    "error": {
        "message": f"Unsupported request. `GET /{WORKSPACE_NAME}/{MISSING}` does not exist "
        "or cannot be loaded due to missing permissions.",
        "type": "GraphMethodException",
    }
}


class TestProjectNotFound(RoboflowTest):
    def _answer(self, status, body):
        responses.add(
            responses.GET,
            f"{API_URL}/{WORKSPACE_NAME}/{MISSING}?api_key={ROBOFLOW_API_KEY}",
            json=body,
            status=status,
        )

    def test_workspace_project_raises_naming_workspace_and_project(self):
        self._answer(404, NOT_FOUND_BODY)

        with self.assertRaises(ProjectNotFoundError) as ctx:
            self.workspace.project(MISSING)

        message = str(ctx.exception)
        self.assertIn(f"'{MISSING}'", message)
        self.assertIn(f"'{WORKSPACE_NAME}'", message)
        self.assertIn("Trash", message)
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.workspace_url, WORKSPACE_NAME)
        self.assertEqual(ctx.exception.project_url, MISSING)

    def test_roboflow_project_raises_and_stays_a_runtime_error(self):
        self._answer(404, NOT_FOUND_BODY)

        with self.assertRaises(ProjectNotFoundError) as ctx:
            self.rf.project(f"{WORKSPACE_NAME}/{MISSING}")

        self.assertIsInstance(ctx.exception, RuntimeError)
        self.assertIn(f"'{WORKSPACE_NAME}'", str(ctx.exception))

    def test_other_errors_keep_the_api_body(self):
        self._answer(500, {"error": "boom"})

        with self.assertRaises(RoboflowError) as ctx:
            self.workspace.project(MISSING)

        self.assertNotIsInstance(ctx.exception, ProjectNotFoundError)
        self.assertIn("boom", str(ctx.exception))
        self.assertEqual(ctx.exception.status_code, 500)

    def test_roboflow_project_other_errors_stay_runtime_error(self):
        self._answer(403, {"error": "forbidden"})

        with self.assertRaises(RuntimeError) as ctx:
            self.rf.project(f"{WORKSPACE_NAME}/{MISSING}")

        self.assertIn("forbidden", str(ctx.exception))
