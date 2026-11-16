"""stdout stays valid JSON in --json mode when SDK calls print progress."""

import contextlib
import io
import json
import os
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch


def _sdk_print(*_args: object, **_kwargs: object) -> None:
    print("progress line printed by the SDK")


def _run(handler, *args: object) -> tuple:  # noqa: ANN001
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        handler(*args)
    return stdout.getvalue(), stderr.getvalue()


class TestJsonStdout(unittest.TestCase):
    def assert_json_stdout(self, stdout: str, stderr: str) -> dict:
        self.assertNotIn("progress line printed by the SDK", stdout)
        self.assertIn("progress line printed by the SDK", stderr)
        return json.loads(stdout)

    def test_search_export(self) -> None:
        from roboflow.cli.handlers.search import _do_export

        def search_export(**_kwargs: object) -> str:
            _sdk_print()
            return "export-dir"

        workspace = MagicMock()
        workspace.search_export.side_effect = search_export
        args = types.SimpleNamespace(
            json=True,
            query="tag:a",
            format="coco",
            location="export-dir",
            dataset=None,
            annotation_group=None,
            name=None,
            no_extract=False,
        )

        stdout, stderr = _run(_do_export, args, workspace)

        self.assertEqual(self.assert_json_stdout(stdout, stderr)["status"], "completed")

    @patch("roboflow.Roboflow")
    def test_image_upload_directory(self, mock_rf_cls: MagicMock) -> None:
        from roboflow.cli.handlers.image import _handle_upload_directory

        mock_rf_cls.return_value.workspace.return_value.upload_dataset.side_effect = _sdk_print
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "a.jpg"), "w") as f:
                f.write("x")
            args = types.SimpleNamespace(
                json=True,
                workspace="ws",
                project="proj",
                concurrency=1,
                retries=0,
                tag=None,
                batch=None,
                split=None,
                is_prediction=False,
                zip_upload=False,
                annotation_overwrite=None,
                no_wait=False,
            )

            stdout, stderr = _run(_handle_upload_directory, args, "key", tmpdir)

        self.assertEqual(self.assert_json_stdout(stdout, stderr)["count"], 1)

    @patch("roboflow.Roboflow")
    def test_model_upload(self, mock_rf_cls: MagicMock) -> None:
        from roboflow.cli.handlers.model import _upload_model

        def workspace(*_args: object) -> MagicMock:
            print("loading Roboflow workspace...")
            mock_workspace = MagicMock()
            mock_workspace.project.return_value.version.return_value.deploy.side_effect = _sdk_print
            return mock_workspace

        mock_rf_cls.return_value.workspace.side_effect = workspace
        args = types.SimpleNamespace(
            json=True,
            api_key="key",
            workspace="ws",
            project=["proj"],
            version_number=1,
            model_type="yolov8",
            model_path="/path/to/model",
            filename="weights/best.pt",
            model_name=None,
        )

        stdout, stderr = _run(_upload_model, args)

        self.assertEqual(self.assert_json_stdout(stdout, stderr)["status"], "uploaded")
        self.assertNotIn("loading Roboflow workspace", stdout + stderr)

    @patch("roboflow.Roboflow")
    def test_version_download(self, mock_rf_cls: MagicMock) -> None:
        from roboflow.cli.handlers.version import _download

        project = mock_rf_cls.return_value.workspace.return_value.project.return_value
        project.version.return_value.download.side_effect = _sdk_print
        args = types.SimpleNamespace(json=True, url_or_id="ws/proj/1", format="coco", location="dataset-dir")

        stdout, stderr = _run(_download, args)

        self.assertEqual(self.assert_json_stdout(stdout, stderr)["version"], 1)

    def test_text_mode_keeps_sdk_output_on_stdout(self) -> None:
        from roboflow.cli._output import sdk_output_to_stderr

        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with sdk_output_to_stderr(types.SimpleNamespace(json=False)):
                _sdk_print()

        self.assertIn("progress line printed by the SDK", stdout.getvalue())
        self.assertEqual(stderr.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
