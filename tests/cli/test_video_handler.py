"""Tests for the video CLI handler."""

import json
import os
import unittest
from unittest.mock import patch

from typer.testing import CliRunner

from roboflow.cli import app

runner = CliRunner()


class TestVideoRegistration(unittest.TestCase):
    """Verify video handler registers expected subcommands."""

    def test_video_app_exists(self) -> None:
        from roboflow.cli.handlers.video import video_app

        self.assertIsNotNone(video_app)

    def test_video_infer_exists(self) -> None:
        result = runner.invoke(app, ["video", "infer", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_video_status_exists(self) -> None:
        result = runner.invoke(app, ["video", "status", "--help"])
        self.assertEqual(result.exit_code, 0)


class TestVideoStatus(unittest.TestCase):
    """Test video status handler."""

    @patch("roboflow.config.load_roboflow_api_key", return_value=None)
    def test_status_no_api_key(self, _mock_key) -> None:
        result = runner.invoke(app, ["--json", "video", "status", "job-123"])
        self.assertNotEqual(result.exit_code, 0)

    @patch("roboflow.adapters.rfapi.get_video_job_status")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_status_success(self, _mock_key, mock_api) -> None:
        mock_api.return_value = {"status": "completed", "progress": "100%"}
        result = runner.invoke(app, ["video", "status", "job-abc"])
        self.assertIn("job-abc", result.output)
        self.assertIn("completed", result.output)

    @patch("roboflow.adapters.rfapi.get_video_job_status")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_status_json_output(self, _mock_key, mock_api) -> None:
        mock_api.return_value = {"status": "processing", "progress": "50%"}
        result = runner.invoke(app, ["--json", "video", "status", "job-abc"])
        data = json.loads(result.output)
        self.assertEqual(data["status"], "processing")

    @patch("roboflow.adapters.rfapi.get_video_job_status")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_status_passes_job_id_to_api(self, _mock_key, mock_api) -> None:
        mock_api.return_value = {"status": "completed"}
        runner.invoke(app, ["video", "status", "my-unique-job-777"])
        mock_api.assert_called_once_with("fake-key", "my-unique-job-777")


AR_PROJECT_PAYLOAD = {
    "project": {
        "annotation": "actions",
        "classes": {"walking": 2},
        "colors": {"walking": "#FF00FF"},
        "created": 1759000000.0,
        "id": "model-evaluation-workspace/penguin-actions",
        "images": 1,
        "name": "Penguin Actions",
        "public": False,
        "splits": {"train": 1, "test": 0, "valid": 0},
        "type": "action-recognition",
        "unannotated": 0,
        "updated": 1759000001.0,
    }
}

# A complete video-coco document in the shape the import schema accepts, taken
# from a real MOV: `segments` is top level, `time_base` is a rational object,
# and `images`/`annotations` stay empty. Tests assert it reaches the SDK with
# every value identical.
VIDEO_COCO_DOCUMENT = {
    "info": {"format": "roboflow-video-coco"},
    "videos": [
        {
            "id": 1,
            "file_name": "clip.mov",
            "width": 1620,
            "height": 1080,
            "duration": 4.566667,
            "fps": 30,
            "frame_count": 137,
            "time_base": {"numerator": 1, "denominator": 15360},
        }
    ],
    "categories": [{"id": 1, "name": "hand_gesture"}],
    # Real MOV presentation timestamps are not frame_index * ticks_per_frame,
    # so they must survive the read exactly rather than being recomputed.
    "segments": [
        {
            "id": 1,
            "video_id": 1,
            "category_id": 1,
            "start_frame": 5,
            "end_frame": 64,
            "start_pts": 3067,
            "end_pts": 33275,
        }
    ],
    "images": [],
    "annotations": [],
}


def _write_json(directory, name, payload):
    import json as json_mod
    import os

    path = os.path.join(directory, name)
    with open(path, "w") as handle:
        json_mod.dump(payload, handle)
    return path


class NativeVideoCliTest(unittest.TestCase):
    """Shared fixtures that let real command dispatch build a real Project."""

    def setUp(self) -> None:
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.video_path = os.path.join(self.tmp.name, "clip.mp4")
        with open(self.video_path, "wb") as handle:
            handle.write(b"original video bytes")

        key_patch = patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
        key_patch.start()
        self.addCleanup(key_patch.stop)

        project_patch = patch("roboflow.adapters.rfapi.get_project", return_value=AR_PROJECT_PAYLOAD)
        self.mock_get_project = project_patch.start()
        self.addCleanup(project_patch.stop)

    @property
    def project_ref(self) -> str:
        return "model-evaluation-workspace/penguin-actions"


class TestNativeVideoRegistration(NativeVideoCliTest):
    """The real CLI exposes the native video commands alongside inference."""

    def test_native_commands_are_registered(self) -> None:
        for command in ("upload", "upload-status", "annotate"):
            with self.subTest(command=command):
                result = runner.invoke(app, ["video", command, "--help"])
                self.assertEqual(result.exit_code, 0)


class TestVideoUpload(NativeVideoCliTest):
    """`roboflow video upload` streams original bytes and reports canonical IDs."""

    @patch("roboflow.core.project.Project.wait_for_video_upload")
    @patch("roboflow.core.project.Project.upload_video")
    def test_forwards_all_options_and_waits_by_default(self, mock_upload, mock_wait) -> None:
        mock_upload.return_value = {"videoId": "upload-1", "status": "pending"}
        mock_wait.return_value = {
            "videoId": "source-9",
            "status": "uploaded",
            "resolvedBatch": {"id": "b1", "name": "clips"},
        }

        result = runner.invoke(
            app,
            [
                "video",
                "upload",
                "-p",
                self.project_ref,
                "-f",
                self.video_path,
                "-b",
                "clips",
                "-t",
                "indoor, penguin",
                "--metadata",
                '{"camera": "one"}',
                "-s",
                "valid",
                "--poll-interval",
                "0.5",
                "--poll-timeout",
                "30",
            ],
        )

        self.assertEqual(result.exit_code, 0, result.output)
        mock_upload.assert_called_once_with(
            self.video_path,
            batch_name="clips",
            tag_names=["indoor", "penguin"],
            metadata={"camera": "one"},
            split="valid",
            wait=False,
        )
        # The bounded wait continues on the ID the first status reported.
        mock_wait.assert_called_once_with("upload-1", poll_interval=0.5, poll_timeout=30.0)
        self.assertIn("source-9", result.output)
        self.assertIn("uploaded", result.output)

    @patch("roboflow.core.project.Project.wait_for_video_upload")
    @patch("roboflow.core.project.Project.upload_video")
    def test_no_wait_reports_reservation_without_polling(self, mock_upload, mock_wait) -> None:
        mock_upload.return_value = {"videoId": "upload-1", "status": "pending"}

        result = runner.invoke(
            app, ["--json", "video", "upload", "-p", self.project_ref, "-f", self.video_path, "--no-wait"]
        )

        self.assertEqual(result.exit_code, 0, result.output)
        mock_wait.assert_not_called()
        data = json.loads(result.output)
        self.assertEqual(data, {"videoId": "upload-1", "status": "pending"})

    @patch("roboflow.core.project.Project.wait_for_video_upload")
    @patch("roboflow.core.project.Project.upload_video")
    def test_terminal_dedup_status_skips_the_wait(self, mock_upload, mock_wait) -> None:
        mock_upload.return_value = {
            "videoId": "source-2",
            "status": "uploaded",
            "duplicate": True,
            "resolvedBatch": None,
        }

        result = runner.invoke(app, ["--json", "video", "upload", "-p", self.project_ref, "-f", self.video_path])

        self.assertEqual(result.exit_code, 0, result.output)
        mock_wait.assert_not_called()
        data = json.loads(result.output)
        self.assertEqual(data["videoId"], "source-2")
        self.assertIs(data["duplicate"], True)
        self.assertIsNone(data["resolvedBatch"])

    @patch("roboflow.core.project.Project.upload_video")
    def test_failed_processing_exits_nonzero(self, mock_upload) -> None:
        mock_upload.return_value = {"videoId": "upload-1", "status": "failed"}

        result = runner.invoke(app, ["--json", "video", "upload", "-p", self.project_ref, "-f", self.video_path])

        self.assertNotEqual(result.exit_code, 0)
        payload = json.loads(result.output)
        self.assertIn("failed", payload["error"]["message"])

    @patch("roboflow.core.project.Project.upload_video")
    def test_wait_timeout_names_the_video_id_to_recheck(self, mock_upload) -> None:
        from roboflow.adapters.rfapi import RoboflowError

        mock_upload.return_value = {"videoId": "upload-7", "status": "pending"}
        with patch(
            "roboflow.core.project.Project.wait_for_video_upload",
            side_effect=RoboflowError("Video upload upload-7 is still pending after 30s"),
        ):
            result = runner.invoke(
                app,
                ["--json", "video", "upload", "-p", self.project_ref, "-f", self.video_path, "--poll-timeout", "30"],
            )

        self.assertNotEqual(result.exit_code, 0)
        payload = json.loads(result.output)
        self.assertIn("upload-status upload-7", payload["error"]["hint"])

    @patch("roboflow.core.project.Project.upload_video")
    def test_sdk_file_rejections_get_their_own_hint(self, mock_upload) -> None:
        cases = [
            ("Native video upload accepts .mp4 and .mov files", "accepts original .mp4 and .mov"),
            ("Video file not found: /tmp/absent.mp4", "Check the path"),
        ]
        for message, hint in cases:
            with self.subTest(message=message):
                mock_upload.side_effect = ValueError(message)

                result = runner.invoke(
                    app, ["--json", "video", "upload", "-p", self.project_ref, "-f", self.video_path]
                )

                self.assertEqual(result.exit_code, 1)
                error = json.loads(result.output)["error"]
                self.assertEqual(error["message"], message)
                self.assertIn(hint, error["hint"])

    @patch("roboflow.core.project.Project.upload_video")
    def test_invalid_metadata_never_reaches_the_api(self, mock_upload) -> None:
        cases = [("{not json", "Invalid metadata JSON"), ("[1, 2]", "Metadata must be a JSON object")]
        for metadata, message in cases:
            with self.subTest(metadata=metadata):
                result = runner.invoke(
                    app,
                    [
                        "--json",
                        "video",
                        "upload",
                        "-p",
                        self.project_ref,
                        "-f",
                        self.video_path,
                        "--metadata",
                        metadata,
                    ],
                )

                self.assertEqual(result.exit_code, 1)
                self.assertIn(message, json.loads(result.output)["error"]["message"])
        mock_upload.assert_not_called()
        self.mock_get_project.assert_not_called()

    @patch("roboflow.core.project.Project.upload_video")
    def test_server_error_exits_nonzero(self, mock_upload) -> None:
        from roboflow.adapters.rfapi import RoboflowError

        mock_upload.side_effect = RoboflowError("quota exceeded")
        result = runner.invoke(app, ["--json", "video", "upload", "-p", self.project_ref, "-f", self.video_path])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("quota exceeded", json.loads(result.output)["error"]["message"])

    def test_project_lookup_failure_follows_exit_code_contract(self) -> None:
        from roboflow.adapters.rfapi import RoboflowError

        for status_code, exit_code in ((401, 2), (404, 3), (500, 1)):
            with self.subTest(status_code=status_code):
                self.mock_get_project.side_effect = RoboflowError("project lookup failed", status_code=status_code)

                result = runner.invoke(
                    app, ["--json", "video", "upload", "-p", self.project_ref, "-f", self.video_path]
                )

                self.assertEqual(result.exit_code, exit_code)
                self.assertIn("project lookup failed", json.loads(result.output)["error"]["message"])

    def test_missing_api_key_exits_with_auth_code(self) -> None:
        with patch("roboflow.config.load_roboflow_api_key", return_value=None):
            result = runner.invoke(app, ["--json", "video", "upload", "-p", self.project_ref, "-f", self.video_path])
        self.assertEqual(result.exit_code, 2)

    @patch("roboflow.core.project.Project.upload_video")
    def test_explicit_api_key_takes_precedence(self, mock_upload) -> None:
        mock_upload.return_value = {"videoId": "v1", "status": "uploaded"}
        result = runner.invoke(
            app,
            ["--api-key", "explicit-key", "video", "upload", "-p", self.project_ref, "-f", self.video_path],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.mock_get_project.assert_called_once_with("explicit-key", "model-evaluation-workspace", "penguin-actions")


class TestVideoUploadStatus(NativeVideoCliTest):
    """`roboflow video upload-status` reads native ingestion state."""

    @patch("roboflow.core.project.Project.get_video_upload_status")
    def test_single_read_by_default(self, mock_status) -> None:
        mock_status.return_value = {"videoId": "source-3", "status": "uploaded"}

        result = runner.invoke(app, ["--json", "video", "upload-status", "source-3", "-p", self.project_ref])

        self.assertEqual(result.exit_code, 0, result.output)
        mock_status.assert_called_once_with("source-3")
        self.assertEqual(json.loads(result.output)["status"], "uploaded")

    @patch("roboflow.core.project.Project.wait_for_video_upload")
    @patch("roboflow.core.project.Project.get_video_upload_status")
    def test_wait_uses_the_bounded_poll(self, mock_status, mock_wait) -> None:
        mock_wait.return_value = {"videoId": "source-3", "status": "uploaded"}

        result = runner.invoke(
            app,
            ["video", "upload-status", "source-3", "-p", self.project_ref, "--wait", "--poll-timeout", "12"],
        )

        self.assertEqual(result.exit_code, 0, result.output)
        mock_status.assert_not_called()
        mock_wait.assert_called_once_with("source-3", poll_interval=2.0, poll_timeout=12.0)

    @patch("roboflow.core.project.Project.get_video_upload_status")
    def test_pending_state_points_at_the_recheck_command(self, mock_status) -> None:
        mock_status.return_value = {"videoId": "source-3", "status": "pending"}

        result = runner.invoke(app, ["video", "upload-status", "source-3", "-p", self.project_ref])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("upload-status source-3", result.output)

    def test_api_errors_follow_exit_code_contract(self) -> None:
        from roboflow.adapters.rfapi import RoboflowError

        reads = [("get_video_upload_status", []), ("wait_for_video_upload", ["--wait"])]
        for method, flags in reads:
            for status_code, exit_code in ((401, 2), (404, 3), (500, 1)):
                with self.subTest(method=method, status_code=status_code):
                    with patch(
                        f"roboflow.core.project.Project.{method}",
                        side_effect=RoboflowError("status read failed", status_code=status_code),
                    ):
                        result = runner.invoke(
                            app, ["--json", "video", "upload-status", "nope", "-p", self.project_ref, *flags]
                        )

                    self.assertEqual(result.exit_code, exit_code)
                    if status_code == 404:
                        self.assertIn("Check the video ID", json.loads(result.output)["error"]["hint"])

    @patch("roboflow.core.project.Project.get_video_upload_status")
    def test_failed_state_exits_nonzero(self, mock_status) -> None:
        mock_status.return_value = {"videoId": "source-3", "status": "failed"}
        result = runner.invoke(app, ["--json", "video", "upload-status", "source-3", "-p", self.project_ref])
        self.assertNotEqual(result.exit_code, 0)


class TestVideoAnnotate(NativeVideoCliTest):
    """`roboflow video annotate` forwards the video-coco document unchanged."""

    def setUp(self) -> None:
        super().setUp()
        self.document_path = _write_json(self.tmp.name, "segments.json", VIDEO_COCO_DOCUMENT)

    @patch("roboflow.core.project.Project.annotate_video_segments")
    def test_document_and_defaults_are_forwarded_unchanged(self, mock_annotate) -> None:
        mock_annotate.return_value = {"success": True, "inDataset": True, "createdClasses": ["walking"]}

        result = runner.invoke(
            app,
            ["video", "annotate", "-p", self.project_ref, "-i", "source-9", "-a", self.document_path],
        )

        self.assertEqual(result.exit_code, 0, result.output)
        mock_annotate.assert_called_once_with(
            "source-9",
            VIDEO_COCO_DOCUMENT,
            overwrite=False,
            split=None,
            add_to_dataset=None,
        )
        self.assertIn("walking", result.output)

    @patch("roboflow.core.project.Project.annotate_video_segments")
    def test_explicit_overwrite_split_and_membership_are_forwarded(self, mock_annotate) -> None:
        mock_annotate.return_value = {"success": True, "inDataset": False}

        result = runner.invoke(
            app,
            [
                "video",
                "annotate",
                "-p",
                self.project_ref,
                "-i",
                "source-9",
                "-a",
                self.document_path,
                "--overwrite",
                "-s",
                "test",
                "--no-add-to-dataset",
            ],
        )

        self.assertEqual(result.exit_code, 0, result.output)
        mock_annotate.assert_called_once_with(
            "source-9",
            VIDEO_COCO_DOCUMENT,
            overwrite=True,
            split="test",
            add_to_dataset=False,
        )
        self.assertIn("In dataset: no", result.output)

        mock_annotate.reset_mock()
        runner.invoke(
            app,
            ["video", "annotate", "-p", self.project_ref, "-i", "s1", "-a", self.document_path, "--add-to-dataset"],
        )
        self.assertIs(mock_annotate.call_args.kwargs["add_to_dataset"], True)

    @patch("roboflow.core.project.Project.annotate_video_segments")
    def test_json_output_is_the_server_response(self, mock_annotate) -> None:
        mock_annotate.return_value = {"success": True, "inDataset": True, "createdClasses": []}

        result = runner.invoke(
            app,
            ["--json", "video", "annotate", "-p", self.project_ref, "-i", "s1", "-a", self.document_path],
        )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(json.loads(result.output), {"success": True, "inDataset": True, "createdClasses": []})

    @patch("roboflow.core.project.Project.annotate_video_segments")
    def test_api_rejections_follow_exit_code_contract(self, mock_annotate) -> None:
        from roboflow.adapters.rfapi import AnnotationSaveError

        cases = [
            (409, 1, "--overwrite"),
            (400, 1, "at least one segment"),
            (404, 3, "canonical video ID"),
            (401, 2, "ROBOFLOW_API_KEY"),
            (None, 1, None),  # transport failure: no status, no document hint
        ]
        for status_code, exit_code, hint in cases:
            with self.subTest(status_code=status_code):
                mock_annotate.side_effect = AnnotationSaveError("server said no", status_code=status_code)

                result = runner.invoke(
                    app,
                    ["--json", "video", "annotate", "-p", self.project_ref, "-i", "s1", "-a", self.document_path],
                )

                self.assertEqual(result.exit_code, exit_code)
                error = json.loads(result.output)["error"]
                self.assertEqual(error["message"], "server said no")
                if hint is None:
                    self.assertNotIn("hint", error)
                else:
                    self.assertIn(hint, error["hint"])

    @patch("roboflow.core.project.Project.annotate_video_segments")
    def test_unreadable_document_never_reaches_the_api(self, mock_annotate) -> None:
        malformed = os.path.join(self.tmp.name, "bad.json")
        with open(malformed, "w") as handle:
            handle.write('{"segments": ')
        utf16 = os.path.join(self.tmp.name, "utf16.json")
        with open(utf16, "w", encoding="utf-16") as handle:  # what Windows PowerShell 5.1 redirection writes
            json.dump(VIDEO_COCO_DOCUMENT, handle)
        cases = [
            (malformed, "Invalid JSON"),
            (utf16, "is not UTF-8"),
            (_write_json(self.tmp.name, "list.json", [1, 2, 3]), "must contain a JSON object"),
            (os.path.join(self.tmp.name, "absent.json"), "Cannot read annotation file"),
        ]
        for path, message in cases:
            with self.subTest(message=message):
                result = runner.invoke(
                    app, ["--json", "video", "annotate", "-p", self.project_ref, "-i", "s1", "-a", path]
                )

                self.assertEqual(result.exit_code, 1)
                self.assertIn(message, json.loads(result.output)["error"]["message"])
        mock_annotate.assert_not_called()
        self.mock_get_project.assert_not_called()


class TestLegacyVideoContractsIntact(unittest.TestCase):
    """The native commands must not disturb legacy video inference."""

    def test_infer_still_takes_a_version_number(self) -> None:
        result = runner.invoke(app, ["video", "infer", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--version", result.output)


if __name__ == "__main__":
    unittest.main()
