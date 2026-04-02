"""Tests for backward-compatibility aliases in _aliases.py."""

import unittest


class TestAliases(unittest.TestCase):
    """Verify top-level aliases parse correctly and delegate to the right handler."""

    def _parse(self, argv: list[str]):
        from roboflow.cli import build_parser

        parser = build_parser()
        return parser.parse_args(argv)

    def test_login_alias_exists(self) -> None:
        args = self._parse(["login"])
        self.assertIsNotNone(args.func)

    def test_login_alias_with_api_key(self) -> None:
        args = self._parse(["login", "--api-key", "test-key"])
        self.assertEqual(args.api_key_flag, "test-key")

    def test_whoami_alias_exists(self) -> None:
        args = self._parse(["whoami"])
        self.assertIsNotNone(args.func)

    def test_upload_alias_exists(self) -> None:
        args = self._parse(["upload", "img.jpg", "-p", "my-project"])
        self.assertIsNotNone(args.func)
        self.assertEqual(args.path, "img.jpg")
        self.assertEqual(args.project, "my-project")

    def test_import_alias_exists(self) -> None:
        args = self._parse(["import", "/data/images", "-p", "my-project"])
        self.assertIsNotNone(args.func)
        self.assertEqual(args.path, "/data/images")
        self.assertEqual(args.project, "my-project")

    def test_download_alias_parses_url(self) -> None:
        """Regression: download alias must use url_or_id as dest, not datasetUrl."""
        args = self._parse(["download", "my-ws/my-proj/3"])
        self.assertIsNotNone(args.func)
        # The critical check: args.url_or_id must exist (not args.datasetUrl)
        self.assertEqual(args.url_or_id, "my-ws/my-proj/3")

    def test_download_alias_with_format(self) -> None:
        args = self._parse(["download", "my-ws/my-proj/3", "-f", "yolov8"])
        self.assertEqual(args.format, "yolov8")

    def test_download_alias_with_location(self) -> None:
        args = self._parse(["download", "my-ws/my-proj/3", "-l", "/tmp/out"])
        self.assertEqual(args.location, "/tmp/out")

    def test_download_alias_delegates_to_version_download(self) -> None:
        """The download alias should use the same handler as 'version download'."""
        from roboflow.cli.handlers.version import _download

        args = self._parse(["download", "my-ws/my-proj/3"])
        self.assertIs(args.func, _download)

    def test_upload_model_alias_hidden(self) -> None:
        """upload_model is a hidden alias — it should still parse."""
        args = self._parse(["upload_model", "-p", "my-proj", "-t", "yolov8", "-m", "/weights"])
        self.assertIsNotNone(args.func)


if __name__ == "__main__":
    unittest.main()
