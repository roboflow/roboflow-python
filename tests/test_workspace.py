"""Unit tests for module-level helpers in roboflow.core.workspace."""

import os
import tempfile
import unittest
import zipfile

from roboflow.core.workspace import _zip_directory


class TestZipDirectory(unittest.TestCase):
    def test_filters_hidden_and_junk_entries(self):
        with tempfile.TemporaryDirectory() as src:
            # Real content
            with open(os.path.join(src, "sample.jpg"), "wb") as fh:
                fh.write(b"jpg bytes")
            # Hidden / junk files at the top level
            with open(os.path.join(src, ".DS_Store"), "wb") as fh:
                fh.write(b"x")
            with open(os.path.join(src, "Thumbs.db"), "wb") as fh:
                fh.write(b"x")
            # macOS junk directory
            mac_dir = os.path.join(src, "__MACOSX")
            os.mkdir(mac_dir)
            with open(os.path.join(mac_dir, "whatever.txt"), "wb") as fh:
                fh.write(b"x")
            # Hidden directory
            hidden_dir = os.path.join(src, ".hidden")
            os.mkdir(hidden_dir)
            with open(os.path.join(hidden_dir, "inside.txt"), "wb") as fh:
                fh.write(b"x")

            zip_path = _zip_directory(src)
            try:
                with zipfile.ZipFile(zip_path) as zf:
                    names = set(zf.namelist())
                self.assertEqual(names, {"sample.jpg"})
            finally:
                os.unlink(zip_path)


if __name__ == "__main__":
    unittest.main()
