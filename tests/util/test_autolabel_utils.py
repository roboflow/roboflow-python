"""Unit tests for roboflow.util.autolabel_utils."""

import base64
import os
import tempfile
import unittest
from unittest.mock import patch

from roboflow.util.autolabel_utils import image_payload, ontology_payload, resolve_model


class TestImagePayload(unittest.TestCase):
    def test_url(self):
        self.assertEqual(
            image_payload("https://example.com/cat.jpg"),
            {"type": "url", "value": "https://example.com/cat.jpg"},
        )

    def test_local_file_is_base64_encoded(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
            handle.write(b"fake-image-bytes")
            path = handle.name
        try:
            payload = image_payload(path)
        finally:
            os.unlink(path)
        self.assertEqual(payload["type"], "base64")
        self.assertEqual(base64.b64decode(payload["value"]), b"fake-image-bytes")

    def test_other_strings_are_treated_as_base64(self):
        encoded = base64.b64encode(b"bytes").decode("ascii")
        self.assertEqual(image_payload(encoded), {"type": "base64", "value": encoded})

    def test_line_wrapped_base64_is_compacted(self):
        encoded = base64.b64encode(b"some longer image bytes").decode("ascii")
        wrapped = encoded[:8] + "\n" + encoded[8:] + "\n"
        self.assertEqual(image_payload(wrapped), {"type": "base64", "value": encoded})

    def test_home_directory_is_expanded(self):
        with tempfile.TemporaryDirectory() as home:
            with open(os.path.join(home, "cat.jpg"), "wb") as handle:
                handle.write(b"fake-image-bytes")
            # expanduser reads HOME on POSIX and USERPROFILE on Windows.
            with patch.dict(os.environ, {"HOME": home, "USERPROFILE": home}):
                payload = image_payload("~/cat.jpg")
        self.assertEqual(base64.b64decode(payload["value"]), b"fake-image-bytes")

    def test_mistyped_path_is_rejected_instead_of_sent_as_base64(self):
        with self.assertRaises(ValueError) as ctx:
            image_payload("smaple.jpg")
        self.assertIn("smaple.jpg", str(ctx.exception))

    def test_missing_home_path_is_rejected(self):
        with self.assertRaises(ValueError):
            image_payload("~/definitely-missing-image.png")

    def test_unreadable_file_raises_oserror(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
            path = handle.name
        try:
            with patch("roboflow.util.autolabel_utils.open", side_effect=PermissionError(13, "denied"), create=True):
                with self.assertRaises(OSError):
                    image_payload(path)
        finally:
            os.unlink(path)


class TestOntologyPayload(unittest.TestCase):
    def test_none_stays_none(self):
        self.assertIsNone(ontology_payload(None))

    def test_prompt_keyed_mapping_is_passed_through(self):
        self.assertEqual(ontology_payload({"a cat": "cat"}), {"a cat": "cat"})

    def test_several_prompts_may_share_one_class(self):
        # The reason the object is keyed by prompt rather than by class: a
        # class-keyed object has room for exactly one prompt per class.
        self.assertEqual(
            ontology_payload({"kitten": "cat", "tabby": "cat", "puppy": "dog"}),
            {"kitten": "cat", "tabby": "cat", "puppy": "dog"},
        )

    def test_list_of_classes_becomes_identity_prompts(self):
        self.assertEqual(ontology_payload(["cat", "dog"]), {"cat": "cat", "dog": "dog"})

    def test_the_result_is_a_copy(self):
        source = {"a cat": "cat"}
        self.assertIsNot(ontology_payload(source), source)

    def test_bare_string_is_rejected_rather_than_iterated_per_character(self):
        with self.assertRaises(ValueError):
            ontology_payload("cat")

    def test_empty_is_preserved_as_empty(self):
        self.assertEqual(ontology_payload({}), {})


class TestResolveModel(unittest.TestCase):
    def test_foundational_is_pass_through(self):
        self.assertEqual(resolve_model("gpt-6-astra-boxes", "foundational"), ("gpt-6-astra-boxes", None))
        self.assertEqual(
            resolve_model("sam3-rle", "foundational", {"outputFormat": "rle"}),
            ("sam3-rle", {"outputFormat": "rle"}),
        )

    def test_roboflow_model_rides_in_model_options(self):
        self.assertEqual(
            resolve_model("proj/3", "roboflow", {"outputFormat": "polygon"}),
            ("custom_roboflow", {"outputFormat": "polygon", "modelId": "proj/3"}),
        )
        self.assertEqual(resolve_model("proj/3", "roboflow"), ("custom_roboflow", {"modelId": "proj/3"}))

    def test_unknown_model_type_raises(self):
        with self.assertRaises(ValueError):
            resolve_model("x", "hosted")
