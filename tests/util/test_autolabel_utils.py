"""Unit tests for roboflow.util.autolabel_utils."""

import base64
import os
import tempfile
import unittest

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


class TestOntologyPayload(unittest.TestCase):
    def test_none_stays_none(self):
        self.assertIsNone(ontology_payload(None))

    def test_class_to_prompt_mapping_is_serialized_explicitly(self):
        # The wire form names both sides, so the API never has to guess which
        # of the two strings is the class and which is the prompt.
        self.assertEqual(
            ontology_payload({"cat": "a cat", "dog": "a dog"}),
            [{"class": "cat", "prompt": "a cat"}, {"class": "dog", "prompt": "a dog"}],
        )

    def test_list_of_classes_becomes_identity_prompts(self):
        self.assertEqual(
            ontology_payload(["cat", "dog"]),
            [{"class": "cat", "prompt": "cat"}, {"class": "dog", "prompt": "dog"}],
        )

    def test_one_class_may_carry_several_prompts(self):
        # The reason the list form is accepted at all: {"cat": ...} has room for
        # exactly one prompt, because dict keys are unique.
        self.assertEqual(
            ontology_payload(
                [
                    {"class": "cat", "prompt": "kitten"},
                    {"class": "cat", "prompt": "tabby"},
                ]
            ),
            [{"class": "cat", "prompt": "kitten"}, {"class": "cat", "prompt": "tabby"}],
        )

    def test_list_entries_default_the_prompt_to_the_class(self):
        self.assertEqual(
            ontology_payload([{"class": "cat"}, "dog"]),
            [{"class": "cat", "prompt": "cat"}, {"class": "dog", "prompt": "dog"}],
        )

    def test_one_prompt_claimed_by_two_classes_is_rejected(self):
        # The API keys its ontology by prompt, so it would keep "dog" and drop
        # "cat" without saying so. Name the collision instead.
        with self.assertRaises(ValueError) as ctx:
            ontology_payload([{"class": "cat", "prompt": "animal"}, {"class": "dog", "prompt": "animal"}])
        self.assertIn("animal", str(ctx.exception))
        self.assertIn("cat", str(ctx.exception))
        self.assertIn("dog", str(ctx.exception))

    def test_a_repeated_class_prompt_pair_is_not_a_collision(self):
        self.assertEqual(
            ontology_payload([{"class": "cat", "prompt": "cat"}, "cat"]),
            [{"class": "cat", "prompt": "cat"}, {"class": "cat", "prompt": "cat"}],
        )

    def test_bare_string_is_rejected_rather_than_iterated_per_character(self):
        with self.assertRaises(ValueError):
            ontology_payload("cat")

    def test_malformed_entry_is_rejected(self):
        with self.assertRaises(ValueError):
            ontology_payload([{"prompt": "a cat"}])

    def test_empty_is_preserved_as_empty(self):
        self.assertEqual(ontology_payload({}), [])


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
