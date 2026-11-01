import json
import os
import tempfile
import unittest
from os.path import abspath, dirname
from pathlib import Path

from roboflow.util import folderparser

thisdir = dirname(abspath(__file__))


class TestFolderParser(unittest.TestCase):
    def test_parse_chess(self):
        chessfolder = f"{thisdir}/../datasets/chess"
        parsed = folderparser.parsefolder(chessfolder)
        _assertJsonMatchesFile(parsed, f"{thisdir}/../output/parse-chess.json")

    def test_parse_sharks_coco(self):
        sharksfolder = f"{thisdir}/../datasets/sharks-tiny-coco"
        parsed = folderparser.parsefolder(sharksfolder)
        testImagePath = "/train/sharks_mp4-20_jpg.rf.90ba2e8e9ca0613f71359efb7ed48b26.jpg"
        testImage = [i for i in parsed["images"] if i["file"] == testImagePath][0]
        assert len(json.loads(testImage["annotationfile"]["rawText"])["annotations"]) == 5

    def test_parse_sharks_createml(self):
        sharksfolder = f"{thisdir}/../datasets/sharks-tiny-createml"
        parsed = folderparser.parsefolder(sharksfolder)
        testImagePath = "/train/sharks_mp4-20_jpg.rf.5359121123e86e016401ea2731e47949.jpg"
        testImage = [i for i in parsed["images"] if i["file"] == testImagePath][0]
        imgParsedAnnotations = json.loads(testImage["annotationfile"]["rawText"])
        assert len(imgParsedAnnotations) == 1
        imgReference = imgParsedAnnotations[0]
        assert len(imgReference["annotations"]) == 5

    def test_parse_sharks_yolov9(self):
        def test(sharksfolder):
            parsed = folderparser.parsefolder(sharksfolder)
            testImagePath = "/train/images/sharks_mp4-20_jpg.rf.5359121123e86e016401ea2731e47949.jpg"
            testImage = [i for i in parsed["images"] if i["file"] == testImagePath][0]
            expectAnnotationFile = "/train/labels/sharks_mp4-20_jpg.rf.5359121123e86e016401ea2731e47949.txt"
            assert testImage["annotationfile"]["file"] == expectAnnotationFile
            assert testImage["annotationfile"]["labelmap"] == {0: "fish", 1: "primary", 2: "shark"}

        test(f"{thisdir}/../datasets/sharks-tiny-yolov9")
        test(f"{thisdir}/../datasets/sharks-tiny-yolov9/")  # this was a bug once, can you believe it?

    def test_parse_mosquitos_csv(self):
        folder = f"{thisdir}/../datasets/mosquitos"
        parsed = folderparser.parsefolder(folder)
        testImagePath = "/train_10308.jpeg"
        testImage = [i for i in parsed["images"] if i["file"] == testImagePath][0]
        assert testImage["annotationfile"]["name"] == "annotation.csv"
        expected = "img_fName,img_w,img_h,class_label,bbx_xtl,bbx_ytl,bbx_xbr,bbx_ybr\n"
        expected += "train_10308.jpeg,1058,943,japonicus/koreicus,28,187,908,815\n"
        assert testImage["annotationfile"]["rawText"] == expected

    def test_paligemma_format(self):
        folder = f"{thisdir}/../datasets/paligemma"
        parsed = folderparser.parsefolder(folder)
        testImagePath = "/dataset/de48275e1ff70fab78bee31e09fc896d_png.rf.01a97b1ad053aa1e6525ac0451cee8b7.jpg"
        testImage = [i for i in parsed["images"] if i["file"] == testImagePath][0]
        assert testImage["annotationfile"]["name"] == "annotation.jsonl"
        expected = (
            '{"image": "de48275e1ff70fab78bee31e09fc896d_png.rf.01a97b1ad053aa1e6525ac0451cee8b7.jpg",'
            ' "prefix": "Which sector had the highest ROI in 2013?", "suffix": "Retail"}\n'
            '{"image": "de48275e1ff70fab78bee31e09fc896d_png.rf.01a97b1ad053aa1e6525ac0451cee8b7.jpg",'
            ' "prefix": "Which sector had the highest ROI in 2014?", "suffix": "Electronics"}'
        )
        assert testImage["annotationfile"]["rawText"] == expected

    def test_parse_classification_folder_structure(self):
        classification_folder = f"{thisdir}/../datasets/corrosion-singlelabel-classification"
        parsed = folderparser.parsefolder(classification_folder, is_classification=False)
        for img in parsed["images"]:
            self.assertIsNone(img.get("annotationfile"))

        parsed_classification = folderparser.parsefolder(classification_folder, is_classification=True)
        corrosion_images = [i for i in parsed_classification["images"] if "Corrosion" in i["dirname"]]
        self.assertTrue(len(corrosion_images) > 0)
        for img in corrosion_images:
            self.assertIsNotNone(img.get("annotationfile"))
            self.assertEqual(img["annotationfile"]["type"], "classification_folder")
            self.assertEqual(img["annotationfile"]["classification_label"], "Corrosion")
        no_corrosion_images = [i for i in parsed_classification["images"] if "no-corrosion" in i["dirname"]]
        self.assertTrue(len(no_corrosion_images) > 0)
        for img in no_corrosion_images:
            self.assertIsNotNone(img.get("annotationfile"))
            self.assertEqual(img["annotationfile"]["type"], "classification_folder")
            self.assertEqual(img["annotationfile"]["classification_label"], "no-corrosion")

    def test_parse_multilabel_classification_csv(self):
        folder = f"{thisdir}/../datasets/skinproblem-multilabel-classification"
        parsed = folderparser.parsefolder(folder, is_classification=True)
        images = {img["name"]: img for img in parsed["images"]}
        img1 = images.get("101_jpg.rf.ffb91e580c891eb04b715545274b2469.jpg")
        self.assertIsNotNone(img1)
        self.assertEqual(img1["annotationfile"]["type"], "classification_multilabel")
        self.assertEqual(set(img1["annotationfile"]["labels"]), {"Blackheads"})

    def test_coco_with_subdir_file_name_should_match_annotations(self):
        # COCO file_name includes a subdirectory, but the actual image is at dataset root.
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested image path: /2/100002/img.jpeg
            image_name = "example_2_100002_02f2f7c6e15f09b401575ae6.jpeg"
            image_relpath = os.path.join("2", "100002", image_name)
            image_path = os.path.join(tmpdir, image_name)
            # Create an empty image file (content not used by parser)
            open(image_path, "wb").close()

            # Create COCO annotation JSON at dataset root, referencing the image with subdir in file_name
            coco = {
                "info": {},
                "licenses": [],
                "categories": [{"id": 1, "name": "thing"}],
                "images": [
                    {
                        "id": 10000000,
                        "file_name": image_relpath.replace(os.sep, "/"),
                        "width": 800,
                        "height": 533,
                    }
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 10000000,
                        "category_id": 1,
                        "bbox": [10, 10, 100, 50],
                        "area": 5000,
                        "segmentation": [],
                        "iscrowd": 0,
                    }
                ],
            }
            coco_path = os.path.join(tmpdir, "_annotations.coco.json")
            with open(coco_path, "w") as f:
                json.dump(coco, f)

            parsed = folderparser.parsefolder(tmpdir)
            # Image entries store file with a leading slash relative to root
            expected_file_key = f"/{image_name}"
            img_entries = [i for i in parsed["images"] if i["file"] == expected_file_key]
            self.assertTrue(len(img_entries) == 1)
            img_entry = img_entries[0]

            # Expect annotationfile to be populated, but this currently fails due to basename-only matching
            self.assertIsNotNone(img_entry.get("annotationfile"))

    def test_coco_root_annotation_matches_images_in_subdirs(self):
        """Test that COCO annotation at root can match images in subdirectories.

        This tests the fix for the bug where annotation file dirname (/) didn't match
        image dirname (/1/100001), causing annotations to not be found.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create image in subdirectory
            subdir = os.path.join(tmpdir, "1", "100001")
            os.makedirs(subdir, exist_ok=True)
            image_name = "image.jpeg"
            image_path = os.path.join(subdir, image_name)
            open(image_path, "wb").close()

            # Create COCO annotation at root referencing image with subdirectory path
            coco = {
                "info": {},
                "licenses": [],
                "categories": [{"id": 1, "name": "object"}],
                "images": [
                    {
                        "id": 10000000,
                        "file_name": "1/100001/image.jpeg",
                        "width": 800,
                        "height": 600,
                    }
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 10000000,
                        "category_id": 1,
                        "bbox": [10, 20, 100, 200],
                        "area": 20000,
                        "segmentation": [[10, 20, 110, 20, 110, 220, 10, 220]],
                        "iscrowd": 0,
                    }
                ],
            }
            coco_path = os.path.join(tmpdir, "_annotations.coco.json")
            with open(coco_path, "w") as f:
                json.dump(coco, f)

            parsed = folderparser.parsefolder(tmpdir)

            # Find the image
            img_entries = [i for i in parsed["images"] if image_name in i["file"]]
            self.assertEqual(len(img_entries), 1, "Should find exactly one image")
            img_entry = img_entries[0]

            # Verify annotation was matched
            self.assertIsNotNone(img_entry.get("annotationfile"), "Image should have annotation")

            # Verify annotation content
            ann_data = json.loads(img_entry["annotationfile"]["rawText"])
            self.assertEqual(len(ann_data["images"]), 1, "Should have one image reference")
            self.assertEqual(len(ann_data["annotations"]), 1, "Should have one annotation")
            self.assertEqual(ann_data["annotations"][0]["bbox"], [10, 20, 100, 200])


class TestCocoImageMatching(unittest.TestCase):
    def _parse(self, image_paths, documents):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for path in image_paths:
                image = root / path
                image.parent.mkdir(parents=True, exist_ok=True)
                image.touch()
            for path, references in documents.items():
                annotation = root / path
                annotation.parent.mkdir(parents=True, exist_ok=True)
                data = {
                    "info": {},
                    "licenses": [],
                    "categories": [{"id": category, "name": str(category)} for _, category in references],
                    "images": [
                        {"id": i, "file_name": name, "width": 4, "height": 3} for i, (name, _) in enumerate(references)
                    ],
                    "annotations": [
                        {
                            "id": i,
                            "image_id": i,
                            "category_id": category,
                            "segmentation": {"size": [3, 4], "counts": [0, 3, 3, 3, 3]},
                        }
                        for i, (_, category) in enumerate(references)
                    ],
                }
                annotation.write_text(json.dumps(data), encoding="utf-8")
            parsed = folderparser.parsefolder(directory)
            return {image["file"]: json.loads(image["annotationfile"]["rawText"]) for image in parsed["images"]}

    def test_split_annotations_match_their_own_images(self):
        result = self._parse(
            ["train/same.jpg", "valid/same.jpg"],
            {
                "train/_annotations.coco.json": [("same.jpg", 11)],
                "valid/_annotations.coco.json": [("same.jpg", 22)],
            },
        )
        self.assertEqual(result["/train/same.jpg"]["annotations"][0]["category_id"], 11)
        self.assertEqual(result["/valid/same.jpg"]["annotations"][0]["category_id"], 22)
        self.assertEqual(
            result["/valid/same.jpg"]["annotations"][0]["segmentation"],
            {"size": [3, 4], "counts": [0, 3, 3, 3, 3]},
        )

    def test_one_coco_file_can_reference_duplicate_basenames(self):
        result = self._parse(
            ["train/same.jpg", "valid/same.jpg"],
            {"_annotations.coco.json": [("train/same.jpg", 11), ("valid/same.jpg", 22)]},
        )
        self.assertEqual(result["/train/same.jpg"]["annotations"][0]["category_id"], 11)
        self.assertEqual(result["/valid/same.jpg"]["annotations"][0]["category_id"], 22)

    def test_nested_coco_paths_are_normalized(self):
        for name in ("images/same.jpg", "images\\same.jpg", "./images/same.jpg", "train/images/same.jpg"):
            with self.subTest(name=name):
                result = self._parse(["train/images/same.jpg"], {"train/_annotations.coco.json": [(name, 11)]})
                self.assertEqual(result["/train/images/same.jpg"]["images"][0]["file_name"], name)

    def test_unique_basename_and_stem_fallbacks_still_work(self):
        for name in ("old/path/same.jpg", "C:\\old\\path\\same.jpg", "old/path/same.png"):
            with self.subTest(name=name):
                result = self._parse(["same.jpg"], {"_annotations.coco.json": [(name, 11)]})
                self.assertEqual(result["/same.jpg"]["annotations"][0]["category_id"], 11)

    def test_ambiguous_image_fallback_raises(self):
        with self.assertRaisesRegex(ValueError, "[Aa]mbiguous.*same"):
            self._parse(
                ["train/same.jpg", "valid/same.jpg"],
                {"_annotations.coco.json": [("old/same.jpg", 11)]},
            )

    def test_competing_coco_references_raise(self):
        cases = [
            {"first.coco.json": [("same.jpg", 11)], "second.coco.json": [("same.jpg", 22)]},
            {"_annotations.coco.json": [("old/same.jpg", 11), ("other/same.jpg", 22)]},
        ]
        for documents in cases:
            with self.subTest(documents=documents), self.assertRaisesRegex(ValueError, "[Aa]mbiguous.*same"):
                self._parse(["same.jpg"], documents)


def _assertJsonMatchesFile(actual, filename):
    with open(filename) as file:
        expected = json.load(file)
        return actual == expected
