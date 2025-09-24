import json
import unittest
from os.path import abspath, dirname

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


def _assertJsonMatchesFile(actual, filename):
    with open(filename) as file:
        expected = json.load(file)
        return actual == expected
