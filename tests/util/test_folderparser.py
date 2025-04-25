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


def _assertJsonMatchesFile(actual, filename):
    with open(filename) as file:
        expected = json.load(file)
        return actual == expected
