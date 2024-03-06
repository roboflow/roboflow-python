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
        assert len(testImage["annotationfile"]["parsed"]["annotations"]) == 5

    def test_parse_sharks_createml(self):
        sharksfolder = f"{thisdir}/../datasets/sharks-tiny-createml"
        parsed = folderparser.parsefolder(sharksfolder)
        testImagePath = "/train/sharks_mp4-20_jpg.rf.5359121123e86e016401ea2731e47949.jpg"
        testImage = [i for i in parsed["images"] if i["file"] == testImagePath][0]
        assert len(testImage["annotationfile"]["parsed"]) == 1
        imgReference = testImage["annotationfile"]["parsed"][0]
        assert len(imgReference["annotations"]) == 5

    def test_parse_sharks_yolov9(self):
        sharksfolder = f"{thisdir}/../datasets/sharks-tiny-yolov9"
        parsed = folderparser.parsefolder(sharksfolder)
        testImagePath = "/train/images/sharks_mp4-20_jpg.rf.5359121123e86e016401ea2731e47949.jpg"
        testImage = [i for i in parsed["images"] if i["file"] == testImagePath][0]
        expectAnnotationFile = "/train/labels/sharks_mp4-20_jpg.rf.5359121123e86e016401ea2731e47949.txt"
        assert testImage["annotationfile"]["file"] == expectAnnotationFile
        assert testImage["annotationfile"]["labelmap"] == {0: "fish", 1: "primary", 2: "shark"}


def _assertJsonMatchesFile(actual, filename):
    with open(filename, "r") as file:
        expected = json.load(file)
        return actual == expected
