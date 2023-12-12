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

    # def test_parse_sharks(self):
    #     sharksfolder = f"{thisdir}/../datasets/sharks-tiny-coco"
    #     parsed = folderparser.parsefolder(sharksfolder)
    #     print(parsed)


def _assertJsonMatchesFile(actual, filename):
    with open(filename, "r") as file:
        expected = json.load(file)
        return actual == expected
