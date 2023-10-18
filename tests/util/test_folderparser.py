import unittest
from os.path import dirname, abspath
from roboflow.util import folderparser

thisdir = dirname(abspath(__file__))


class TestFolderParser(unittest.TestCase):
    def test_parse_chess(self):
        chessfolder = f"{thisdir}/../datasets/chess"
        parsed = folderparser.parsefolder(chessfolder)
        print(parsed)

    # def test_parse_sharks(self):
    #     sharksfolder = f"{thisdir}/../datasets/sharks-tiny-coco"
    #     parsed = folderparser.parsefolder(sharksfolder)
    #     print(parsed)
