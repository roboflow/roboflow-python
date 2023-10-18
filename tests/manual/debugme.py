import sys
import os

thisdir = os.path.dirname(os.path.abspath(__file__))
rootdir = os.path.abspath(f"{thisdir}/../..")
os.environ["ROBOFLOW_CONFIG_DIR"] = f"{thisdir}/data/.config"
sys.path.append(rootdir)

from roboflowpy import _argparser

if __name__ == "__main__":
    parser = _argparser()
    args = parser.parse_args(["login"])
    # args = parser.parse_args(
    #     [
    #         "upload",
    #         "./data/cultura-pepino-voc",
    #         "-w",
    #         "wolfodorpythontests",
    #         "-p",
    #         "cultura-pepino-upload-test-voc",
    #         "-f",
    #         "voc",
    #     ]
    # )
    args.func(args)
