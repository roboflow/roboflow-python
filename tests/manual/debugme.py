import os
import sys

# import requests

# requests.urllib3.disable_warnings()

thisdir = os.path.dirname(os.path.abspath(__file__))
rootdir = os.path.abspath(f"{thisdir}/../..")
os.environ["ROBOFLOW_CONFIG_DIR"] = f"{thisdir}/data/.config"
sys.path.append(rootdir)

from roboflow.roboflowpy import _argparser

if __name__ == "__main__":
    parser = _argparser()
    # args = parser.parse_args(["login"])
    # args = parser.parse_args(f"upload {thisdir}/../datasets/chess -w wolfodorpythontests -p chess -f auto".split())
    args = parser.parse_args(
        f"import {thisdir}/data/cultura-pepino-voc -w wolfodorpythontests -p cultura-pepino-voc -f auto -c 50".split()
        # f"import {thisdir}/data/cultura-pepino-darknet -w wolfodorpythontests -p cultura-pepino-darknet -f auto -c 100".split()
        # f"import {thisdir}/data/0311fisheye -w wolfodorpythontests -p 0311fisheye -f auto -c 50".split()
        # f"upload {thisdir}/data/cultura-pepino-darknet/train/10_jpg.rf.2b3a401b0ffd8482e52137ad22faa14f.jpg -a {thisdir}/data/cultura-pepino-darknet/train/10_jpg.rf.2b3a401b0ffd8482e52137ad22faa14f.txt -m {thisdir}/data/cultura-pepino-darknet/train/_darknet.labels -w wolfodorpythontests -p cultura-pepino-darknet -r 3".split()
        # f" {thisdir}/data/cultura-pepino-darknet -w wolfodorpythontests -p cultura-pepino-darknet -f auto -c 100".split()
    )
    args.func(args)
