import os
import sys

thisdir = os.path.dirname(os.path.abspath(__file__))
os.environ["ROBOFLOW_CONFIG_DIR"] = f"{thisdir}/data/.config"

from roboflow.roboflowpy import _argparser  # flake8: noqa: E402

# import requests
# requests.urllib3.disable_warnings()

rootdir = os.path.abspath(f"{thisdir}/../..")
sys.path.append(rootdir)

if __name__ == "__main__":
    parser = _argparser()
    # args = parser.parse_args(["login"])
    # args = parser.parse_args(f"upload {thisdir}/../datasets/chess -w wolfodorpythontests -p chess -f auto".split())   # noqa: E501 // docs
    args = parser.parse_args(
        # "download https://universe.roboflow.com/gdit/aerial-airport".split()
        # "project list".split()
        # "project get cultura-pepino-dark".split()
        # "workspace list".split()
        "workspace get wolfodorpythontests".split()
        # f"infer -m cultura-pepino-voc/1 {thisdir}/data/cultura-pepino-coco/test/21_jpg.rf.d3a7fb90b0fafc6541378a6b362ab295.jpg".split()
        # f"import {thisdir}/data/cultura-pepino-voc -w wolfodorpythontests -p cultura-pepino-voc -f auto -c 50".split()   # noqa: E501 // docs
        # f"import {thisdir}/data/cultura-pepino-darknet -w wolfodorpythontests -p cultura-pepino-darknet -f auto -c 100".split()   # noqa: E501 // docs
        # f"import {thisdir}/data/0311fisheye -w wolfodorpythontests -p 0311fisheye -f auto -c 50".split()   # noqa: E501 // docs
        # f"upload {thisdir}/data/cultura-pepino-darknet/train/10_jpg.rf.2b3a401b0ffd8482e52137ad22faa14f.jpg -a {thisdir}/data/cultura-pepino-darknet/train/10_jpg.rf.2b3a401b0ffd8482e52137ad22faa14f.txt -m {thisdir}/data/cultura-pepino-darknet/train/_darknet.labels -w wolfodorpythontests -p cultura-pepino-darknet -r 3".split()   # noqa: E501 // docs
        # f" {thisdir}/data/cultura-pepino-darknet -w wolfodorpythontests -p cultura-pepino-darknet -f auto -c 100".split()   # noqa: E501 // docs
    )
    args.func(args)
