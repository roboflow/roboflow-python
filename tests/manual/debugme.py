import os
import sys

thisdir = os.path.dirname(os.path.abspath(__file__))
os.environ["ROBOFLOW_CONFIG_DIR"] = f"{thisdir}/data/.config"

from roboflow.roboflowpy import _argparser  # noqa: E402

# import requests
# requests.urllib3.disable_warnings()

rootdir = os.path.abspath(f"{thisdir}/../..")
sys.path.append(rootdir)

if __name__ == "__main__":
    parser = _argparser()
    # args = parser.parse_args(["login"])
    # args = parser.parse_args(f"upload {thisdir}/../datasets/chess -w wolfodorpythontests -p chess -f auto".split())   # noqa: E501 // docs
    args = parser.parse_args(
        # ["login"]
        # "download https://universe.roboflow.com/gdit/aerial-airport".split()
        # "project list -w wolfodorpythontests".split()
        # "project get cultura-pepino-dark".split()
        # "workspace list".split()
        # "workspace get wolfodorpythontests".split()
        # f"infer -w jacob-solawetz -m rock-paper-scissors-slim/5 -c .01 {thisdir}/data/scissors.png".split()  # noqa: E501 // docs
        # f"infer -w roboflow-6tyri -m usa-states/3 -c .94 -t instance-segmentation {thisdir}/data/unitedstates.jpg".split()  # noqa: E501 // docs
        # f"infer -w naumov-igor-segmentation -m car-segmetarion/2 -t semantic-segmentation {thisdir}/data/car.jpg".split()  # noqa: E501 // docs
        # f"import {thisdir}/data/cultura-pepino-voc -w wolfodorpythontests -p cultura-pepino-voc -f auto -c 50".split()   # noqa: E501 // docs
        # f"import {thisdir}/data/0311fisheye -w wolfodorpythontests -p 0311fisheye -f auto -c 50".split()   # noqa: E501 // docs
        # f"upload {thisdir}/data/cultura-pepino-darknet/train/10_jpg.rf.2b3a401b0ffd8482e52137ad22faa14f.jpg -a {thisdir}/data/cultura-pepino-darknet/train/10_jpg.rf.2b3a401b0ffd8482e52137ad22faa14f.txt -m {thisdir}/data/cultura-pepino-darknet/train/_darknet.labels -w wolfodorpythontests -p cultura-pepino-darknet -r 3".split()   # noqa: E501 // docs
        # f"upload -p ordered-uploading {thisdir}/data/ordered-upload/1.jpg".split()
        # f"import -p ordered-uploading {thisdir}/data/ordered-upload".split()
        # f"import -p yellow-auto {thisdir}/data/ordered-upload".split()
        # f" {thisdir}/data/cultura-pepino-darknet -w wolfodorpythontests -p cultura-pepino-darknet -f auto -c 100".split()   # noqa: E501 // docs
        # f"import {thisdir}/data/cultura-pepino-darknet -w wolfodorpythontests -p yellow-auto -f auto -c 100".split()  # noqa: E501 // docs
        # f"import {thisdir}/data/cultura-pepino-clip -w wolfodorpythontests -p yellow-auto -f auto -c 100".split()  # noqa: E501 // docs
        # f"import {thisdir}/data/cultura-pepino-voc -w wolfodorpythontests -p yellow-auto -f auto -c 100".split()  # noqa: E501 // docs
        # f"import {thisdir}/data/cultura-pepino-coco -w wolfodorpythontests -p yellow-auto -f auto -c 100".split()  # noqa: E501 // docs
        # f"import {thisdir}/data/cultura-pepino-yolov8 -w wolfodorpythontests -p yellow-auto -f auto -c 100".split()  # noqa: E501 // docs
        # f"import {thisdir}/data/cultura-pepino-yolov8_voc -w wolfodorpythontests -p yellow-auto -f auto -c 100".split()  # noqa: E501 // docs
        f"import {thisdir}/data/cultura-pepino-yolov5pytorch -w wolfodorpythontests -p yellow-auto -f auto -c 100".split()  # noqa: E501 // docs
    )
    args.func(args)
