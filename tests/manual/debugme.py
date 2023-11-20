import sys
import os

thisdir = os.path.dirname(os.path.abspath(__file__))
rootdir = os.path.abspath(f"{thisdir}/../..")
os.environ["ROBOFLOW_CONFIG_DIR"] = f"{thisdir}/data/.config"
sys.path.append(rootdir)

from roboflowpy import _argparser

if __name__ == "__main__":
    parser = _argparser()
    # args = parser.parse_args(["login"])
    # args = parser.parse_args(f"upload {thisdir}/../datasets/chess -w wolfodorpythontests -p chess -f auto".split())
    args = parser.parse_args(
        f"upload {thisdir}/data/cultura-pepino-voc -w wolfodorpythontests -p cultura-pepino-voc -f auto -c 50".split()
    )
    args.func(args)
