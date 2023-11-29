#!/usr/bin/env python3
import argparse
import re

import roboflow
from roboflow.config import DEFAULT_BATCH_NAME


def login(args):
    roboflow.login()


def _parse_url(url):
    regex = r"(?:https?://)?(?:universe|app)\.roboflow\.(?:com|one)/([^/]+)/([^/]+)(?:/dataset)?(?:/(\d+))?|([^/]+)/([^/]+)(?:/(\d+))?"
    match = re.match(regex, url)
    if match:
        organization = match.group(1) or match.group(4)
        dataset = match.group(2) or match.group(5)
        version = match.group(3) or match.group(
            6
        )  # This can be None if not present in the URL
        return organization, dataset, version
    return None, None, None


def download(args):
    rf = roboflow.Roboflow()
    w, p, v = _parse_url(args.datasetUrl)
    project = rf.workspace(w).project(p)
    if not v:
        versions = project.versions()
        if not versions:
            print(f"project {p} does not have any version. exiting")
            exit(1)
        version = versions[-1]
        print(f"Version not provided. Downloading last one ({version.version})")
    else:
        version = project.version(int(v))
    version.download(args.format, location=args.location, overwrite=True)


def import_dataset(args):
    rf = roboflow.Roboflow()
    workspace = rf.workspace(args.workspace)
    workspace.upload_dataset(
        dataset_path=args.folder,
        dataset_format=args.format,
        project_name=args.project,
        num_workers=args.concurrency,
    )


def upload_image(args):
    rf = roboflow.Roboflow()
    workspace = rf.workspace(args.workspace)
    project = workspace.project(args.project)
    project.single_upload(
        image_path=args.imagefile,
        annotation_path=args.annotation,
        annotation_labelmap=args.labelmap,
        split=args.split,
        num_retry_uploads=args.num_retries,
        batch_name=args.batch,
        tag_names=args.tag_names.split(",") if args.tag_names else [],
        is_prediction=args.is_prediction,
    )


def _argparser():
    parser = argparse.ArgumentParser(
        description="Welcome to the roboflow CLI: computer vision at your fingertips 🪄"
    )
    subparsers = parser.add_subparsers(title="subcommands")
    _add_login_parser(subparsers)
    _add_download_parser(subparsers)
    _add_upload_parser(subparsers)
    _add_import_parser(subparsers)
    return parser


def _add_download_parser(subparsers):
    download_parser = subparsers.add_parser(
        "download",
        help="Download a dataset version from your workspace or Roboflow Universe.",
    )
    download_parser.add_argument(
        "datasetUrl", help="Dataset URL (e.g., `roboflow-100/cells-uyemf/2`)"
    )
    download_parser.add_argument(
        "-f",
        dest="format",
        default="voc",
        help="Specify the format to download the version. Available options: [coco, yolov5pytorch, yolov7pytorch, my-yolov6, darknet, voc, tfrecord, createml, clip, multiclass, coco-segmentation, yolo5-obb, png-mask-semantic, yolov8]",
    )
    download_parser.add_argument(
        "-l", dest="location", help="Location to download the dataset"
    )
    download_parser.set_defaults(func=download)


def _add_upload_parser(subparsers):
    upload_parser = subparsers.add_parser(
        "upload", help="Upload a single image to a dataset"
    )
    upload_parser.add_argument(
        "imagefile",
        help="path to image file",
    )
    upload_parser.add_argument(
        "-w",
        dest="workspace",
        help="specify a workspace url or id (will use default workspace if not specified)",
    )
    upload_parser.add_argument(
        "-p",
        dest="project",
        help="project_id to upload the image into",
    )
    upload_parser.add_argument(
        "-a",
        dest="annotation",
        help="path to annotation file (optional)",
    )
    upload_parser.add_argument(
        "-m",
        dest="labelmap",
        help="path to labelmap file (optional)",
    )
    upload_parser.add_argument(
        "-s",
        dest="split",
        help="split set (train, valid, test) - optional",
        default="train",
    )
    upload_parser.add_argument(
        "-r",
        dest="num_retries",
        help="Retry failed uploads this many times (default: 0)",
        type=int,
        default=0,
    )
    upload_parser.add_argument(
        "-b",
        dest="batch",
        help="Batch name to upload to (optional)",
        default=DEFAULT_BATCH_NAME,
    )
    upload_parser.add_argument(
        "-t",
        dest="tag_names",
        help="Tag names to apply to the image (optional)",
    )
    upload_parser.add_argument(
        "-i",
        dest="is_prediction",
        help="Whether this upload is a prediction (optional)",
        action="store_true",
    )
    upload_parser.set_defaults(func=upload_image)


def _add_import_parser(subparsers):
    import_parser = subparsers.add_parser(
        "import", help="Import a dataset from a local folder"
    )
    import_parser.add_argument(
        "folder",
        help="filesystem path to a folder that contains your dataset",
    )
    import_parser.add_argument(
        "-w",
        dest="workspace",
        help="specify a workspace url or id (will use default workspace if not specified)",
    )
    import_parser.add_argument(
        "-p",
        dest="project",
        help="project will be created if it does not exist",
    )
    import_parser.add_argument(
        "-c",
        dest="concurrency",
        type=int,
        help="how many image uploads to perform concurrently (default: 10)",
        default=10,
    )
    import_parser.add_argument(
        "-f",
        dest="format",
        help="dataset format. Valid options are [voc, yolov8, yolov5, auto] (use auto for autodetect)",
        default="auto",
    )
    import_parser.set_defaults(func=import_dataset)


def _add_login_parser(subparsers):
    login_parser = subparsers.add_parser("login", help="Log in to Roboflow")
    login_parser.set_defaults(func=login)


def main():
    parser = _argparser()
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
