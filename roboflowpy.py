import argparse
import roboflow


def login(args):
    roboflow.login()


def download(args):
    rf = roboflow.Roboflow()
    w, p, v = args.datasetUrl.split("/")
    project = rf.workspace(w).project(p)
    project.version(int(v)).download(
        args.format, location=args.location, overwrite=True
    )


def import_dataset(args):
    rf = roboflow.Roboflow()
    workspace = rf.workspace(args.workspace)
    workspace.upload_dataset(
        dataset_path=args.folder,
        dataset_format=args.format,
        project_name=args.project,
        num_workers=args.concurrency,
    )


def _argparser():
    parser = argparse.ArgumentParser(description="main description")
    subparsers = parser.add_subparsers(title="subcommands")
    login_parser = subparsers.add_parser("login", help="Log in to Roboflow")
    login_parser.set_defaults(func=login)
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
        help="Specify the format to download the version. Available options: [coco, yolov5pytorch, yolov7pytorch, my-yolov6, darknet, voc, tfrecord, createml, clip, multiclass, coco-segmentation, yolo5-obb, png-mask-semantic, yolov8]",
    )
    download_parser.add_argument(
        "-l", dest="location", help="Location to download the dataset"
    )
    download_parser.set_defaults(func=download)
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
        help="project url or id (or the program will prompt you to select which project in your workspace to upload to)",
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
    return parser


def main():
    parser = _argparser()
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
