import argparse
import roboflow

def login(args):
    roboflow.login()

def download(args):
    rf = roboflow.Roboflow()
    w, p, v = args.datasetUrl.split("/")
    format, location = args.f, args.l
    project = rf.workspace(w).project(p)
    project.version(int(v)).download(format, location=location, overwrite=True)

def upload(args):
    rf = roboflow.Roboflow()
    f, w, p, folder = args.f, args.w, args.p, args.folder
    workspace = rf.workspace(w)
    workspace.upload_dataset(
        dataset_path=folder,
        dataset_format=f,
        project_name=p,
    )


def main():
    parser = argparse.ArgumentParser(description="main description")
    subparsers = parser.add_subparsers(title="subcommands")

    login_parser = subparsers.add_parser("login", help="Log in to Roboflow")
    login_parser.set_defaults(func=login)

    download_parser = subparsers.add_parser("download", help="Download a dataset version from your workspace or Roboflow Universe.")
    download_parser.add_argument("datasetUrl", help="Dataset URL (e.g., `roboflow-100/cells-uyemf/2`)")
    download_parser.add_argument("-f", 
                                 choices=["coco", "yolov5pytorch", "yolov7pytorch", "my-yolov6", "darknet", "voc", "tfrecord", 
                                          "createml", "clip", "multiclass", "coco-segmentation", "yolo5-obb", "png-mask-semantic"], 
                                 help="Specify the format to download the version in (default: interactive prompt)")
    download_parser.add_argument("-l", help="Location to download the dataset")
    download_parser.set_defaults(func=download)

    upload_parser = subparsers.add_parser("upload", help="Upload a dataset")
    upload_parser.add_argument("folder", help="filesystem path to a folder that contains your dataset")
    upload_parser.add_argument("-w", help="workspace url")
    upload_parser.add_argument("-p", help="Project name")
    upload_parser.add_argument("-f", choices=["voc", "yolov8", "yolov5"], help="format")
    upload_parser.set_defaults(func=upload)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
