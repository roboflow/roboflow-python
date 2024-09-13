#!/usr/bin/env python3
import argparse
import json
import re

import roboflow
from roboflow import config as roboflow_config
from roboflow import deployment
from roboflow.adapters import rfapi
from roboflow.config import APP_URL, get_conditional_configuration_variable, load_roboflow_api_key
from roboflow.models.classification import ClassificationModel
from roboflow.models.instance_segmentation import InstanceSegmentationModel
from roboflow.models.keypoint_detection import KeypointDetectionModel
from roboflow.models.object_detection import ObjectDetectionModel
from roboflow.models.semantic_segmentation import SemanticSegmentationModel


def login(args):
    roboflow.login()


def _parse_url(url):
    regex = r"(?:https?://)?(?:universe|app)\.roboflow\.(?:com|one)/([^/]+)/([^/]+)(?:/dataset)?(?:/(\d+))?|([^/]+)/([^/]+)(?:/(\d+))?"  # noqa: E501
    match = re.match(regex, url)
    if match:
        organization = match.group(1) or match.group(4)
        dataset = match.group(2) or match.group(5)
        version = match.group(3) or match.group(6)  # This can be None if not present in the URL
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
    api_key = load_roboflow_api_key(args.workspace)
    rf = roboflow.Roboflow(api_key)
    workspace = rf.workspace(args.workspace)
    workspace.upload_dataset(
        dataset_path=args.folder,
        project_name=args.project,
        num_workers=args.concurrency,
        batch_name=args.batch_name,
        num_retries=args.num_retries,
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


def upload_model(args):
    rf = roboflow.Roboflow(args.api_key)
    workspace = rf.workspace(args.workspace)
    project = workspace.project(args.project)
    version = project.version(args.version_number)
    print(args.model_type, args.model_path, args.filename)
    version.deploy(str(args.model_type), str(args.model_path), str(args.filename))


def list_projects(args):
    rf = roboflow.Roboflow()
    workspace = rf.workspace(args.workspace)
    projects = workspace.project_list
    for p in projects:
        print()
        print(p["name"])
        print(f"  link: {APP_URL}/{p['id']}")
        print(f"  id: {p['id']}")
        print(f"  type: {p['type']}")
        print(f"  versions: {p['versions']}")
        print(f"  images: {p['images']}")
        print(f"  classes: {p['classes'].keys()}")


def list_workspaces(args):
    workspaces = roboflow_config.RF_WORKSPACES.values()
    rf_workspace = get_conditional_configuration_variable("RF_WORKSPACE", default=None)
    for w in workspaces:
        print()
        print(f"{w['name']}{' (default workspace)' if w['url'] == rf_workspace else ''}")
        print(f"  link: {APP_URL}/{w['url']}")
        print(f"  id: {w['url']}")


def get_workspace(args):
    api_key = load_roboflow_api_key(args.workspaceId)
    workspace_json = rfapi.get_workspace(api_key, args.workspaceId)
    print(json.dumps(workspace_json, indent=2))


def run_video_inference_api(args):
    rf = roboflow.Roboflow(args.api_key)
    project = rf.workspace().project(args.project)
    version = project.version(args.version_number)
    model = project.version(version).model

    # model = VideoInferenceModel(args.api_key, project.id, version.version, project.id)  # Pass dataset_id
    # Pass model_id and version
    job_id, signed_url, expire_time = model.predict_video(
        args.video_file,
        args.fps,
        prediction_type="batch-video",
    )
    results = model.poll_until_video_results(job_id)
    with open("test_video.json", "w") as f:
        json.dump(results, f)


def get_workspace_project_version(args):
    # api_key = load_roboflow_api_key(args.workspaceId)
    rf = roboflow.Roboflow(args.api_key)
    workspace = rf.workspace()
    print("workspace", workspace)
    project = workspace.project(args.project)
    print("project", project)
    version = project.version(args.version_number)
    print("version", version)


def get_project(args):
    workspace_url = args.workspace or get_conditional_configuration_variable("RF_WORKSPACE", default=None)
    api_key = load_roboflow_api_key(workspace_url)
    dataset_json = rfapi.get_project(api_key, workspace_url, args.projectId)
    print(json.dumps(dataset_json, indent=2))


def infer(args):
    workspace_url = args.workspace or get_conditional_configuration_variable("RF_WORKSPACE", default=None)
    api_key = load_roboflow_api_key(workspace_url)
    project_url = f"{workspace_url}/{args.model}"
    projectType = args.type
    if not projectType:
        projectId, _ = args.model.split("/")
        dataset_json = rfapi.get_project(api_key, workspace_url, projectId)
        projectType = dataset_json["project"]["type"]
    modelClass = {
        "object-detection": ObjectDetectionModel,
        "classification": ClassificationModel,
        "instance-segmentation": InstanceSegmentationModel,
        "semantic-segmentation": SemanticSegmentationModel,
        "keypoint-detection": KeypointDetectionModel,
    }[projectType]
    model = modelClass(api_key, project_url)
    kwargs = {}
    if args.confidence is not None and projectType in [
        "object-detection",
        "instance-segmentation",
        "semantic-segmentation",
    ]:
        kwargs["confidence"] = int(args.confidence * 100)
    if args.overlap is not None and projectType == "object-detection":
        kwargs["overlap"] = int(args.overlap * 100)
    group = model.predict(args.file, **kwargs)
    print(group)


def _argparser():
    parser = argparse.ArgumentParser(description="Welcome to the roboflow CLI: computer vision at your fingertips 🪄")
    subparsers = parser.add_subparsers(title="subcommands")
    _add_login_parser(subparsers)
    _add_download_parser(subparsers)
    _add_upload_parser(subparsers)
    _add_import_parser(subparsers)
    _add_infer_parser(subparsers)
    _add_projects_parser(subparsers)
    _add_workspaces_parser(subparsers)
    _add_upload_model_parser(subparsers)
    _add_get_workspace_project_version_parser(subparsers)
    _add_run_video_inference_api_parser(subparsers)
    deployment.add_deployment_parser(subparsers)
    _add_whoami_parser(subparsers)

    parser.add_argument("-v", "--version", help="show version info", action="store_true")
    parser.set_defaults(func=show_version)

    return parser


def show_version(args):
    print(roboflow.__version__)


def show_whoami(args):
    RF_WORKSPACES = get_conditional_configuration_variable("workspaces", default={})
    workspaces_by_url = {w["url"]: w for w in RF_WORKSPACES.values()}
    default_workspace_url = get_conditional_configuration_variable("RF_WORKSPACE", default=None)
    default_workspace = workspaces_by_url.get(default_workspace_url, None)
    default_workspace["apiKey"] = "**********"
    print(json.dumps(default_workspace, indent=2))


def _add_whoami_parser(subparsers):
    download_parser = subparsers.add_parser("whoami", help="show current user info")
    download_parser.set_defaults(func=show_whoami)


def _add_download_parser(subparsers):
    download_parser = subparsers.add_parser(
        "download",
        help="Download a dataset version from your workspace or Roboflow Universe.",
    )
    download_parser.add_argument("datasetUrl", help="Dataset URL (e.g., `roboflow-100/cells-uyemf/2`)")
    download_parser.add_argument(
        "-f",
        dest="format",
        default="voc",
        help="Specify the format to download the version. Available options: [coco, "
        "yolov5pytorch, yolov7pytorch, my-yolov6, darknet, voc, tfrecord, "
        "createml, clip, multiclass, coco-segmentation, yolo5-obb, "
        "png-mask-semantic, yolov8, yolov9]",
    )
    download_parser.add_argument("-l", dest="location", help="Location to download the dataset")
    download_parser.set_defaults(func=download)


def _add_upload_parser(subparsers):
    upload_parser = subparsers.add_parser("upload", help="Upload a single image to a dataset")
    upload_parser.add_argument(
        "imagefile",
        help="path to image file",
    )
    upload_parser.add_argument(
        "-w",
        dest="workspace",
        help="specify a workspace url or id " "(will use default workspace if not specified)",
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
    import_parser = subparsers.add_parser("import", help="Import a dataset from a local folder")
    import_parser.add_argument(
        "folder",
        help="filesystem path to a folder that contains your dataset",
    )
    import_parser.add_argument(
        "-w",
        dest="workspace",
        help="specify a workspace url or id " "(will use default workspace if not specified)",
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
        "-n",
        dest="batch_name",
        help="name of batch to upload to within project",
    )
    import_parser.add_argument(
        "-r", dest="num_retries", type=int, help="Retry failed uploads this many times (default=0)", default=0
    )
    import_parser.set_defaults(func=import_dataset)


def _add_projects_parser(subparsers):
    project_parser = subparsers.add_parser(
        "project",
        help="project related commands.  type 'roboflow project' to see detailed command help",
    )
    projectsubparsers = project_parser.add_subparsers(title="project subcommands")
    projectlist_parser = projectsubparsers.add_parser("list", help="list projects")
    projectlist_parser.add_argument(
        "-w",
        dest="workspace",
        help="specify a workspace url or id (will use default workspace if not specified)",
    )
    projectlist_parser.set_defaults(func=list_projects)
    projectget_parser = projectsubparsers.add_parser("get", help="show detailed info for a project")
    projectget_parser.add_argument(
        "projectId",
        help="project ID",
    )
    projectget_parser.add_argument(
        "-w",
        dest="workspace",
        help="specify a workspace url or id (will use default workspace if not specified)",
    )
    projectget_parser.set_defaults(func=get_project)


def _add_workspaces_parser(subparsers):
    workspace_parser = subparsers.add_parser(
        "workspace",
        help="workspace related commands.  type 'roboflow workspace' to see detailed command help",
    )
    workspacesubparsers = workspace_parser.add_subparsers(title="workspace subcommands")
    workspacelist_parser = workspacesubparsers.add_parser("list", help="list workspaces")
    workspacelist_parser.set_defaults(func=list_workspaces)
    workspaceget_parser = workspacesubparsers.add_parser("get", help="show detailed info for a workspace")
    workspaceget_parser.add_argument(
        "workspaceId",
        help="project ID",
    )
    workspaceget_parser.set_defaults(func=get_workspace)


def _add_run_video_inference_api_parser(subparsers):
    run_video_inference_api_parser = subparsers.add_parser(
        "run_video_inference_api",
        help="run video inference api",
    )

    run_video_inference_api_parser.add_argument(
        "-a",
        dest="api_key",
        help="api_key",
    )
    run_video_inference_api_parser.add_argument(
        "-p",
        dest="project",
        help="project_id to upload the image into",
    )
    run_video_inference_api_parser.add_argument(
        "-v",
        dest="version_number",
        type=int,
        help="version number to upload the model to",
    )
    run_video_inference_api_parser.add_argument(
        "-f",
        dest="video_file",
        help="path to video file",
    )
    run_video_inference_api_parser.add_argument(
        "-fps",
        dest="fps",
        type=int,
        help="fps",
        default=5,
    )
    run_video_inference_api_parser.set_defaults(func=run_video_inference_api)


def _add_infer_parser(subparsers):
    infer_parser = subparsers.add_parser(
        "infer",
        help="perform inference on an image",
    )
    infer_parser.add_argument(
        "file",
        help="filesystem path to an image file",
    )
    infer_parser.add_argument(
        "-w",
        dest="workspace",
        help="specify a workspace url or id (will use default workspace if not specified)",
    )
    infer_parser.add_argument(
        "-m",
        dest="model",
        help="model id (id of a version with trained model e.g. my-project/3)",
    )
    infer_parser.add_argument(
        "-c",
        dest="confidence",
        type=float,
        help="specify a confidence threshold between 0.0 and 1.0, default is 0.5"
        "(only applies to object-detection models)",
        default=0.5,
    )
    infer_parser.add_argument(
        "-o",
        dest="overlap",
        type=float,
        help="specify an overlap threshold between 0.0 and 1.0, default is 0.5"
        "(only applies to object-detection models)",
        default=0.5,
    )
    infer_parser.add_argument(
        "-t",
        dest="type",
        help="specify the model type to skip api call to look it up",
        choices=[
            "object-detection",
            "classification",
            "instance-segmentation",
            "semantic-segmentation",
        ],
    )
    infer_parser.set_defaults(func=infer)


def _add_upload_model_parser(subparsers):
    upload_model_parser = subparsers.add_parser(
        "upload_model",
        help="Upload a trained model to Roboflow",
    )
    upload_model_parser.add_argument(
        "-a",
        dest="api_key",
        help="api_key",
    )
    upload_model_parser.add_argument(
        "-w",
        dest="workspace",
        help="specify a workspace url or id (will use default workspace if not specified)",
    )
    upload_model_parser.add_argument(
        "-p",
        dest="project",
        help="project_id to upload the model into",
    )
    upload_model_parser.add_argument(
        "-v",
        dest="version_number",
        type=int,
        help="version number to upload the model to",
    )
    upload_model_parser.add_argument(
        "-t",
        dest="model_type",
        help="type of the model (e.g., yolov8, yolov5)",
    )
    upload_model_parser.add_argument(
        "-m",
        dest="model_path",
        help="path to the trained model file",
    )
    upload_model_parser.add_argument(
        "-f",
        dest="filename",
        default="weights/best.pt",
        help="name of the model file",
    )
    upload_model_parser.set_defaults(func=upload_model)


def _add_get_workspace_project_version_parser(subparsers):
    workspace_project_version_parser = subparsers.add_parser(
        "get_workspace_info",
        help="get workspace project version info",
    )
    workspace_project_version_parser.add_argument(
        "-a",
        dest="api_key",
        help="api_key",
    )
    workspace_project_version_parser.add_argument(
        "-w",
        dest="workspace",
        help="specify a workspace url or id (will use default workspace if not specified)",
    )
    workspace_project_version_parser.add_argument(
        "-p",
        dest="project",
        help="project_id to upload the model into",
    )
    workspace_project_version_parser.add_argument(
        "-v",
        dest="version_number",
        type=int,
        help="version number to upload the model to",
    )
    workspace_project_version_parser.set_defaults(func=get_workspace_project_version)


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
