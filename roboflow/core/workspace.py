from __future__ import annotations

import concurrent.futures
import glob
import json
import os
import sys
import tempfile
import time
import zipfile
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional

import requests
from requests.exceptions import HTTPError
from tqdm import tqdm

from roboflow.adapters import rfapi, vision_events_api
from roboflow.adapters.rfapi import AnnotationSaveError, ImageUploadError, RoboflowError
from roboflow.config import API_URL, APP_URL, DEMO_KEYS

if TYPE_CHECKING:
    from roboflow.core.device import Device


class Workspace:
    """
    Manage a Roboflow workspace.
    """

    def __init__(self, info, api_key, default_workspace, model_format):
        if api_key:
            self.__api_key = api_key

            workspace_info = info["workspace"]
            self.name = workspace_info["name"]
            self.project_list = workspace_info["projects"]
            if "members" in workspace_info.keys():
                self.members = workspace_info["members"]
            self.url = workspace_info["url"]
            self.model_format = model_format

        elif DEMO_KEYS:
            self.__api_key = DEMO_KEYS[0]
            self.model_format = model_format
            self.project_list = []

        else:
            raise ValueError("A valid API key must be provided.")

    def list_projects(self):
        """
        Print all projects in the workspace to the console.
        """
        print(self.project_list)

    def projects(self):
        """
        Retrieve all projects in the workspace.

        Returns:
            List of Project objects.
        """
        from roboflow.core.project import Project

        projects_array = []
        for a_project in self.project_list:
            proj = Project(self.__api_key, a_project, self.model_format)
            projects_array.append(proj.id)

        return projects_array

    def project(self, project_id):
        """
        Retrieve a Project() object that represents a project in the workspace.

        This object can be used to retrieve the model through which to run inference.

        Args:
            project_id (str): id of the project

        Returns:
            Project Object
        """
        from roboflow.core.project import Project

        sys.stdout.write("\r" + "loading Roboflow project...")
        sys.stdout.write("\n")
        sys.stdout.flush()

        if self.__api_key in DEMO_KEYS:
            return Project(self.__api_key, {}, self.model_format)

        # project_id = project_id.replace(self.url + "/", "")

        if "/" in project_id:
            raise RuntimeError(f"The {project_id} project is not available in this ({self.url}) workspace")

        dataset_info = rfapi.get_project(self.__api_key, self.url, project_id)
        dataset_info = dataset_info["project"]

        return Project(self.__api_key, dataset_info, self.model_format)

    def create_project(self, project_name, project_type, project_license, annotation):
        """
        Create a project in a Roboflow workspace.

        Args:
            project_name (str): name of the project
            project_type (str): type of the project
            project_license (str): license of the project (set to `Private` for private projects, only available for paid customers)
            annotation (str): annotation of the project

        Returns:
            Project Object
        """  # noqa: E501 // docs
        from roboflow.core.project import Project

        data = {
            "name": project_name,
            "type": project_type,
            "license": project_license,
            "annotation": annotation,
        }

        r = requests.post(API_URL + "/" + self.url + "/projects?api_key=" + self.__api_key, json=data)

        r.raise_for_status()

        if "error" in r.json().keys():
            raise RuntimeError(r.json()["error"])

        return Project(self.__api_key, r.json(), self.model_format)

    def devices(self) -> List["Device"]:
        """List v2 devices registered in this workspace.

        Returns:
            List of :class:`roboflow.core.device.Device` objects. Each
            wraps the entity returned by ``GET /:workspace/devices/v2``
            (id, name, status, last_heartbeat, hardware, tags, …).
        """
        from roboflow.adapters import devicesapi
        from roboflow.core.device import Device

        rows = devicesapi.list_devices(self.__api_key, self.url).get("data", [])
        return [Device(self.__api_key, self.url, row) for row in rows]

    def device(self, device_id: str) -> "Device":
        """Get a single device by id.

        Args:
            device_id: The device id (as returned by :meth:`devices` or by
                :meth:`create_device`).

        Returns:
            A :class:`roboflow.core.device.Device` instance.
        """
        from roboflow.adapters import devicesapi
        from roboflow.core.device import Device

        info = devicesapi.get_device(self.__api_key, self.url, device_id)
        return Device(self.__api_key, self.url, info)

    def create_device(
        self,
        device_name: str,
        device_type: Optional[str] = None,
        *,
        workflow_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        offline_mode: Optional[bool] = None,
        source_device_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new v2 device in the workspace.

        Args:
            device_name: Human-readable device name (required).
            device_type: ``"ai1"``, ``"edge"``, or any custom string.
            workflow_id: Optional initial workflow assignment. For AI1 devices
                this seeds the default ``aione`` stream.
            tags: Optional list of string tags.
            offline_mode: Boolean; only valid for AI1 devices on workspaces
                with the ``roboflowLiteMode`` feature.
            source_device_id: When set, duplicates the named existing
                device's config instead of generating a fresh one.

        Returns:
            Dict with ``deviceId`` and ``installId`` (the short-lived install
            token to feed into ``GET /devices/v2/:installId/install.sh``).
        """
        from roboflow.adapters import devicesapi

        return devicesapi.create_device(
            self.__api_key,
            self.url,
            device_name=device_name,
            device_type=device_type,
            workflow_id=workflow_id,
            tags=tags,
            offline_mode=offline_mode,
            source_device_id=source_device_id,
        )

    def clip_compare(self, dir: str = "", image_ext: str = ".png", target_image: str = "") -> List[dict]:
        """
        Compare all images in a directory to a target image using CLIP

        Args:
            dir (str): name reference to a directory of images for comparison
            image_ext (str): file format for expected images (don't include the . before the file type name)
            target_image (str): name reference for target image to compare individual images from directory against

        Returns:
            # TODO: fix docs
            dict: a key:value mapping of image_name:comparison_score_to_target
        """  # noqa: E501 // docs

        from roboflow.config import CLIP_FEATURIZE_URL
        from roboflow.util.active_learning_utils import clip_encode

        # list to store comparison results in
        comparisons = []
        # grab all images in a given directory with ext type
        for image in glob.glob(f"./{dir}/*{image_ext}"):
            # compare image
            similarity = clip_encode(image, target_image, CLIP_FEATURIZE_URL)
            # map image name to similarity score
            comparisons.append({image: similarity})
            comparisons = sorted(comparisons, key=lambda item: -list(item.values())[0])
        return comparisons

    def two_stage(
        self,
        image: str = "",
        first_stage_model_name: str = "",
        first_stage_model_version: int = 0,
        second_stage_model_name: str = "",
        second_stage_model_version: int = 0,
    ) -> List[dict]:
        """
        For each prediction in a first stage detection, perform detection with the second stage model

        Args:
            image (str): name of the image to be processed
            first_stage_model_name (str): name of the first stage detection model
            first_stage_model_version (int): version number for the first stage model
            second_stage_mode (str): name of the second stage detection model
            second_stage_model_version (int): version number for the second stage model

        Returns:
            # TODO: fix docs
            dict: a json obj containing the results of the second stage detection
        """  # noqa: E501 // docs
        from PIL import Image

        results = []

        # create PIL image for cropping
        pil_image = Image.open(image).convert("RGB")

        # grab first and second stage model from project
        stage_one_project = self.project(first_stage_model_name)
        stage_one_model = stage_one_project.version(first_stage_model_version).model
        stage_two_project = self.project(second_stage_model_name)
        stage_two_model = stage_two_project.version(second_stage_model_version).model

        print(self.project(first_stage_model_name))

        # perform first inference
        predictions = stage_one_model.predict(image)  # type: ignore[attribute-error]

        if stage_one_project.type == "object-detection" and stage_two_project == "classification":
            # interact with each detected object from stage one inference results
            for boundingbox in predictions:
                # rip bounding box coordinates from json1
                # note: infer returns center points of box as (x,y) and width, height
                # ----- but pillow crop requires the top left and bottom right points to crop  # noqa: E501 // docs
                box = (
                    boundingbox["x"] - boundingbox["width"] / 2,
                    boundingbox["y"] - boundingbox["height"] / 2,
                    boundingbox["x"] + boundingbox["width"] / 2,
                    boundingbox["y"] + boundingbox["height"] / 2,
                )

                # create a new cropped image using the first stage prediction coordinates (for each box!)  # noqa: E501 // docs
                croppedImg = pil_image.crop(box)
                croppedImg.save("./temp.png")

                # capture results of second stage inference from cropped image
                results.append(stage_two_model.predict("./temp.png")[0])  # type: ignore[attribute-error]

            # delete the written image artifact
            try:
                os.remove("./temp.png")
            except FileNotFoundError:
                print("no detections")

        else:
            print(
                "please use an object detection model for the first stage--can only"
                " perform two stage with bounding box results",
                "please use a classification model for the second stage",
            )

        return results

    def two_stage_ocr(
        self,
        image: str = "",
        first_stage_model_name: str = "",
        first_stage_model_version: int = 0,
    ) -> List[dict]:
        """
        For each prediction in the first stage object detection, perform OCR as second stage.

        Args:
            image (str): name of the image to be processed
            first_stage_model_name (str): name of the first stage detection model
            first_stage_model_version (int): version number for the first stage model

        Returns:
            # TODO: fix docs
            dict: a json obj containing the results of the second stage detection
        """  # noqa: E501 // docs
        from PIL import Image

        from roboflow.util.two_stage_utils import ocr_infer

        results = []

        # create PIL image for cropping
        pil_image = Image.open(image).convert("RGB")

        # grab first and second stage model from project
        stage_one_project = self.project(first_stage_model_name)
        stage_one_model = stage_one_project.version(first_stage_model_version).model

        # perform first inference
        predictions = stage_one_model.predict(image)  # type: ignore[attribute-error]

        # interact with each detected object from stage one inference results
        if stage_one_project.type == "object-detection":
            for boundingbox in predictions:
                # rip bounding box coordinates from json1
                # note: infer returns center points of box as (x,y) and width, height
                # but pillow crop requires the top left and bottom right points to crop
                box = (
                    boundingbox["x"] - boundingbox["width"] / 2,
                    boundingbox["y"] - boundingbox["height"] / 2,
                    boundingbox["x"] + boundingbox["width"] / 2,
                    boundingbox["y"] + boundingbox["height"] / 2,
                )

                # create a new cropped image using the first stage
                # prediction coordinates (for each box!)
                croppedImg = pil_image.crop(box)

                # capture OCR results from cropped image
                results.append(ocr_infer(croppedImg)["results"])
        else:
            print("please use an object detection model--can only perform two stage with bounding box results")

        return results

    def upload_dataset(
        self,
        dataset_path: str,
        project_name: str,
        num_workers: int = 10,
        dataset_format: str = "NOT_USED",  # deprecated. keep for backward compatibility
        project_license: str = "MIT",
        project_type: str = "object-detection",
        batch_name=None,
        num_retries=0,
        is_prediction=False,
        *,
        use_zip_upload: bool = False,
        tags: Optional[List[str]] = None,
        split: Optional[str] = None,
        wait: bool = True,
        poll_interval: float = 5.0,
        poll_timeout: float = 3600.0,
    ) -> Optional[dict]:
        """
        Upload a dataset to Roboflow.

        A `.zip` ``dataset_path`` or ``use_zip_upload=True`` routes to the
        server's async zip upload flow. Everything else (directory inputs by
        default) keeps the legacy per-image flow.

        Args:
            dataset_path (str): path to the dataset directory or a `.zip` file.
            project_name (str): name of the project
            num_workers (int): number of workers to use for parallel uploads (per-image flow only)
            dataset_format (str): format of the dataset (`voc`, `yolov8`, `yolov5`)
            project_license (str): license of the project (set to `private` for private projects, only available for paid customers)
            project_type (str): type of the project (only `object-detection` is supported)
            batch_name (str, optional): name of the batch to upload the images to. Defaults to an automatically generated value.
            num_retries (int, optional): number of times to retry uploading an image if the upload fails. Defaults to 0.
            is_prediction (bool, optional): whether the annotations provided in the dataset are predictions and not ground truth. Defaults to False.
            use_zip_upload (bool, optional): opt-in to the zip flow for a directory input (the SDK zips it client-side). Ignored when dataset_path is already a `.zip`.
            tags (list[str], optional): zip flow only — tags to apply to the uploaded batch.
            split (str, optional): zip flow only — dataset split for the uploaded batch.
            wait (bool, optional): zip flow only — poll for processing completion. Defaults to True.
            poll_interval (float, optional): zip flow only — seconds between status polls.
            poll_timeout (float, optional): zip flow only — total seconds to wait before timing out.

        Returns:
            dict | None: zip flow returns the final/pending status dict; per-image flow returns None.
        """  # noqa: E501 // docs
        if dataset_format != "NOT_USED":
            print("Warning: parameter 'dataset_format' is deprecated and will be removed in a future release")
        project, created = self._get_or_create_project(
            project_id=project_name, license=project_license, type=project_type
        )
        if created:
            print(f"Created project {project.id}")
        else:
            print(f"Uploading to existing project {project.id}")

        is_zip_file = dataset_path.lower().endswith(".zip") and os.path.isfile(dataset_path)
        use_zip_flow = is_zip_file or use_zip_upload
        if use_zip_flow and is_prediction:
            raise RoboflowError(
                "Zip upload flow does not support is_prediction=True. "
                "Call upload_dataset without use_zip_upload for prediction uploads."
            )

        if use_zip_flow:
            project_slug = project.id.rsplit("/")[1]
            temp_zip = None
            try:
                if dataset_path.lower().endswith(".zip") and os.path.isfile(dataset_path):
                    zip_path = dataset_path
                else:
                    zip_path = temp_zip = _zip_directory(dataset_path)
                    print(f"Zipped {dataset_path} -> {zip_path}")

                init = rfapi.init_zip_upload(
                    self.__api_key,
                    self.url,
                    project_slug,
                    split=split,
                    tags=tags,
                    batch_name=batch_name,
                )
                print(f"Uploading zip to Roboflow (task_id={init['taskId']})...")
                rfapi.upload_zip_to_signed_url(init["signedUrl"], zip_path)

                if not wait:
                    print(f"Zip uploaded; not waiting for processing. task_id={init['taskId']}")
                    return {"task_id": init["taskId"], "status": "pending"}

                return _poll_zip_status(self.__api_key, self.url, init["taskId"], poll_interval, poll_timeout)
            finally:
                if temp_zip and os.path.exists(temp_zip):
                    os.unlink(temp_zip)

        from roboflow.util import folderparser
        from roboflow.util.image_utils import load_labelmap

        is_classification = project.type == "classification"
        parsed_dataset = folderparser.parsefolder(dataset_path, is_classification=is_classification)
        images = parsed_dataset["images"]

        location = parsed_dataset["location"]

        def _log_img_upload(
            image_path, image, annotation, image_upload_time, image_upload_retry_attempts, annotation_time
        ):
            image_id = image.get("id")
            img_success = image.get("success")
            img_duplicate = image.get("duplicate")

            upload_time_str = f"[{image_upload_time:.1f}s]"
            annotation_time_str = f"[{annotation_time:.1f}s]" if annotation_time else ""
            retry_attempts = f" (with {image_upload_retry_attempts} retries)" if image_upload_retry_attempts > 0 else ""

            if img_duplicate:
                msg = f"[DUPLICATE]{retry_attempts} {image_path} ({image_id}) {upload_time_str}"
            elif img_success:
                msg = f"[UPLOADED]{retry_attempts} {image_path} ({image_id}) {upload_time_str}"
            else:
                msg = f"[LOG ERROR]: Unrecognized image upload status ({image_id=})"
            if annotation:
                if annotation.get("success"):
                    msg += f" / annotations = OK {annotation_time_str}"
                elif annotation.get("warn"):
                    msg += f" / annotations = WARN: {annotation['warn']} {annotation_time_str}"
                else:
                    msg += " / annotations = ERR: Unrecognized annotation upload status"

            print(msg)

        def _upload_image(imagedesc):
            image_path = f"{location}{imagedesc['file']}"
            split = imagedesc["split"]

            image, upload_time, upload_retry_attempts = project.upload_image(
                image_path=image_path,
                split=split,
                batch_name=batch_name,
                sequence_number=imagedesc.get("index"),
                sequence_size=len(images),
                num_retry_uploads=num_retries,
            )

            return image, upload_time, upload_retry_attempts

        def _save_annotation(image_id, imagedesc):
            labelmap = None
            annotation_path = None

            annotationdesc = imagedesc.get("annotationfile")
            if isinstance(annotationdesc, dict):
                if annotationdesc.get("type") == "classification_folder":
                    annotation_path = annotationdesc.get("classification_label")
                elif annotationdesc.get("type") == "classification_multilabel":
                    annotation_path = json.dumps(annotationdesc.get("labels", []))
                elif annotationdesc.get("rawText"):
                    annotation_path = annotationdesc
                elif annotationdesc.get("file"):
                    annotation_path = f"{location}{annotationdesc['file']}"
                    labelmap = annotationdesc.get("labelmap")

                if isinstance(labelmap, str):
                    labelmap = load_labelmap(labelmap)

            # If annotation_path is still None at this point, then no annotation will be saved.
            if annotation_path is None:
                return None, None

            annotation, upload_time, _retry_attempts = project.save_annotation(
                annotation_path=annotation_path,
                annotation_labelmap=labelmap,
                image_id=image_id,
                job_name=batch_name,
                num_retry_uploads=num_retries,
                is_prediction=is_prediction,
            )

            return annotation, upload_time

        def _upload(imagedesc):
            image_path = f"{location}{imagedesc['file']}"

            image_id = None
            image_upload_time = None
            image_retry_attempts = None

            try:
                image, image_upload_time, image_retry_attempts = _upload_image(imagedesc)
                image_id = image["id"]
                annotation, annotation_time = _save_annotation(image_id, imagedesc)
                _log_img_upload(image_path, image, annotation, image_upload_time, image_retry_attempts, annotation_time)
            except ImageUploadError as e:
                retry_attempts = f" (with {e.retries} retries)" if e.retries > 0 else ""
                print(f"[ERR]{retry_attempts} {image_path} ({e.message})")
            except AnnotationSaveError as e:
                upload_time_str = f"[{image_upload_time:.1f}s]"
                retry_attempts = f" (with {image_retry_attempts} retries)" if image_retry_attempts > 0 else ""
                image_msg = f"[UPLOADED]{retry_attempts} {image_path} ({image_id}) {upload_time_str}"
                annotation_msg = f"annotations = ERR: {e.message}"
                print(f"{image_msg} / {annotation_msg}")
            except Exception as e:
                print(f"[ERR] {image_path} ({e})")

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            list(executor.map(_upload, images))

        return None

    def _get_or_create_project(self, project_id, license: str = "MIT", type: str = "object-detection"):
        try:
            existing_project = self.project(project_id)
            return existing_project, False
        except RoboflowError:
            return (
                self.create_project(
                    project_name=project_id,
                    project_license=license,
                    annotation=project_id,
                    project_type=type,
                ),
                True,
            )

    def active_learning(
        self,
        raw_data_location: str = "",
        raw_data_extension: str = "",
        inference_endpoint: Optional[List[str]] = None,
        upload_destination: str = "",
        conditionals: Optional[Dict] = None,
        use_localhost: bool = False,
        local_server="http://localhost:9001/",
    ) -> Any:
        """perform inference on each image in directory and upload based on conditions
        @params:
            raw_data_location: (str) = folder of frames to be processed
            raw_data_extension: (str) = extension of frames to be processed
            inference_endpoint: (List[str, int]) = name of the project
            upload_destination: (str) = name of the upload project
            conditionals: (dict) = dictionary of upload conditions
            use_localhost: (bool) = determines if local http format used or remote endpoint
            local_server: (str) = local http address for inference server, use_localhost must be True for this to be used
        """  # noqa: E501 // docs
        from roboflow.config import CLIP_FEATURIZE_URL
        from roboflow.util.active_learning_utils import check_box_size, clip_encode, count_comparisons

        if inference_endpoint is None:
            inference_endpoint = []
        if conditionals is None:
            conditionals = {}

        import numpy as np

        prediction_results = []

        # ensure that all fields of conditionals have a key:value pair
        conditionals["target_classes"] = [] if "target_classes" not in conditionals else conditionals["target_classes"]
        conditionals["confidence_interval"] = (
            [30, 99] if "confidence_interval" not in conditionals else conditionals["confidence_interval"]
        )
        conditionals["required_class_variance_count"] = (
            1 if "required_class_variance_count" not in conditionals else conditionals["required_class_variance_count"]
        )
        conditionals["required_objects_count"] = (
            1 if "required_objects_count" not in conditionals else conditionals["required_objects_count"]
        )
        conditionals["required_class_count"] = (
            0 if "required_class_count" not in conditionals else conditionals["required_class_count"]
        )
        conditionals["minimum_size_requirement"] = (
            float("-inf")
            if "minimum_size_requirement" not in conditionals
            else conditionals["minimum_size_requirement"]
        )
        conditionals["maximum_size_requirement"] = (
            float("inf") if "maximum_size_requirement" not in conditionals else conditionals["maximum_size_requirement"]
        )

        # check if inference_model references endpoint or local
        if use_localhost:
            local = local_server
        else:
            local = None

        inference_model = (
            self.project(inference_endpoint[0]).version(version_number=inference_endpoint[1], local=local).model
        )
        upload_project = self.project(upload_destination)

        print("inference reference point: ", inference_model)
        print("upload destination: ", upload_project)

        # check if raw data type is cv2 frame
        if issubclass(type(raw_data_location), np.ndarray):
            globbed_files = [raw_data_location]
        else:
            globbed_files = glob.glob(raw_data_location + "/*" + raw_data_extension)

        image1 = globbed_files[0]
        similarity_timeout_counter = 0

        for index, image in enumerate(globbed_files):
            try:
                print(
                    "*** Processing image [" + str(index + 1) + "/" + str(len(globbed_files)) + "] - " + image + " ***"
                )
            except Exception:
                pass

            if "similarity_confidence_threshold" in conditionals.keys():
                image2 = image
                # measure the similarity of two images using CLIP (hits an endpoint hosted by Roboflow)   # noqa: E501 // docs
                similarity = clip_encode(image1, image2, CLIP_FEATURIZE_URL)
                similarity_timeout_counter += 1

                if (
                    similarity <= conditionals["similarity_confidence_threshold"]
                    or similarity_timeout_counter == conditionals["similarity_timeout_limit"]
                ):
                    image1 = image
                    similarity_timeout_counter = 0
                else:
                    print(image2 + " --> similarity too high to --> " + image1)
                    continue  # skip this image if too similar or counter hits limit

            predictions = inference_model.predict(image).json()["predictions"]  # type: ignore[attribute-error]
            # collect all predictions to return to user at end
            prediction_results.append({"image": image, "predictions": predictions})

            # compare object and class count of predictions if enabled,
            # continue if not enough occurrences
            if not count_comparisons(
                predictions,
                conditionals["required_objects_count"],
                conditionals["required_class_count"],
                conditionals["target_classes"],
            ):
                print(" [X] image failed count cases")
                continue

            # iterate through all predictions
            for i, prediction in enumerate(predictions):
                print(i)

                # check if box size of detection fits requirements
                if not check_box_size(
                    prediction,
                    conditionals["minimum_size_requirement"],
                    conditionals["maximum_size_requirement"],
                ):
                    print(" [X] prediction failed box size cases")
                    continue

                # compare confidence of detected object to confidence thresholds
                # confidence comes in as a .XXX instead of XXX%
                if (
                    prediction["confidence"] * 100 >= conditionals["confidence_interval"][0]
                    and prediction["confidence"] * 100 <= conditionals["confidence_interval"][1]
                ):
                    # filter out non-target_class uploads if enabled
                    if (
                        len(conditionals["target_classes"]) > 0
                        and prediction["class"] not in conditionals["target_classes"]
                    ):
                        print(" [X] prediction failed target_classes")
                        continue

                    # upload on success!
                    print(" >> image uploaded!")
                    upload_project.upload(image, num_retry_uploads=3)
                    break

        # return predictions with filenames if globbed images from dir,
        # otherwise return latest prediction result
        return (
            prediction_results if type(raw_data_location) is not np.ndarray else prediction_results[-1]["predictions"]
        )

    def deploy_model(
        self,
        model_type: str,
        model_path: str,
        project_ids: list[str],
        model_name: str,
        filename: str = "weights/best.pt",
    ):
        """Uploads provided weights file to Roboflow.
        Args:
            model_type (str): The type of the model to be deployed.
            model_path (str): File path to the model weights to be uploaded.
            project_ids (list[str]): List of project IDs to deploy the model to.
            filename (str, optional): The name of the weights file. Defaults to "weights/best.pt".
        """

        from roboflow.util.model_processor import process, validate_model_type_for_project
        from roboflow.util.versions import normalize_yolo_model_type

        if not project_ids:
            raise ValueError("At least one project ID must be provided")

        # Validate if provided project URLs belong to user's projects, and look up
        # each one's type (already cached on self.project_list — no extra API call).
        projects_by_id = {p["id"].split("/")[-1]: p for p in self.project_list if "id" in p}
        for project_id in project_ids:
            if project_id not in projects_by_id:
                raise ValueError(f"Project {project_id} is not accessible in this workspace")

        model_type = normalize_yolo_model_type(model_type)
        zip_file_name, model_type = process(model_type, model_path, filename)

        if zip_file_name is None:
            raise RuntimeError("Failed to process model")

        for project_id in project_ids:
            validate_model_type_for_project(model_type, projects_by_id[project_id].get("type", ""), project_id)

        self._upload_zip(model_type, model_path, project_ids, model_name, zip_file_name)

    def _upload_zip(
        self,
        model_type: str,
        model_path: str,
        project_ids: list[str],
        model_name: str,
        model_file_name: str,
    ):
        # This endpoint returns a signed URL to upload the model
        res = requests.post(
            f"{API_URL}/{self.url}/models/prepareUpload?api_key={self.__api_key}&modelType={model_type}&modelName={model_name}&projectIds={','.join(project_ids)}&nocache=true"
        )
        try:
            res.raise_for_status()
        except Exception as e:
            error_message = str(e)
            status_code = str(res.status_code)

            print("\n\033[91m❌ ERROR\033[0m: Failed to get model deployment URL")
            print("\033[93mDetails\033[0m:", error_message)
            print("\033[93mStatus\033[0m:", status_code)
            print(f"\033[93mResponse\033[0m:\n{res.text}\n")
            return

        # Upload the model to the signed URL
        res = requests.put(
            res.json()["url"],
            data=open(os.path.join(model_path, model_file_name), "rb"),
        )
        try:
            res.raise_for_status()

            for project_id in project_ids:
                print(
                    f"View the status of your deployment for project {project_id} at:"
                    f" {APP_URL}/{self.url}/{project_id}/models"
                )

        except Exception as e:
            print(f"An error occured when uploading the model: {e}")

    def search(
        self,
        query: str,
        page_size: int = 50,
        fields: Optional[List[str]] = None,
        continuation_token: Optional[str] = None,
    ) -> dict:
        """Search across all images in the workspace using RoboQL syntax.

        Args:
            query: RoboQL search query (e.g. ``"tag:review"``, ``"project:false"``
                for orphan images, or free-text for semantic CLIP search).
            page_size: Number of results per page (default 50).
            fields: Fields to include in each result.
                Defaults to ``["tags", "projects", "filename"]``.
            continuation_token: Token returned by a previous call for fetching
                the next page.

        Returns:
            Dict with ``results`` (list), ``total`` (int), and
            ``continuationToken`` (str or None).

        Example:
            >>> ws = rf.workspace()
            >>> page = ws.search("tag:review", page_size=10)
            >>> print(page["total"])
            >>> for img in page["results"]:
            ...     print(img["filename"])
        """
        if fields is None:
            fields = ["tags", "projects", "filename"]

        return rfapi.workspace_search(
            api_key=self.__api_key,
            workspace_url=self.url,
            query=query,
            page_size=page_size,
            fields=fields,
            continuation_token=continuation_token,
        )

    def delete_images(self, image_ids: List[str]) -> dict:
        """Delete orphan images from the workspace.

        Only deletes images not associated with any project.
        Images still in projects are skipped.

        Args:
            image_ids: List of image IDs to delete.

        Returns:
            Dict with ``deletedSources`` and ``skippedSources`` counts.

        Example:
            >>> ws = rf.workspace()
            >>> result = ws.delete_images(["img_id_1", "img_id_2"])
            >>> print(result["deletedSources"])
        """
        return rfapi.workspace_delete_images(
            api_key=self.__api_key,
            workspace_url=self.url,
            image_ids=image_ids,
        )

    def search_all(
        self,
        query: str,
        page_size: int = 50,
        fields: Optional[List[str]] = None,
    ) -> Generator[List[dict], None, None]:
        """Paginated search across all images in the workspace.

        Yields one page of results at a time, automatically following
        ``continuationToken`` until all results have been returned.

        Args:
            query: RoboQL search query.
            page_size: Number of results per page (default 50).
            fields: Fields to include in each result.
                Defaults to ``["tags", "projects", "filename"]``.

        Yields:
            A list of result dicts for each page.

        Example:
            >>> ws = rf.workspace()
            >>> for page in ws.search_all("tag:review"):
            ...     for img in page:
            ...         print(img["filename"])
        """
        token = None
        while True:
            response = self.search(
                query=query,
                page_size=page_size,
                fields=fields,
                continuation_token=token,
            )
            results = response.get("results", [])
            if not results:
                break
            yield results
            token = response.get("continuationToken")
            if not token:
                break

    def search_export(
        self,
        query: str,
        format: str = "coco",
        location: Optional[str] = None,
        dataset: Optional[str] = None,
        annotation_group: Optional[str] = None,
        name: Optional[str] = None,
        extract_zip: bool = True,
    ) -> str:
        """Export search results as a downloaded dataset.

        Args:
            query: Search query string (e.g. ``"tag:annotate"`` or ``"class:apple"``).
            format: Annotation format for the export (default ``"coco"``).
            location: Local directory to save the exported dataset.
                Defaults to ``./search-export-{format}``.
            dataset: Limit export to a specific dataset (project) slug.
            annotation_group: Limit export to a specific annotation group.
            name: Optional name for the export.
            extract_zip: If True (default), extract the zip and remove it.
                If False, keep the zip file as-is.

        Returns:
            Absolute path to the extracted directory or the zip file.

        Raises:
            ValueError: If both *dataset* and *annotation_group* are provided.
            RoboflowError: On API errors or export timeout.
        """
        from roboflow.util.general import extract_zip as _extract_zip

        if dataset is not None and annotation_group is not None:
            raise ValueError("dataset and annotation_group are mutually exclusive; provide only one")

        if location is None:
            location = f"./search-export-{format}"
        location = os.path.abspath(location)

        # 1. Start the export
        session = requests.Session()
        export_id = rfapi.start_search_export(
            api_key=self.__api_key,
            workspace_url=self.url,
            query=query,
            format=format,
            dataset=dataset,
            annotation_group=annotation_group,
            name=name,
            session=session,
        )
        print(f"Export started (id={export_id}). Polling for completion...")

        status_url = f"{API_URL}/{self.url}/search/export/{export_id}?api_key=YOUR_API_KEY"
        print(f"If this takes too long, you can check the export status at: {status_url}")

        # 2. Poll until ready
        timeout = 1800
        poll_interval = 5
        elapsed = 0

        while elapsed < timeout:
            status = rfapi.get_search_export(
                api_key=self.__api_key,
                workspace_url=self.url,
                export_id=export_id,
                session=session,
            )
            if status.get("ready"):
                break
            time.sleep(poll_interval)
            elapsed += poll_interval
        else:
            raise RoboflowError(f"Search export timed out after {timeout}s")

        download_url = status["link"]

        # 3. Download zip
        if not os.path.exists(location):
            os.makedirs(location)

        zip_path = os.path.join(location, "roboflow.zip")
        response = session.get(download_url, stream=True)
        try:
            response.raise_for_status()
        except HTTPError as e:
            raise RoboflowError(f"Failed to download search export: {e}")

        total_length = response.headers.get("content-length")
        try:
            total_kib = int(total_length) // 1024 + 1 if total_length is not None else None
        except (TypeError, ValueError):
            total_kib = None
        with open(zip_path, "wb") as f:
            for chunk in tqdm(
                response.iter_content(chunk_size=1024),
                desc=f"Downloading search export to {location}",
                total=total_kib,
            ):
                if chunk:
                    f.write(chunk)
                    f.flush()

        if extract_zip:
            _extract_zip(location, desc=f"Extracting search export to {location}")
            print(f"Search export extracted to {location}")
            return location
        else:
            print(f"Search export saved to {zip_path}")
            return zip_path

    # -----------------------------------------------------------------
    # Phase 2: Folder management
    # -----------------------------------------------------------------

    def list_folders(self):
        """List project folders in this workspace."""
        from roboflow.adapters import rfapi

        return rfapi.list_folders(self.__api_key, self.url)

    def create_folder(self, name, parent_id=None, project_ids=None):
        """Create a project folder in this workspace."""
        from roboflow.adapters import rfapi

        return rfapi.create_folder(self.__api_key, self.url, name, parent_id=parent_id, project_ids=project_ids)

    # -----------------------------------------------------------------
    # Phase 2: Workflow management
    # -----------------------------------------------------------------

    def list_workflows(self):
        """List workflows in this workspace."""
        from roboflow.adapters import rfapi

        return rfapi.list_workflows(self.__api_key, self.url)

    def get_workflow(self, workflow_url):
        """Get workflow details."""
        from roboflow.adapters import rfapi

        return rfapi.get_workflow(self.__api_key, self.url, workflow_url)

    def create_workflow(self, name, definition=None):
        """Create a new workflow."""
        import json

        from roboflow.adapters import rfapi

        config = json.dumps(definition) if definition else None
        return rfapi.create_workflow(self.__api_key, self.url, name=name, config=config)

    # -----------------------------------------------------------------
    # Phase 2: Workspace statistics
    # -----------------------------------------------------------------

    def get_usage(self):
        """Get billing usage report for this workspace."""
        from roboflow.adapters import rfapi

        return rfapi.get_billing_usage(self.__api_key, self.url)

    def get_plan(self):
        """Get workspace plan info and limits."""
        from roboflow.adapters import rfapi

        return rfapi.get_plan_info(self.__api_key)

    # --- Vision Events ---

    def write_vision_event(self, event: Dict[str, Any]) -> dict:
        """Create a single vision event.

        The event dict is passed directly to the server with no client-side
        validation, so new event types and fields work without an SDK update.

        Args:
            event: Event payload containing at minimum ``eventId``,
                ``eventType``, ``useCaseId``, and ``timestamp``.

        Returns:
            Dict with ``eventId`` and ``created``.

        Example:
            >>> ws = rf.workspace()
            >>> ws.write_vision_event({
            ...     "eventId": "evt-001",
            ...     "eventType": "quality_check",
            ...     "useCaseId": "manufacturing-qa",
            ...     "timestamp": "2024-01-15T10:30:00.000Z",
            ...     "eventData": {"result": "pass"},
            ... })
        """
        return vision_events_api.write_event(
            api_key=self.__api_key,
            event=event,
        )

    def write_vision_events_batch(self, events: List[Dict[str, Any]]) -> dict:
        """Create multiple vision events in a single request.

        Args:
            events: List of event payload dicts (server enforces max 100).

        Returns:
            Dict with ``created`` count and ``eventIds`` list.

        Example:
            >>> ws = rf.workspace()
            >>> ws.write_vision_events_batch([
            ...     {"eventId": "e1", "eventType": "custom", "useCaseId": "uc", "timestamp": "2024-01-15T10:00:00Z"},
            ...     {"eventId": "e2", "eventType": "custom", "useCaseId": "uc", "timestamp": "2024-01-15T10:01:00Z"},
            ... ])
        """
        return vision_events_api.write_batch(
            api_key=self.__api_key,
            events=events,
        )

    def query_vision_events(
        self,
        use_case: str,
        *,
        event_type: Optional[str] = None,
        event_types: Optional[List[str]] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
        **filters: Any,
    ) -> dict:
        """Query vision events with filters and pagination.

        Common filter kwargs are passed through to the server as-is,
        supporting ``deviceId``, ``streamId``, ``workflowId``,
        ``customMetadataFilters``, ``eventFieldFilters``, etc.

        Args:
            use_case: Use case identifier to query.
            event_type: Filter by a single event type.
            event_types: Filter by multiple event types.
            start_time: ISO 8601 start time filter.
            end_time: ISO 8601 end time filter.
            limit: Maximum number of events to return.
            cursor: Pagination cursor from a previous response.
            **filters: Additional filter parameters passed to the API.

        Returns:
            Dict with ``events``, ``nextCursor``, ``hasMore``, and ``lookbackDays``.

        Example:
            >>> ws = rf.workspace()
            >>> page = ws.query_vision_events("manufacturing-qa", event_type="quality_check", limit=50)
            >>> for evt in page["events"]:
            ...     print(evt["eventId"])
        """
        payload: Dict[str, Any] = {"useCaseId": use_case}
        if event_type is not None:
            payload["eventType"] = event_type
        if event_types is not None:
            payload["eventTypes"] = event_types
        if start_time is not None:
            payload["startTime"] = start_time
        if end_time is not None:
            payload["endTime"] = end_time
        if limit is not None:
            payload["limit"] = limit
        if cursor is not None:
            payload["cursor"] = cursor
        payload.update(filters)

        return vision_events_api.query(
            api_key=self.__api_key,
            query_params=payload,
        )

    def query_all_vision_events(
        self,
        use_case: str,
        *,
        event_type: Optional[str] = None,
        event_types: Optional[List[str]] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: Optional[int] = None,
        **filters: Any,
    ) -> Generator[List[dict], None, None]:
        """Paginated query across vision events, yielding one page at a time.

        Automatically follows ``nextCursor`` until all matching events have
        been returned.

        Args:
            use_case: Use case identifier to query.
            event_type: Filter by a single event type.
            event_types: Filter by multiple event types.
            start_time: ISO 8601 start time filter.
            end_time: ISO 8601 end time filter.
            limit: Maximum events per page.
            **filters: Additional filter parameters passed to the API.

        Yields:
            A list of event dicts for each page.

        Example:
            >>> ws = rf.workspace()
            >>> for page in ws.query_all_vision_events("manufacturing-qa"):
            ...     for evt in page:
            ...         print(evt["eventId"])
        """
        cursor = None
        while True:
            response = self.query_vision_events(
                use_case,
                event_type=event_type,
                event_types=event_types,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
                cursor=cursor,
                **filters,
            )
            events = response.get("events", [])
            if not events:
                break
            yield events
            cursor = response.get("nextCursor")
            if not cursor or not response.get("hasMore", False):
                break

    def list_vision_event_use_cases(self, status: Optional[str] = None) -> dict:
        """List all vision event use cases for the workspace.

        Args:
            status: Optional status filter (e.g. "active", "inactive").

        Returns:
            Dict with ``useCases`` list and ``lookbackDays``.

        Example:
            >>> ws = rf.workspace()
            >>> result = ws.list_vision_event_use_cases()
            >>> for uc in result["useCases"]:
            ...     print(uc["name"], uc.get("status"))
        """
        result = vision_events_api.list_use_cases(
            api_key=self.__api_key,
            status=status,
        )
        if "useCases" not in result and "solutions" in result:
            result["useCases"] = result["solutions"]
        return result

    def create_vision_event_use_case(self, name: str) -> dict:
        """Create a new vision event use case.

        Args:
            name: Human-readable name for the use case.

        Returns:
            Dict with ``id`` and ``name``.

        Example:
            >>> ws = rf.workspace()
            >>> result = ws.create_vision_event_use_case("manufacturing-qa")
            >>> use_case_id = result["id"]
        """
        return vision_events_api.create_use_case(
            api_key=self.__api_key,
            name=name,
        )

    def rename_vision_event_use_case(self, use_case: str, name: str) -> dict:
        """Rename an existing vision event use case.

        Args:
            use_case: Use case identifier.
            name: New name for the use case.

        Returns:
            Dict with ``id`` and ``name``.

        Example:
            >>> ws = rf.workspace()
            >>> ws.rename_vision_event_use_case("abc123", "new-name")
        """
        return vision_events_api.rename_use_case(
            api_key=self.__api_key,
            use_case_id=use_case,
            name=name,
        )

    def archive_vision_event_use_case(self, use_case: str) -> dict:
        """Archive a vision event use case.

        Args:
            use_case: Use case identifier.

        Returns:
            Dict with ``success``.

        Example:
            >>> ws = rf.workspace()
            >>> ws.archive_vision_event_use_case("abc123")
        """
        return vision_events_api.archive_use_case(
            api_key=self.__api_key,
            use_case_id=use_case,
        )

    def unarchive_vision_event_use_case(self, use_case: str) -> dict:
        """Unarchive a vision event use case.

        Args:
            use_case: Use case identifier.

        Returns:
            Dict with ``success``.

        Example:
            >>> ws = rf.workspace()
            >>> ws.unarchive_vision_event_use_case("abc123")
        """
        return vision_events_api.unarchive_use_case(
            api_key=self.__api_key,
            use_case_id=use_case,
        )

    def get_vision_event_metadata_schema(self, use_case: str) -> dict:
        """Get the custom metadata schema for a vision event use case.

        Returns discovered field names and their types, useful for building
        queries with ``customMetadataFilters``.

        Args:
            use_case: Use case identifier.

        Returns:
            Dict with ``fields`` mapping field names to ``{"types": [...]}``.

        Example:
            >>> ws = rf.workspace()
            >>> schema = ws.get_vision_event_metadata_schema("manufacturing-qa")
            >>> for field, info in schema["fields"].items():
            ...     print(field, info["types"])
        """
        return vision_events_api.get_custom_metadata_schema(
            api_key=self.__api_key,
            use_case_id=use_case,
        )

    def upload_vision_event_image(
        self,
        image_path: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """Upload an image for use in vision events.

        Args:
            image_path: Local path to the image file.
            name: Optional custom name for the image.
            metadata: Optional flat dict of metadata to attach.

        Returns:
            Dict with ``sourceId`` for referencing in events.

        Example:
            >>> ws = rf.workspace()
            >>> result = ws.upload_vision_event_image("photo.jpg")
            >>> source_id = result["sourceId"]
        """
        return vision_events_api.upload_image(
            api_key=self.__api_key,
            image_path=image_path,
            name=name,
            metadata=metadata,
        )

    def trash(self) -> dict:
        """
        List items currently in the workspace Trash.

        Returns a dict with:
          - `items`: flat list of everything in Trash
          - `sections`: grouped by `projects`, `versions`, `workflows`
        Each item includes `id`, `type`, `name`, `deletedAt`,
        `scheduledCleanupAt`, and — for versions — `parentId` / `parentUrl`.

        Example:
            >>> import roboflow
            >>> rf = roboflow.Roboflow(api_key="")
            >>> ws = rf.workspace()
            >>> trash = ws.trash()
            >>> for item in trash["items"]:
            ...     print(item["type"], item["name"])
        """
        return rfapi.list_trash(self.__api_key, self.url)

    def restore_from_trash(self, item_type: str, item_id: str, parent_id: Optional[str] = None):
        """
        Restore an item from Trash.

        Args:
            item_type: one of "project", "version", "workflow"
            item_id: the item's Firestore id (found via `trash()`)
            parent_id: required when restoring a version — the parent project id

        Returns:
            dict: Server response with `{restored: True, type, id}`.
        """
        return rfapi.restore_trash_item(self.__api_key, self.url, item_type, item_id, parent_id)

    # Permanent-delete actions (empty trash / delete a single trash item
    # immediately) are intentionally not exposed in the SDK — they destroy
    # data irrecoverably and are only available through the web UI's Trash
    # view. Items left in Trash are cleaned up automatically after 30 days.

    def __str__(self):
        projects = self.projects()
        json_value = {"name": self.name, "url": self.url, "projects": projects}

        return json.dumps(json_value, indent=2)


def _zip_directory(src_dir: str) -> str:
    """Zip src_dir into a temp file, skipping hidden and macOS-junk entries."""
    fd, zip_path = tempfile.mkstemp(suffix=".zip", prefix="roboflow-upload-")
    os.close(fd)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(src_dir):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__MACOSX"]
            for name in files:
                if name.startswith(".") or name == "Thumbs.db":
                    continue
                abs_path = os.path.join(root, name)
                rel = os.path.relpath(abs_path, src_dir)
                zf.write(abs_path, arcname=rel)
    return zip_path


def _poll_zip_status(
    api_key: str,
    workspace_url: str,
    task_id: str,
    poll_interval: float,
    poll_timeout: float,
) -> dict:
    deadline = time.monotonic() + poll_timeout
    last_progress = None
    while True:
        status = rfapi.get_zip_upload_status(api_key, workspace_url, task_id)
        state = status.get("status")
        progress = (status.get("progress") or {}).get("current")
        if progress is not None and progress != last_progress:
            print(f"  zip-upload progress: {progress}")
            last_progress = progress
        if state in {"completed", "failed"}:
            return status
        if time.monotonic() >= deadline:
            raise RoboflowError(
                f"Zip upload polling timed out after {poll_timeout}s "
                f"(task_id={task_id}, last_status={state}). "
                f"Call Workspace.upload_dataset(..., wait=False) and poll with "
                f"rfapi.get_zip_upload_status to check later."
            )
        time.sleep(poll_interval)
