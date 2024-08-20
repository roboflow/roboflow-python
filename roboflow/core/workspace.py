import concurrent.futures
import glob
import json
import os
import sys
from typing import Any, List

import requests
from PIL import Image

from roboflow.adapters import rfapi
from roboflow.adapters.rfapi import AnnotationSaveError, ImageUploadError, RoboflowError
from roboflow.config import API_URL, CLIP_FEATURIZE_URL, DEMO_KEYS
from roboflow.core.project import Project
from roboflow.util import folderparser
from roboflow.util.active_learning_utils import check_box_size, clip_encode, count_comparisons
from roboflow.util.image_utils import load_labelmap
from roboflow.util.two_stage_utils import ocr_infer


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
            project_license (str): license of the project (set to `private` for private projects, only available for paid customers)
            annotation (str): annotation of the project

        Returns:
            Project Object
        """  # noqa: E501 // docs
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

        return self.project(r.json()["id"].split("/")[-1])

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
            print("please use an object detection model--can only perform two stage with" " bounding box results")

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
    ):
        """
        Upload a dataset to Roboflow.

        Args:
            dataset_path (str): path to the dataset
            project_name (str): name of the project
            num_workers (int): number of workers to use for parallel uploads
            dataset_format (str): format of the dataset (`voc`, `yolov8`, `yolov5`)
            project_license (str): license of the project (set to `private` for private projects, only available for paid customers)
            project_type (str): type of the project (only `object-detection` is supported)
        """  # noqa: E501 // docs
        if dataset_format != "NOT_USED":
            print("Warning: parameter 'dataset_format' is deprecated and will be removed in a future release")
        parsed_dataset = folderparser.parsefolder(dataset_path)
        project, created = self._get_or_create_project(
            project_id=project_name, license=project_license, type=project_type
        )
        if created:
            print(f"Created project {project.id}")
        else:
            print(f"Uploading to existing project {project.id}")
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
            )

            return image, upload_time, upload_retry_attempts

        def _save_annotation(image_id, imagedesc):
            labelmap = None
            annotation_path = None

            annotationdesc = imagedesc.get("annotationfile")
            if annotationdesc:
                if annotationdesc.get("rawText"):
                    annotation_path = annotationdesc
                else:
                    annotation_path = f"{location}{annotationdesc['file']}"
                labelmap = annotationdesc.get("labelmap")

                if isinstance(labelmap, str):
                    labelmap = load_labelmap(labelmap)

            if not annotation_path:
                return None, None

            annotation, upload_time = project.save_annotation(
                annotation_path=annotation_path,
                annotation_labelmap=labelmap,
                image_id=image_id,
                job_name=batch_name,
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
        inference_endpoint: list = [],
        upload_destination: str = "",
        conditionals: dict = {},
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

    def __str__(self):
        projects = self.projects()
        json_value = {"name": self.name, "url": self.url, "projects": projects}

        return json.dumps(json_value, indent=2)
