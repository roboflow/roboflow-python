import json
import os
import urllib
from typing import Optional

import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder

from roboflow.config import API_URL, DEFAULT_BATCH_NAME, DEFAULT_JOB_NAME
from roboflow.util import image_utils


class RoboflowError(Exception):
    pass


class ImageUploadError(RoboflowError):
    def __init__(self, message, status_code=None):
        self.message = message
        self.status_code = status_code
        self.retries = 0
        super().__init__(self.message)


class AnnotationSaveError(RoboflowError):
    def __init__(self, message, status_code=None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


def get_workspace(api_key, workspace_url):
    url = f"{API_URL}/{workspace_url}?api_key={api_key}"
    response = requests.get(url)
    if response.status_code != 200:
        raise RoboflowError(response.text)
    result = response.json()
    return result


def get_project(api_key, workspace_url, project_url):
    url = f"{API_URL}/{workspace_url}/{project_url}?api_key={api_key}"
    response = requests.get(url)
    if response.status_code != 200:
        raise RoboflowError(response.text)
    result = response.json()
    return result


def upload_image(
    api_key,
    project_url,
    image_path: str,
    hosted_image: bool = False,
    split: str = "train",
    batch_name: str = DEFAULT_BATCH_NAME,
    tag_names: list = [],
    sequence_number: Optional[int] = None,
    sequence_size: Optional[int] = None,
    **kwargs,
):
    """
    Upload an image to a specific project.

    Args:
        image_path (str): path to image you'd like to upload
        hosted_image (bool): whether the image is hosted on Roboflow
        split (str): the dataset split the image to
    """

    coalesced_batch_name = batch_name or DEFAULT_BATCH_NAME

    # If image is not a hosted image
    if not hosted_image:
        image_name = os.path.basename(image_path)
        imgjpeg = image_utils.file2jpeg(image_path)

        upload_url = _local_upload_url(
            api_key, project_url, coalesced_batch_name, tag_names, sequence_number, sequence_size, kwargs
        )
        m = MultipartEncoder(
            fields={
                "name": image_name,
                "split": split,
                "file": ("imageToUpload", imgjpeg, "image/jpeg"),
            }
        )
        response = requests.post(upload_url, data=m, headers={"Content-Type": m.content_type}, timeout=(300, 300))

    else:
        # Hosted image upload url
        upload_url = _hosted_upload_url(api_key, project_url, image_path, split, coalesced_batch_name, tag_names)

        # Get response
        response = requests.post(upload_url, timeout=(300, 300))

    responsejson = None
    try:
        responsejson = response.json()
    except Exception:
        pass

    if response.status_code != 200:
        if responsejson:
            err_msg = responsejson

            if err_msg.get("error"):
                err_msg = err_msg["error"]

            if err_msg.get("message"):
                err_msg = err_msg["message"]

            raise ImageUploadError(err_msg, status_code=response.status_code)
        else:
            raise ImageUploadError(str(response), status_code=response.status_code)

    if not responsejson:  # fail fast
        raise ImageUploadError(str(response), status_code=response.status_code)

    if not (responsejson.get("success") or responsejson.get("duplicate")):
        message = responsejson.get("message") or str(responsejson)
        raise ImageUploadError(message)

    return responsejson


def save_annotation(
    api_key: str,
    project_url: str,
    annotation_name: str,
    annotation_string: str,
    image_id: str,
    job_name: str = DEFAULT_JOB_NAME,
    is_prediction: bool = False,
    annotation_labelmap=None,
    overwrite: bool = False,
):
    """
    Upload an annotation to a specific project.

    Args:
        annotation_path (str): path to annotation you'd like to upload
        image_id (str): image id you'd like to upload that has annotations for it.
    """

    upload_url = _save_annotation_url(
        api_key, project_url, annotation_name, image_id, job_name, is_prediction, overwrite
    )

    response = requests.post(
        upload_url,
        data=json.dumps({"annotationFile": annotation_string, "labelmap": annotation_labelmap}),
        headers={"Content-Type": "application/json"},
        timeout=(60, 60),
    )

    # Handle response
    responsejson = None
    try:
        responsejson = response.json()
    except Exception:
        pass

    if not responsejson:
        raise _save_annotation_error(response)
    if response.status_code not in (200, 409):
        raise _save_annotation_error(response)
    if response.status_code == 409:
        if "already annotated" in responsejson.get("error", {}).get("message"):
            return {"warn": "already annotated"}
        else:
            raise _save_annotation_error(response)
    if responsejson.get("error"):
        raise _save_annotation_error(response)
    if not responsejson.get("success"):
        raise _save_annotation_error(response)

    return responsejson


def _save_annotation_url(api_key, project_url, name, image_id, job_name, is_prediction, overwrite=False):
    url = f"{API_URL}/dataset/{project_url}/annotate/{image_id}?api_key={api_key}" f"&name={name}"
    if job_name:
        url += f"&jobName={job_name}"
    if is_prediction:
        url += "&prediction=true"
    if overwrite:
        url += "&overwrite=true"
    return url


def _upload_url(api_key, project_url, **kwargs):
    url = f"{API_URL}/dataset/{project_url}/upload?api_key={api_key}"

    if kwargs:
        querystring = urllib.parse.urlencode(kwargs, doseq=True)
        url += f"&{querystring}"

    return url


def _hosted_upload_url(api_key, project_url, image_path, split, batch_name, tag_names):
    return _upload_url(
        api_key,
        project_url,
        name=os.path.basename(image_path),
        split=split,
        image=image_path,
        batch=batch_name,
        tag=tag_names,
    )


def _local_upload_url(api_key, project_url, batch_name, tag_names, sequence_number, sequence_size, kwargs):
    query_params = dict(batch=batch_name, tag=tag_names, **kwargs)

    if sequence_number is not None and sequence_size is not None:
        query_params.update(sequence_number=sequence_number, sequence_size=sequence_size)

    return _upload_url(api_key, project_url, **query_params)


def _save_annotation_error(response):
    responsejson = None
    try:
        responsejson = response.json()
    except Exception:
        pass

    if not responsejson:
        return AnnotationSaveError(response, status_code=response.status_code)

    if responsejson.get("error"):
        err_msg = responsejson["error"]
        if err_msg.get("message"):
            err_msg = err_msg["message"]
        return AnnotationSaveError(err_msg, status_code=response.status_code)

    return AnnotationSaveError(str(responsejson), status_code=response.status_code)
