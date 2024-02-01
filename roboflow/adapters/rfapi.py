import json
import os
import urllib

import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder

from roboflow.config import API_URL, DEFAULT_BATCH_NAME
from roboflow.util import image_utils


class RoboflowError(Exception):
    pass


class UploadError(RoboflowError):
    pass


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
    sequence_number: int = None,
    sequence_size: int = None,
    **kwargs,
):
    """
    Upload an image to a specific project.

    Args:
        image_path (str): path to image you'd like to upload
        hosted_image (bool): whether the image is hosted on Roboflow
        split (str): the dataset split the image to
    """

    # If image is not a hosted image
    if not hosted_image:
        batch_name = batch_name or DEFAULT_BATCH_NAME
        image_name = os.path.basename(image_path)
        imgjpeg = image_utils.file2jpeg(image_path)

        upload_url = _local_upload_url(
            api_key, project_url, batch_name, tag_names, sequence_number, sequence_size, kwargs
        )
        m = MultipartEncoder(
            fields={
                "name": image_name,
                "split": split,
                "file": ("imageToUpload", imgjpeg, "image/jpeg"),
            }
        )
        response = requests.post(upload_url, data=m, headers={"Content-Type": m.content_type})

    else:
        # Hosted image upload url

        upload_url = _hosted_upload_url(api_key, project_url, image_path, split)
        # Get response
        response = requests.post(upload_url)
    responsejson = None
    try:
        responsejson = response.json()
    except Exception:
        pass
    if response.status_code != 200:
        if responsejson:
            raise UploadError(f"Bad response: {response.status_code}: {responsejson}")
        else:
            raise UploadError(f"Bad response: {response}")
    if not responsejson:  # fail fast
        raise UploadError(f"upload image {image_path} 200 OK, unexpected response: {response}")
    if not (responsejson.get("success") or responsejson.get("duplicate")):
        raise UploadError(f"Server rejected image: {responsejson}")
    return responsejson


def save_annotation(
    api_key: str,
    project_url: str,
    annotation_name: str,
    annotation_string: str,
    image_id: str,
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

    upload_url = _save_annotation_url(api_key, project_url, annotation_name, image_id, is_prediction, overwrite)

    response = requests.post(
        upload_url,
        data=json.dumps({"annotationFile": annotation_string, "labelmap": annotation_labelmap}),
        headers={"Content-Type": "application/json"},
    )
    responsejson = None
    try:
        responsejson = response.json()
    except Exception:
        pass
    if not responsejson:
        raise _save_annotation_error(image_id, response)
    if response.status_code not in (200, 409):
        raise _save_annotation_error(image_id, response)
    if response.status_code == 409:
        if "already annotated" in responsejson.get("error", {}).get("message"):
            return {"warn": "already annotated"}
        else:
            raise _save_annotation_error(image_id, response)
    if responsejson.get("error"):
        raise _save_annotation_error(image_id, response)
    if not responsejson.get("success"):
        raise _save_annotation_error(image_id, response)
    return responsejson


def _save_annotation_url(api_key, project_url, name, image_id, is_prediction, overwrite=False):
    url = f"{API_URL}/dataset/{project_url}/annotate/{image_id}?api_key={api_key}" f"&name={name}"
    if is_prediction:
        url += "&prediction=true"
    if overwrite:
        url += "&overwrite=true"
    return url


def _hosted_upload_url(api_key, project_url, image_path, split):
    url = f"{API_URL}/dataset/{project_url}/upload?api_key={api_key}"
    url += f"&name={os.path.basename(image_path)}&split={split}"
    url += f"&image={urllib.parse.quote_plus(image_path)}"
    return url


def _local_upload_url(api_key, project_url, batch_name, tag_names, sequence_number, sequence_size, kwargs):
    url = f"{API_URL}/dataset/{project_url}/upload?api_key={api_key}&batch={batch_name}"
    if sequence_number is not None and sequence_size is not None:
        url += f"&sequence_number={sequence_number}&sequence_size={sequence_size}"
    for key, value in kwargs.items():
        url += f"&{str(key)}={str(value)}"
    for tag in tag_names:
        url += f"&tag={tag}"
    return url


def _save_annotation_error(image_id, response):
    errmsg = f"save annotation for {image_id} / "
    responsejson = None
    try:
        responsejson = response.json()
    except Exception:
        pass
    if not responsejson:
        errmsg += f"bad response: {response.status_code}: {response}"
    elif responsejson.get("error"):
        errmsg += f"bad response: {response.status_code}: {responsejson['error']}"
    else:
        errmsg += f"bad response: {response.status_code}: {responsejson}"
    return UploadError(errmsg)
