import json
import os
import urllib
from typing import Dict, List, Optional, Union

import requests
from requests.exceptions import RequestException
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
        self.retries = 0
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


def start_version_training(
    api_key: str,
    workspace_url: str,
    project_url: str,
    version: str,
    *,
    speed: Optional[str] = None,
    checkpoint: Optional[str] = None,
    model_type: Optional[str] = None,
    epochs: Optional[int] = None,
):
    """
    Start a training job for a specific version.

    This is a thin plumbing wrapper around the backend endpoint.
    """
    url = f"{API_URL}/{workspace_url}/{project_url}/{version}/train?api_key={api_key}&nocache=true"

    data: Dict[str, Union[str, int]] = {}
    if speed is not None:
        data["speed"] = speed
    if checkpoint is not None:
        data["checkpoint"] = checkpoint
    if model_type is not None:
        # API expects camelCase
        data["modelType"] = model_type
    if epochs is not None:
        data["epochs"] = epochs

    response = requests.post(url, json=data)
    if not response.ok:
        raise RoboflowError(response.text)
    return True


def get_version(api_key: str, workspace_url: str, project_url: str, version: str, nocache: bool = False):
    """
    Fetch detailed information about a specific dataset version.

    Args:
        api_key: Roboflow API key
        workspace_url: Workspace slug/url
        project_url: Project slug/url
        version: Version identifier (number or slug)
        nocache: If True, bypass server-side cache

    Returns:
        Parsed JSON response from the API.

    Raises:
        RoboflowError: On non-200 response status codes.
    """
    url = f"{API_URL}/{workspace_url}/{project_url}/{version}?api_key={api_key}"
    if nocache:
        url += "&nocache=true"

    response = requests.get(url)
    if response.status_code != 200:
        raise RoboflowError(response.text)
    return response.json()


def get_version_export(
    api_key: str,
    workspace_url: str,
    project_url: str,
    version: str,
    format: str,
):
    """
    Fetch export status or finalized link for a specific version/format.

    Returns either:
      - {"ready": False, "progress": float} when the export is in progress (HTTP 202)
      - The raw JSON payload (dict) from the server when the export is ready (HTTP 200)

    Raises RoboflowError on non-200/202 statuses or invalid/missing JSON when 200/202.
    """
    url = f"{API_URL}/{workspace_url}/{project_url}/{version}/{format}?api_key={api_key}&nocache=true"
    response = requests.get(url)

    # Non-success codes other than 202 are errors
    if response.status_code not in (200, 202):
        raise RoboflowError(response.text)

    try:
        payload = response.json()
    except Exception:
        # If server returns a 200/202 without JSON, treat as error for consumers
        raise RoboflowError(str(response))

    if response.status_code == 202:
        progress = payload.get("progress")
        try:
            progress_val = float(progress) if progress is not None else 0.0
        except Exception:
            progress_val = 0.0
        return {"ready": False, "progress": progress_val}

    # 200 OK: export is ready; return payload unchanged
    return payload


def upload_image(
    api_key,
    project_url,
    image_path: str,
    hosted_image: bool = False,
    split: str = "train",
    batch_name: str = DEFAULT_BATCH_NAME,
    tag_names: Optional[List[str]] = None,
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
    if tag_names is None:
        tag_names = []

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

        try:
            response = requests.post(upload_url, data=m, headers={"Content-Type": m.content_type}, timeout=(300, 300))
        except RequestException as e:
            raise ImageUploadError(str(e)) from e

    else:
        # Hosted image upload url
        upload_url = _hosted_upload_url(api_key, project_url, image_path, split, coalesced_batch_name, tag_names)

        try:
            # Get response
            response = requests.post(upload_url, timeout=(300, 300))
        except RequestException as e:
            raise ImageUploadError(str(e)) from e

    responsejson = None
    try:
        responsejson = response.json()
    except Exception:
        pass

    if response.status_code != 200:
        if responsejson and isinstance(responsejson, dict):
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

    try:
        response = requests.post(
            upload_url,
            data=json.dumps({"annotationFile": annotation_string, "labelmap": annotation_labelmap}),
            headers={"Content-Type": "application/json"},
            timeout=(60, 60),
        )
    except RequestException as e:
        raise AnnotationSaveError(str(e)) from e

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
    url = f"{API_URL}/dataset/{project_url}/annotate/{image_id}?api_key={api_key}&name={name}"
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
