import urllib

import requests

from roboflow.config import DEDICATED_DEPLOYMENT_URL


class DeploymentApiError(Exception):
    pass


def add_deployment(
    api_key, creator_email, machine_type, duration, delete_on_expiration, deployment_name, inference_version
):
    url = f"{DEDICATED_DEPLOYMENT_URL}/add"
    params = {
        "api_key": api_key,
        "creator_email": creator_email,
        # "security_level": security_level,
        "duration": duration,
        "delete_on_expiration": delete_on_expiration,
        "deployment_name": deployment_name,
        "inference_version": inference_version,
    }
    if machine_type is not None:
        params["machine_type"] = machine_type
    response = requests.post(url, json=params)
    if response.status_code != 200:
        return response.status_code, response.text
    return response.status_code, response.json()


def get_deployment(api_key, deployment_name):
    url = f"{DEDICATED_DEPLOYMENT_URL}/get?api_key={api_key}&deployment_name={deployment_name}"
    response = requests.get(url)
    if response.status_code != 200:
        return response.status_code, response.text
    return response.status_code, response.json()


def list_deployment(api_key):
    url = f"{DEDICATED_DEPLOYMENT_URL}/list?api_key={api_key}"
    response = requests.get(url)
    if response.status_code != 200:
        return response.status_code, response.text
    return response.status_code, response.json()


def get_workspace_usage(api_key, from_timestamp, to_timestamp):
    params = {"api_key": api_key}
    if from_timestamp is not None:
        params["from_timestamp"] = from_timestamp.isoformat()  # may contain + sign
    if to_timestamp is not None:
        params["to_timestamp"] = to_timestamp.isoformat()  # may contain + sign
    url = f"{DEDICATED_DEPLOYMENT_URL}/usage_workspace?{urllib.parse.urlencode(params)}"
    response = requests.get(url)
    if response.status_code != 200:
        return response.status_code, response.text
    return response.status_code, response.json()


def get_deployment_usage(api_key, deployment_name, from_timestamp, to_timestamp):
    params = {"api_key": api_key, "deployment_name": deployment_name}
    if from_timestamp is not None:
        params["from_timestamp"] = from_timestamp.isoformat()  # may contain + sign
    if to_timestamp is not None:
        params["to_timestamp"] = to_timestamp.isoformat()  # may contain + sign
    url = f"{DEDICATED_DEPLOYMENT_URL}/usage_deployment?{urllib.parse.urlencode(params)}"
    response = requests.get(url)
    if response.status_code != 200:
        return response.status_code, response.text
    return response.status_code, response.json()


def pause_deployment(api_key, deployment_name):
    url = f"{DEDICATED_DEPLOYMENT_URL}/pause"
    response = requests.post(url, json={"api_key": api_key, "deployment_name": deployment_name})
    if response.status_code != 200:
        return response.status_code, response.text
    return response.status_code, response.json()


def resume_deployment(api_key, deployment_name):
    url = f"{DEDICATED_DEPLOYMENT_URL}/resume"
    response = requests.post(url, json={"api_key": api_key, "deployment_name": deployment_name})
    if response.status_code != 200:
        return response.status_code, response.text
    return response.status_code, response.json()


def delete_deployment(api_key, deployment_name):
    url = f"{DEDICATED_DEPLOYMENT_URL}/delete"
    response = requests.post(url, json={"api_key": api_key, "deployment_name": deployment_name})
    if response.status_code != 200:
        return response.status_code, response.text
    return response.status_code, response.json()


def list_machine_types(api_key):
    url = f"{DEDICATED_DEPLOYMENT_URL}/machine_types?api_key={api_key}"
    response = requests.get(url)
    if response.status_code != 200:
        return response.status_code, response.text
    return response.status_code, response.json()


def get_deployment_log(api_key, deployment_name, from_timestamp=None, to_timestamp=None, max_entries=-1):
    params = {"api_key": api_key, "deployment_name": deployment_name}
    if from_timestamp is not None:
        params["from_timestamp"] = from_timestamp.isoformat()  # may contain + sign
    if to_timestamp is not None:
        params["to_timestamp"] = to_timestamp.isoformat()  # may contain + sign
    if max_entries > 0:
        params["max_entries"] = max_entries
    url = f"{DEDICATED_DEPLOYMENT_URL}/get_log?{urllib.parse.urlencode(params)}"
    response = requests.get(url)
    if response.status_code != 200:
        return response.status_code, response.text
    return response.status_code, response.json()
