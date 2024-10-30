import requests

from roboflow.config import DEDICATED_DEPLOYMENT_URL


class DeploymentApiError(Exception):
    pass


def add_deployment(api_key, machine_type, duration, delete_on_expiration, deployment_name, inference_version):
    url = f"{DEDICATED_DEPLOYMENT_URL}/add"
    params = {
        "api_key": api_key,
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
    url = f"{DEDICATED_DEPLOYMENT_URL}/get_log?api_key={api_key}&deployment_name={deployment_name}"
    if from_timestamp is not None:
        url += f"&from_timestamp={from_timestamp.isoformat()}"
    if to_timestamp is not None:
        url += f"&to_timestamp={to_timestamp.isoformat()}"
    if max_entries > 0:
        url += f"&max_entries={max_entries}"
    response = requests.get(url)
    if response.status_code != 200:
        return response.status_code, response.text
    return response.status_code, response.json()
