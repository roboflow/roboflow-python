import requests

from roboflow.config import DEDICATED_DEPLOYMENT_URL


class DeploymentApiError(Exception):
    pass


def add_deployment(api_key, machine_type, deployment_name, inference_version):
    url = f"{DEDICATED_DEPLOYMENT_URL}/add"
    response = requests.post(
        url,
        json={
            "api_key": api_key,
            # "security_level": security_level,
            "machine_type": machine_type,
            "deployment_name": deployment_name,
            "inference_version": inference_version,
        },
    )
    if response.status_code != 200:
        raise DeploymentApiError(f"{response.status_code}: {response.text}")
    result = response.json()
    return result


def get_deployment(api_key, deployment_id):
    url = f"{DEDICATED_DEPLOYMENT_URL}/get"
    response = requests.get(url, json={"api_key": api_key, "deployment_id": deployment_id})
    if response.status_code != 200:
        raise DeploymentApiError(f"{response.status_code}: {response.text}")
    result = response.json()
    return result


def list_deployment(api_key):
    url = f"{DEDICATED_DEPLOYMENT_URL}/list"
    response = requests.get(url, json={"api_key": api_key})
    if response.status_code != 200:
        raise DeploymentApiError(f"{response.status_code}: {response.text}")
    result = response.json()
    return result


def delete_deployment(api_key, deployment_id):
    url = f"{DEDICATED_DEPLOYMENT_URL}/delete"
    response = requests.post(url, json={"api_key": api_key, "deployment_id": deployment_id})
    if response.status_code != 200:
        raise DeploymentApiError(f"{response.status_code}: {response.text}")
    result = response.json()
    return result


def list_machine_types(api_key):
    url = f"{DEDICATED_DEPLOYMENT_URL}/machine_types"
    response = requests.get(url, json={"api_key": api_key})
    if response.status_code != 200:
        raise DeploymentApiError(f"{response.status_code}: {response.text}")
    result = response.json()
    return result
