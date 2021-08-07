import json
import time

import requests

from roboflow.core.project import Project

TOKEN = None
API_URL = "http://localhost:5000"
TOKEN_EXPIRES = None
USER_API_KEY = ""


def auth(api_key):
    global TOKEN, TOKEN_EXPIRES
    global USER_API_KEY
    USER_API_KEY = api_key
    response = requests.post(API_URL + "/token", data=({
        "api_key": api_key
    }))

    r = response.json()
    if "error" in r or response.status_code != 200:
        raise RuntimeError(response.text)

    TOKEN = r['token']
    TOKEN_EXPIRES = r['expires_in']
    return Roboflow(api_key, TOKEN, TOKEN_EXPIRES)


class Roboflow():
    def __init__(self, api_key, access_token, token_expires):
        self.api_key = api_key
        self.access_token = access_token
        self.token_expires = token_expires

    def list_datasets(self):
        get_datasets_endpoint = API_URL + '/datasets'
        datasets = requests.get(get_datasets_endpoint + '?access_token=' + self.access_token).json()
        print(json.dumps(datasets, indent=2))
        return datasets

    def load(self, dataset_slug):
        dataset_endpoint = API_URL + "/dataset/" + dataset_slug
        response = requests.get(dataset_endpoint + "?access_token=" + self.access_token).json()
        return Project(self.api_key, response['dataset_slug'], response['type'], response['versions'])

    def __str__(self):
        json_value = {'api_key': self.api_key, 'auth_token': self.access_token, 'token_expires': self.token_expires}
        return json.dumps(json_value, indent=2)