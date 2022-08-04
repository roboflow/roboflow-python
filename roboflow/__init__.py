import json
import sys

import requests

from roboflow.config import API_URL, APP_URL, DEMO_KEYS
from roboflow.core.project import Project
from roboflow.core.workspace import Workspace


def check_key(api_key, model, notebook):
    if type(api_key) is not str:
        raise RuntimeError(
            "API Key is of Incorrect Type \n Expected Type: "
            + str(type(""))
            + "\n Input Type: "
            + str(type(api_key))
        )

    if any(
        c for c in api_key if c.islower()
    ):  # check if any of the api key characters are lowercase
        if api_key in DEMO_KEYS:
            # passthrough for public download of COCO-128 for the time being
            return api_key
        else:
            # validate key normally
            response = requests.post(API_URL + "/?api_key=" + api_key)
            r = response.json()

            if "error" in r or response.status_code != 200:
                raise RuntimeError(response.text)
            else:
                return r
    else:  # then you're using a dummy key
        sys.stdout.write(
            "upload and label your dataset, and get an API KEY here: "
            + APP_URL
            + "/?model="
            + model
            + "&ref="
            + notebook
            + "\n"
        )
        return "onboarding"


def auth(api_key):
    r = check_key(api_key)
    w = r["workspace"]

    return Roboflow(api_key, w)


class Roboflow:
    def __init__(
        self,
        api_key="YOUR ROBOFLOW API KEY HERE",
        model_format="undefined",
        notebook="undefined",
    ):
        self.api_key = api_key
        self.model_format = model_format
        self.notebook = notebook
        self.onboarding = False
        self.auth()

    def auth(self):
        r = check_key(self.api_key, self.model_format, self.notebook)

        if r == "onboarding":
            self.onboarding = True
            return self
        elif r in DEMO_KEYS:
            self.universe = True
            return self
        else:
            w = r["workspace"]
            self.current_workspace = w
            return self

    def workspace(self, the_workspace=None):
        sys.stdout.write("\r" + "loading Roboflow workspace...")
        sys.stdout.write("\n")
        sys.stdout.flush()

        if the_workspace is None:
            the_workspace = self.current_workspace

        if self.api_key in DEMO_KEYS:
            return Workspace({}, self.api_key, the_workspace, self.model_format)

        list_projects = requests.get(
            API_URL + "/" + the_workspace + "?api_key=" + self.api_key
        ).json()

        return Workspace(list_projects, self.api_key, the_workspace, self.model_format)

    def project(self, project_name, the_workspace=None):
        """Function that takes in the name of the project and returns the project object
        :param project_name api_key: project name
        :param the_workspace workspace name
        :return project object
        """

        if the_workspace is None:
            if "/" in project_name:
                splitted_project = project_name.rsplit("/")
                the_workspace, project_name = splitted_project[0], splitted_project[1]
            else:
                the_workspace = self.current_workspace

        dataset_info = requests.get(
            API_URL
            + "/"
            + the_workspace
            + "/"
            + project_name
            + "?api_key="
            + self.api_key
        )

        # Throw error if dataset isn't valid/user doesn't have permissions to access the dataset
        if dataset_info.status_code != 200:
            raise RuntimeError(dataset_info.text)

        dataset_info = dataset_info.json()["project"]

        return Project(self.api_key, dataset_info)

    def __str__(self):
        """to string function"""
        json_value = {"api_key": self.api_key, "workspace": self.workspace}
        return json.dumps(json_value, indent=2)
