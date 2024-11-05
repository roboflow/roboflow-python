import json
import os
import sys
import time
from getpass import getpass
from pathlib import Path
from urllib.parse import urlparse

import requests

from roboflow.adapters import rfapi
from roboflow.config import API_URL, APP_URL, DEMO_KEYS, load_roboflow_api_key
from roboflow.core.project import Project
from roboflow.core.workspace import Workspace
from roboflow.models import CLIPModel, GazeModel  # noqa: F401
from roboflow.util.general import write_line

__version__ = "1.1.49"


def check_key(api_key, model, notebook, num_retries=0):
    if not isinstance(api_key, str):
        raise RuntimeError(
            "API Key is of Incorrect Type \n Expected Type: " + str(str) + "\n Input Type: " + str(type(api_key))
        )

    if any(c for c in api_key if c.islower()):  # check if any of the api key characters are lowercase
        if api_key in DEMO_KEYS:
            # passthrough for public download of COCO-128 for the time being
            return api_key
        else:
            # validate key normally
            response = requests.post(API_URL + "/?api_key=" + api_key)

            if response.status_code == 401:
                raise RuntimeError(response.text)

            if response.status_code != 200:
                # retry 5 times
                if num_retries < 5:
                    print("retrying...")
                    time.sleep(1)
                    num_retries += 1
                    return check_key(api_key, model, notebook, num_retries)
                else:
                    raise RuntimeError("There was an error validating the api key with Roboflow" " server.")
            else:
                r = response.json()
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


def login(workspace=None, force=False):
    os_name = os.name

    if os_name == "nt":
        default_path = str(Path.home() / "roboflow" / "config.json")
    else:
        default_path = str(Path.home() / ".config" / "roboflow" / "config.json")

    # default configuration location
    conf_location = os.getenv("ROBOFLOW_CONFIG_DIR", default=default_path)
    if os.path.isfile(conf_location) and not force:
        write_line("You are already logged into Roboflow. To make a different login," "run roboflow.login(force=True).")
        return None
        # we could eventually return the workspace object here
        # return Roboflow().workspace()
    elif os.path.isfile(conf_location) and force:
        os.remove(conf_location)

    if workspace is None:
        write_line("visit " + APP_URL + "/auth-cli to get your authentication token.")
    else:
        write_line("visit " + APP_URL + "/auth-cli/?workspace=" + workspace + " to get your authentication token.")

    token = getpass("Paste the authentication token here: ")

    r_login = requests.get(APP_URL + "/query/cliAuthToken/" + token)

    if r_login.status_code == 200:
        r_login = r_login.json()
        if r_login is None:
            raise ValueError("Invalid API key. Please check your API key and try again.")

        # make config directory if it doesn't exist
        if not os.path.exists(os.path.dirname(conf_location)):
            os.makedirs(os.path.dirname(conf_location))

        r_login = {"workspaces": r_login}
        # set first workspace as default workspace

        default_workspace_id = list(r_login["workspaces"].keys())[0]
        workspace = r_login["workspaces"][default_workspace_id]
        r_login["RF_WORKSPACE"] = workspace["url"]

        # write config file
        with open(conf_location, "w") as f:
            json.dump(r_login, f, indent=2)

    else:
        r_login.raise_for_status()

    return None
    # we could eventually return the workspace object here
    # return Roboflow().workspace()


active_workspace = None


def initialize_roboflow(the_workspace=None):
    """High level function to initialize Roboflow.

    Args:
        the_workspace: the workspace url to initialize.
            If None, the default workspace will be used.

    Returns:
        None
    """

    global active_workspace

    conf_location = os.getenv("ROBOFLOW_CONFIG_DIR", default=str(Path.home() / ".config" / "roboflow" / "config.json"))

    if not os.path.isfile(conf_location):
        raise RuntimeError("To use this method, you must first login - run roboflow.login()")
    else:
        if the_workspace is None:
            active_workspace = Roboflow().workspace()
        else:
            active_workspace = Roboflow().workspace(the_workspace)

        return active_workspace


def load_model(model_url):
    """High level function to load Roboflow models.

    Args:
        model_url: the model url to load.
            Must be from either app.roboflow.com or universe.roboflow.com

    Returns:
        the model object to use for inference
    """

    operate_workspace = initialize_roboflow()

    if "universe.roboflow.com" in model_url or "app.roboflow.com" in model_url:
        parsed_url = urlparse(model_url)
        path_parts = parsed_url.path.split("/")
        project = path_parts[2]
        version = int(path_parts[-1])
    else:
        raise ValueError("Model URL must be from either app.roboflow.com or universe.roboflow.com")

    project = operate_workspace.project(project)
    version = project.version(version)
    model = version.model
    return model


def download_dataset(dataset_url, model_format, location=None):
    """High level function to download data from Roboflow.

    Args:
        dataset_url: the dataset url to download.
            Must be from either app.roboflow.com or universe.roboflow.com
        model_format: the format the dataset will be downloaded in
        location: the location the dataset will be downloaded to

    Returns:
        The dataset object with location available as dataset.location
    """

    if "universe.roboflow.com" in dataset_url or "app.roboflow.com" in dataset_url:
        parsed_url = urlparse(dataset_url)
        path_parts = parsed_url.path.split("/")
        project = path_parts[2]
        version = int(path_parts[-1])
        the_workspace = path_parts[1]
    else:
        raise ValueError("Model URL must be from either app.roboflow.com or universe.roboflow.com")
    operate_workspace = initialize_roboflow(the_workspace=the_workspace)

    project = operate_workspace.project(project)
    version = project.version(version)
    return version.download(model_format, location)


# continue distributing this object for back compatibility
class Roboflow:
    def __init__(
        self,
        api_key=None,
        model_format="undefined",
        notebook="undefined",
    ):
        self.api_key = api_key
        if self.api_key is None:
            self.api_key = load_roboflow_api_key()

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
            w = r["workspace"]  # type: ignore[arg-type]
            self.current_workspace = w
            return self

    def workspace(self, the_workspace=None):
        sys.stdout.write("\r" + "loading Roboflow workspace...")
        sys.stdout.write("\n")
        sys.stdout.flush()

        if the_workspace is None:
            the_workspace = self.current_workspace

        if self.api_key:  # Check if api_key was passed during __init__
            api_key = self.api_key
            list_projects = rfapi.get_workspace(api_key, the_workspace)
            return Workspace(list_projects, api_key, the_workspace, self.model_format)

        elif self.api_key in DEMO_KEYS:
            return Workspace({}, self.api_key, the_workspace, self.model_format)

        else:
            raise ValueError("A valid API key must be provided.")

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

        dataset_info = requests.get(API_URL + "/" + the_workspace + "/" + project_name + "?api_key=" + self.api_key)

        # Throw error if dataset isn't valid/user doesn't have
        #   permissions to access the dataset
        if dataset_info.status_code != 200:
            raise RuntimeError(dataset_info.text)

        dataset_info = dataset_info.json()["project"]

        return Project(self.api_key, dataset_info)

    def __str__(self):
        """to string function"""
        json_value = {"api_key": self.api_key, "workspace": self.workspace}
        return json.dumps(json_value, indent=2)
