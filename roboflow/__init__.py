import json
import sys
import os

import requests
#import inquirer

from roboflow.config import API_URL, APP_URL, DEMO_KEYS, RF_API_KEY
from roboflow.core.project import Project
from roboflow.core.workspace import Workspace
from roboflow.util.general import write_line



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
        api_key=None,
        model_format="undefined",
        notebook="undefined",
    ):
        if api_key == None:
            if RF_API_KEY != None:

                api_key = RF_API_KEY
        self.api_key = api_key
        self.model_format = model_format
        self.notebook = notebook
        self.onboarding = False

        if api_key == None:
            self.login()
        else:
            self.auth()

    def auth(self):

        if self.api_key == None:
            from roboflow.config import RF_API_KEY
            if RF_API_KEY != None:
                self.api_key = RF_API_KEY

        print("API KEY IS : ", self.api_key)
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

    def login(self):
        write_line("visit " + APP_URL + "/auth-cli" " to get an API KEY")

        token = input("Paste the authentication here token here: ")

        """
            const cli_auth_token = token_input.cli_auth_token;
            const authDataResponse = await axios.get(
                `https://${conf.get("app_domain")}/query/cliAuthToken/${cli_auth_token}`
            );
            const authData = authDataResponse.data;
            conf.set("workspaces", authData);
        """

        r_login = requests.get("https://app.roboflow.one/query/cliAuthToken/" + token)
        


        if r_login.status_code == 200 or r_login.json() == None:

            r_login = r_login.json()

            conf_location = os.getenv(
                "ROBOFLOW_CONFIG_DIR", default=os.getenv("HOME") + "/.config/roboflow/config.json"
            )

            print("r_login repsonse is : ", r_login)

            r_login = {"workspaces": r_login}

            workpace_selector = []
            for k in r_login["workspaces"].keys():
                workspace = r_login["workspaces"][k]
                workpace_selector.append(workspace["name"] + " " + "(" + workspace["url"] + ")")

            questions = [
            inquirer.List('workspace',
                            message="What size do you need?",
                            choices=workpace_selector,
                        ),
            ]
            answers = inquirer.prompt(questions)
            s = answers["workspace"]
            r_login["RF_WORKSPACE"] = s[s.find("(")+1:s.find(")")]

            print("r_login", r_login)

            with open(conf_location, "w") as outfile:
                json.dump(r_login, outfile)

            return self.auth()
        else:
            raise RuntimeError("Error logging in")
        #return self.auth()

    def workspace(self, the_workspace=None):
        write_line("loading Roboflow workspace...")

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
