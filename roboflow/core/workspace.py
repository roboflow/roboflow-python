import requests
import json
from roboflow.core.project import Project
from roboflow.config import *
import sys

class Workspace():
    def __init__(self, info, api_key, default_workspace, model_format):
        if api_key == "coco-128-sample":
            self.__api_key = api_key
            self.model_format = model_format
        else:
            workspace_info = info['workspace']
            self.name = workspace_info['name']
            self.project_list = workspace_info['projects']
            if "members" in workspace_info.keys():
                self.members = workspace_info['members']
            self.url = workspace_info['url']
            self.model_format = model_format

            self.__api_key = api_key


    def list_projects(self):
        """Lists projects out in the workspace
        """
        print(self.project_list)

    def projects(self):
        """Returns all projects as Project() objects in the workspace
        :return an array of project objects
        """
        projects_array = []
        for a_project in self.project_list:
            proj = Project(self.__api_key, a_project, self.model_format)
            projects_array.append(proj.name)

        return projects_array


    def project(self, project_name):
        sys.stdout.write("\r" + "loading Roboflow project...")
        sys.stdout.write("\n")
        sys.stdout.flush()

        if self.__api_key == "coco-128-sample":
            return Project(self.__api_key, {}, self.model_format)
        
        project_name = project_name.replace(self.url + "/", "")

        if "/" in project_name:
            raise RuntimeError("The {} project is not available in this ({}) workspace".format(project_name, self.url))

        dataset_info = requests.get(API_URL + "/" + self.url + "/" + project_name + "?api_key=" + self.__api_key)

        # Throw error if dataset isn't valid/user doesn't have permissions to access the dataset
        if dataset_info.status_code != 200:
            raise RuntimeError(dataset_info.text)

        dataset_info = dataset_info.json()['project']

        return Project(self.__api_key, dataset_info, self.model_format)

    def __str__(self):
        projects = self.projects()
        json_value = {'name': self.name,
                      'url': self.url,
                      'projects': projects
                      }

        return json.dumps(json_value, indent=2)
