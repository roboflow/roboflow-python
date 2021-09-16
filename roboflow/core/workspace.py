import requests
import json
from roboflow.core.project import Project
from roboflow.config import *

class Workspace():
    """Workspace class that stores information about a specific workspace
    :param info: dictionary that contains all of the workspace, project information from the REST API.
    :param api_key: user private roboflow key
    :param default_workspace: the workspace name
    :return workspace object
    """
    def __init__(self, info, api_key, default_workspace):
        workspace_info = info['workspace']
        self.name = workspace_info['name']
        self.project_list = workspace_info['projects']
        self.members = workspace_info['members']
        self.url = workspace_info['url']

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
            proj = Project(self.__api_key, a_project)
            projects_array.append(proj)

        return projects_array


    def project(self, project_name):
        """Retrieves all information about a project from the REST API.
        :param project_name: name of project you're trying to retrieve information about
        :return a project object.
        """
        project_name = project_name.replace(self.url + "/", "")

        if "/" in project_name:
            raise RuntimeError("The {} project is not available in this ({}) workspace".format(project_name, self.url))

        dataset_info = requests.get(API_URL + "/" + self.url + "/" + project_name + "?api_key=" + self.__api_key)

        # Throw error if dataset isn't valid/user doesn't have permissions to access the dataset
        if dataset_info.status_code != 200:
            raise RuntimeError(dataset_info.text)

        dataset_info = dataset_info.json()['project']

        return Project(self.__api_key, dataset_info)

    def __str__(self):
        """to string for the workspace"""
        json_value = {'name': self.name,
                      'url': self.url,
                      'members': self.members,
                      'projects': self.projects
                      }

        return json.dumps(json_value, indent=2)
