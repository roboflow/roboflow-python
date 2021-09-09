import requests
from roboflow.core.project import Project
from roboflow.config import *


class Workspace():
    def __init__(self, info, api_key, default_workspace):
        self.api_key = api_key
        self.current_workspace = default_workspace

        workspace_info = info['workspace']
        self.num_members = workspace_info['members']
        self.name = workspace_info['name']
        self.url = workspace_info['url']

        self.projects = []
        self.fill_projects(workspace_info['projects'])

    def fill_projects(self, projects):
        for value in projects:
            self.projects.append(value)

    def project(self, project_name):

        if "/" in project_name:
            raise RuntimeError("Do not re-specify the workspace {} in your project request".format(project_name.rsplit()[0]))

        dataset_info = requests.get(API_URL + "/" + self.current_workspace + "/" + project_name + "?api_key=" + self.api_key)

        # Throw error if dataset isn't valid/user doesn't have permissions to access the dataset
        if dataset_info.status_code != 200:
            raise RuntimeError(dataset_info.text)

        dataset_info = dataset_info.json()['project']

        return Project(self.api_key, dataset_info['id'], dataset_info['type'], dataset_info['versions'])
