import requests
from roboflow.core.project import Project
from roboflow.config import *

class Workspace():
    def __init__(self, info, api_key, default_workspace):
        self.api_key = api_key
        self.name = default_workspace

        workspace_info = info['workspace']
        self.url = workspace_info['url']
        self.project_list = []

        for value in info['workspace']['projects']:
            self.project_list.append(value)

        self.set_class_variables(info)

    def set_class_variables(self, info):
        w = info['workspace']
        self.members=w['members']
        self.projects=w['projects']
        self.url=w['url']

    def list_projects(self):
        print(self.projects)

    def projects(self):
        projects_array = []
        for a_project in self.project_list:
            split = a_project['id'].rsplit("/")
            workspace, project_name = split[0], split[1]
            proj = Project(self.api_key, project_name, a_project['type'], workspace)
            projects_array.append(proj)

        return projects_array


    def project(self, project_name):

        if "/" in project_name:
            raise RuntimeError("Do not re-specify the workspace {} in your project request".format(project_name.rsplit()[0]))

        dataset_info = requests.get(API_URL + "/" + self.name + "/" + project_name + "?api_key=" + self.api_key)

        # Throw error if dataset isn't valid/user doesn't have permissions to access the dataset
        if dataset_info.status_code != 200:
            raise RuntimeError(dataset_info.text)

        dataset_info = dataset_info.json()['project']
        split = dataset_info['id'].rsplit("/")
        workspace, project_name = split[0], split[1]

        return Project(self.api_key, project_name, dataset_info['type'], workspace)
