class Workspace():
    def __init__(self, info):
        workspace = info['workspace']

        self.num_members = workspace['members']
        self.name = workspace['name']
        self.url = workspace['url']
        self.projects = []

        self.fill_projects(workspace['projects'])

    def fill_projects(self, projects):
        for value in projects:
            self.projects.append(value)


