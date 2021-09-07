from roboflow.core.model import Model


class Version():
    def __init__(self, a_version):
        self.version_id = a_version['id']
        self.model = Model(a_version['model'])


