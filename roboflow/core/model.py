class Model:
    """
    A Roboflow model.
    """

    def __init__(self, model):
        self.id = model["id"]
        self.endpoint = model["endpoint"]
        self.duration = model["end"] - model["start"]
        self.statistics = {
            "recall": model["recall"],
            "precision": model["precision"],
            "map": model["map"],
        }
