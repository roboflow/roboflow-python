from .inference import InferenceModel


class GazeModel(InferenceModel):
    """
    Run inference on a gaze detection model, hosted on Roboflow.
    """

    def __init__(self, api_key: str, version_id: str):
        """
        Initialize a CLIP model.

        Args:
            api_key: Your Roboflow API key.
            version_id (str): the ID of the dataset version to use for inference
        """
        super().__init__(api_key=api_key, version_id=version_id)
