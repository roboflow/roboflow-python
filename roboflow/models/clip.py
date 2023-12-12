from .inference import InferenceModel


class CLIPModel(InferenceModel):
    """
    Run inference on CLIP, hosted on Roboflow.
    """

    def __init__(self, api_key: str):
        """
        Initialize a CLIP model.

        Args:
            api_key: Your Roboflow API key.
        """
        super().__init__(api_key=api_key, version_id="BASE_MODEL")
