from typing import Optional

from roboflow.config import INSTANCE_SEGMENTATION_MODEL, INSTANCE_SEGMENTATION_URL
from roboflow.models.inference import InferenceModel


class InstanceSegmentationModel(InferenceModel):
    """
    Run inference on a instance segmentation model hosted on
        Roboflow or served through Roboflow Inference.
    """

    def __init__(
        self,
        api_key: str,
        version_id: str,
        colors: Optional[dict] = None,
        preprocessing: Optional[dict] = None,
        local: Optional[str] = None,
    ):
        """
        Create a InstanceSegmentationModel object through which you can run inference.

        Args:
            api_key (str): private roboflow api key
            version_id (str): the workspace/project id
            colors (dict): colors to use for the image
            preprocessing (dict): preprocessing to use for the image
            local (str): localhost address and port if pointing towards local inference engine
        """
        super().__init__(api_key, version_id)

        base_url = local or INSTANCE_SEGMENTATION_URL
        self.api_url = f"{base_url}/{self.dataset_id}/{self.version}"
        self.colors = {} if colors is None else colors
        self.preprocessing = {} if preprocessing is None else preprocessing

    def predict(self, image_path, confidence=40):  # type: ignore[override]
        """
        Infers detections based on image from a specified model and image path.

        Args:
            image_path (str): path to the image you'd like to perform prediction on
            confidence (int): confidence threshold for predictions, on a scale from 0-100

        Returns:
            PredictionGroup Object

        Example:
            >>> import roboflow

            >>> rf = roboflow.Roboflow(api_key="")

            >>> project = rf.workspace().project("PROJECT_ID")

            >>> model = project.version("1").model

            >>> prediction = model.predict("YOUR_IMAGE.jpg")
        """  # noqa: E501
        return super().predict(
            image_path,
            confidence=confidence,
            prediction_type=INSTANCE_SEGMENTATION_MODEL,
        )

    def __str__(self):
        return f"<{type(self).__name__} id={self.id}, api_url={self.api_url}>"
