from roboflow.config import SEMANTIC_SEGMENTATION_MODEL, SEMANTIC_SEGMENTATION_URL
from roboflow.models.inference import InferenceModel


class SemanticSegmentationModel(InferenceModel):
    """
    Run inference on a semantic segmentation model hosted on Roboflow or served through Roboflow Inference.
    """  # noqa: E501 // docs

    def __init__(self, api_key: str, version_id: str):
        """
        Create a SemanticSegmentationModel object through which you can run inference.

        Args:
            api_key (str): private roboflow api key
            version_id (str): the workspace/project id
        """  # noqa: E501 // docs
        super().__init__(api_key, version_id)
        self.api_url = f"{SEMANTIC_SEGMENTATION_URL}/{self.dataset_id}/{self.version}"

    def predict(self, image_path: str, confidence: int = 50):  # type: ignore[override]
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
        """  # noqa: E501 // docs
        return super().predict(
            image_path,
            confidence=confidence,
            prediction_type=SEMANTIC_SEGMENTATION_MODEL,
        )

    def __str__(self):
        return f"<{type(self).__name__} id={self.id}, api_url={self.api_url}>"
