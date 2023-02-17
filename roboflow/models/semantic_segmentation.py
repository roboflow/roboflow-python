from roboflow.config import SEMANTIC_SEGMENTATION_MODEL, SEMANTIC_SEGMENTATION_URL
from roboflow.models.inference import InferenceModel


class SemanticSegmentationModel(InferenceModel):
    def __init__(self, api_key, version_id):
        """
        :param api_key: Your API key (obtained via your workspace API settings page)
        :param version_id: The ID of the dataset version to use for predicting
        """
        super(SemanticSegmentationModel, self).__init__(api_key, version_id)
        self.api_url = f"{SEMANTIC_SEGMENTATION_URL}/{self.dataset_id}/{self.version}"

    def predict(self, image_path, confidence=50):
        """
        Infers detections based on image from a specified model and image path

        :param image_path: Path to image (can be local path or hosted URL)
        :param confidence: A threshold for the returned predictions on a scale of 0-100. A lower number will return more predictions. A higher number will return fewer, high-certainty predictions.

        :return: PredictionGroup - a group of predictions based on Roboflow JSON response
        """
        return super(SemanticSegmentationModel, self).predict(
            image_path,
            confidence=confidence,
            prediction_type=SEMANTIC_SEGMENTATION_MODEL,
        )

    def __str__(self):
        return f"<{type(self).__name__} id={self.id}, api_url={self.api_url}>"
