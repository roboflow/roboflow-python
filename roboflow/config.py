import os

OBJECT_DETECTION_MODEL = os.getenv(
    "OBJECT_DETECTION_MODEL", default="ObjectDetectionModel"
)
CLASSIFICATION_MODEL = os.getenv("CLASSIFICATION_MODEL", default="ClassificationModel")
PREDICTION_OBJECT = os.getenv("PREDICTION_OBJECT", default="Prediction")
API_URL = os.getenv("API_URL", default="https://api.roboflow.com")
APP_URL = os.getenv("APP_URL", default="https://app.roboflow.com")
CLIP_FEATURIZE_URL = os.getenv(
    "CLIP_FEATURIZE_URL", default="CLIP FEATURIZE URL NOT IN ENV"
)
DEMO_KEYS = ["coco-128-sample", "chess-sample-only-api-key"]
