import os

CLASSIFICATION_MODEL = os.getenv("CLASSIFICATION_MODEL", "ClassificationModel")
INSTANCE_SEGMENTATION_MODEL = "InstanceSegmentationModel"
OBJECT_DETECTION_MODEL = os.getenv("OBJECT_DETECTION_MODEL", "ObjectDetectionModel")
SEMANTIC_SEGMENTATION_MODEL = "SemanticSegmentationModel"
PREDICTION_OBJECT = os.getenv("PREDICTION_OBJECT", "Prediction")

API_URL = os.getenv("API_URL", "https://api.roboflow.com")
APP_URL = os.getenv("APP_URL", "https://app.roboflow.com")
INSTANCE_SEGMENTATION_URL = os.getenv(
    "INSTANCE_SEGMENTATION_URL", "https://outline.roboflow.com"
)
SEMANTIC_SEGMENTATION_URL = os.getenv(
    "SEMANTIC_SEGMENTATION_URL", "https://segment.roboflow.com"
)

CLIP_FEATURIZE_URL = os.getenv("CLIP_FEATURIZE_URL", "CLIP FEATURIZE URL NOT IN ENV")
OCR_URL = os.getenv("OCR_URL", "OCR URL NOT IN ENV")

DEMO_KEYS = ["coco-128-sample", "chess-sample-only-api-key"]

TYPE_CLASSICATION = "classification"
TYPE_OBJECT_DETECTION = "object-detection"
TYPE_INSTANCE_SEGMENTATION = "instance-segmentation"
TYPE_SEMANTIC_SEGMENTATION = "semantic-segmentation"

DEFAULT_BATCH_NAME = "Pip Package Upload"
