import os
import json


conf_location = os.getenv(
    "ROBOFLOW_CONFIG_DIR", default=os.getenv("HOME") + "/.config/roboflow/config.json"
)

# read from json file if exists
if os.path.exists(conf_location):
    # read json from file
    with open(conf_location) as f:
        config = json.load(f)
else:
    config = {}


def get_var_conditional(key, default):

    if os.getenv(key) != None:
        return os.getenv(key)
    elif key in config.keys():
        return config[key]
    else:
        return default


CLASSIFICATION_MODEL = get_var_conditional(
    "CLASSIFICATION_MODEL", default="ClassificationModel"
)
INSTANCE_SEGMENTATION_MODEL = "InstanceSegmentationModel"
OBJECT_DETECTION_MODEL = get_var_conditional(
    "OBJECT_DETECTION_MODEL", default="ObjectDetectionModel"
)
SEMANTIC_SEGMENTATION_MODEL = get_var_conditional('SEMANTIC_SEGMENTATION_MODEL", default="SemanticSegmentationModel")
PREDICTION_OBJECT = get_var_conditional("PREDICTION_OBJECT", default="Prediction")

API_URL = get_var_conditional("RF_API_URL", default="https://api.roboflow.com")
APP_URL = get_var_conditional("RF_APP_URL", default="https://app.roboflow.com")
INSTANCE_SEGMENTATION_URL = get_var_conditional(
    "INSTANCE_SEGMENTATION_URL", default="https://outline.roboflow.com"
)
SEMANTIC_SEGMENTATION_URL = get_var_conditional(
    "SEMANTIC_SEGMENTATION_URL", default="https://segment.roboflow.com"
)

RF_WORKSPACES = get_var_conditional("workspaces", default={})

CLIP_FEATURIZE_URL = get_var_conditional(
    "CLIP_FEATURIZE_URL", default="CLIP FEATURIZE URL NOT IN ENV"
)
DEMO_KEYS = ["coco-128-sample", "chess-sample-only-api-key"]

TYPE_CLASSICATION = "classification"
TYPE_OBJECT_DETECTION = "object-detection"
TYPE_INSTANCE_SEGMENTATION = "instance-segmentation"
TYPE_SEMANTIC_SEGMENTATION = "semantic-segmentation"
# pull default workspace and API key

RF_WORKSPACE = get_var_conditional("RF_WORKSPACE", default=None)
# DEFAULT_WORKSPACE = get_var_conditional("default_workspace", default=None)
if RF_WORKSPACE == None:
    RF_API_KEY = None
else:
    RF_API_KEY = None
    for k in RF_WORKSPACES.keys():
        workspace = RF_WORKSPACES[k]
        if workspace["url"] == RF_WORKSPACE:
            RF_API_KEY = workspace["apiKey"]
# ENV API_KEY OVERRIDE
if os.getenv("RF_API_KEY") != None:
    RF_API_KEY = os.getenv("RF_API_KEY")
