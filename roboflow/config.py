import os
import json

#default configuration location
conf_location = os.getenv(
    "ROBOFLOW_CONFIG_DIR", default=os.getenv("HOME") + "/.config/roboflow/config.json"
)

#read config file for roboflow if logged in from python or CLI
if os.path.exists(conf_location):
    with open(conf_location) as f:
        config = json.load(f)
else:
    config = {}


#read configuration variables conditionally

##1. check if variable is in environment
##2. check if variable is in config file
##3. return default value
def get_var_conditional(key, default):
    if os.getenv(key) != None:
        return os.getenv(key)
    elif key in config.keys():
        return config[key]
    else:
        return default

    

CLASSIFICATION_MODEL = os.getenv("CLASSIFICATION_MODEL", "ClassificationModel")
INSTANCE_SEGMENTATION_MODEL = "InstanceSegmentationModel"
OBJECT_DETECTION_MODEL = os.getenv("OBJECT_DETECTION_MODEL", "ObjectDetectionModel")
SEMANTIC_SEGMENTATION_MODEL = "SemanticSegmentationModel"
PREDICTION_OBJECT = os.getenv("PREDICTION_OBJECT", "Prediction")

API_URL = get_var_conditional("API_URL", "https://api.roboflow.com")
APP_URL = get_var_conditional("APP_URL", "https://app.roboflow.com")
UNIVERSE_URL = get_var_conditional("UNIVERSE_URL", "https://universe.roboflow.com")

INSTANCE_SEGMENTATION_URL = get_var_conditional(
    "INSTANCE_SEGMENTATION_URL", "https://outline.roboflow.com"
)
SEMANTIC_SEGMENTATION_URL = get_var_conditional(
    "SEMANTIC_SEGMENTATION_URL", "https://segment.roboflow.com"
)
OBJECT_DETECTION_URL = get_var_conditional(
    "SEMANTIC_SEGMENTATION_URL", "https://detect.roboflow.com"
)

CLIP_FEATURIZE_URL = get_var_conditional("CLIP_FEATURIZE_URL", "CLIP FEATURIZE URL NOT IN ENV")
OCR_URL = get_var_conditional("OCR_URL", "OCR URL NOT IN ENV")

DEMO_KEYS = ["coco-128-sample", "chess-sample-only-api-key"]

TYPE_CLASSICATION = "classification"
TYPE_OBJECT_DETECTION = "object-detection"
TYPE_INSTANCE_SEGMENTATION = "instance-segmentation"
TYPE_SEMANTIC_SEGMENTATION = "semantic-segmentation"

DEFAULT_BATCH_NAME = "Pip Package Upload"

RF_WORKSPACES = get_var_conditional("workspaces", default={})
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