import json
import os


def get_conditional_configuration_variable(key, default):
    """Retrieves the configuration variable conditionally.
        ##1. check if variable is in environment
        ##2. check if variable is in config file
        ##3. return default value
    Args:
        key (string): The name of the configuration variable.
        default (string): The default value of the configuration variable.
    Returns:
        string: The value of the conditional configuration variable.
    """  # noqa: E501 // docs

    os_name = os.name

    if os_name == "nt":
        default_path = os.path.join(os.getenv("USERPROFILE"), "roboflow/config.json")
    else:
        default_path = os.path.join(os.getenv("HOME"), ".config/roboflow/config.json")

    # default configuration location
    conf_location = os.getenv(
        "ROBOFLOW_CONFIG_DIR",
        default=default_path,
    )

    # read config file for roboflow if logged in from python or CLI
    if os.path.exists(conf_location):
        with open(conf_location) as f:
            config = json.load(f)
    else:
        config = {}

    if os.getenv(key) is not None:
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

API_URL = get_conditional_configuration_variable("API_URL", "https://api.roboflow.com")
APP_URL = get_conditional_configuration_variable("APP_URL", "https://app.roboflow.com")
UNIVERSE_URL = get_conditional_configuration_variable("UNIVERSE_URL", "https://universe.roboflow.com")

INSTANCE_SEGMENTATION_URL = get_conditional_configuration_variable(
    "INSTANCE_SEGMENTATION_URL", "https://outline.roboflow.com"
)
SEMANTIC_SEGMENTATION_URL = get_conditional_configuration_variable(
    "SEMANTIC_SEGMENTATION_URL", "https://segment.roboflow.com"
)
OBJECT_DETECTION_URL = get_conditional_configuration_variable("OBJECT_DETECTION_URL", "https://detect.roboflow.com")

CLIP_FEATURIZE_URL = get_conditional_configuration_variable("CLIP_FEATURIZE_URL", "CLIP FEATURIZE URL NOT IN ENV")
OCR_URL = get_conditional_configuration_variable("OCR_URL", "OCR URL NOT IN ENV")

DEDICATED_DEPLOYMENT_URL = get_conditional_configuration_variable("DEDICATED_DEPLOYMENT_URL", "https://roboflow.cloud")

DEMO_KEYS = ["coco-128-sample", "chess-sample-only-api-key"]

TYPE_CLASSICATION = "classification"
TYPE_OBJECT_DETECTION = "object-detection"
TYPE_INSTANCE_SEGMENTATION = "instance-segmentation"
TYPE_SEMANTIC_SEGMENTATION = "semantic-segmentation"
TYPE_KEYPOINT_DETECTION = "keypoint-detection"

DEFAULT_BATCH_NAME = "Pip Package Upload"
DEFAULT_JOB_NAME = "Annotated via API"

RF_WORKSPACES = get_conditional_configuration_variable("workspaces", default={})
TQDM_DISABLE = os.getenv("TQDM_DISABLE", None)


def load_roboflow_api_key(workspace_url=None):
    if os.getenv("ROBOFLOW_API_KEY") is not None:
        return os.getenv("ROBOFLOW_API_KEY")
    RF_WORKSPACES = get_conditional_configuration_variable("workspaces", default={})
    workspaces_by_url = {w["url"]: w for w in RF_WORKSPACES.values()}
    default_workspace_url = get_conditional_configuration_variable("RF_WORKSPACE", default=None)
    default_workspace = workspaces_by_url.get(default_workspace_url, None)
    workspace = workspaces_by_url.get(workspace_url, default_workspace)
    workspace = workspace or get_conditional_configuration_variable("RF_WORKSPACE", default=None)
    if workspace:
        return workspace.get("apiKey", None)
