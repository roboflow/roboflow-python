import json
import os
import sys

REGION_URL_DEFAULTS = {
    "us": {},
    "eu": {
        "API_URL": "https://api.roboflow.eu",
        "APP_URL": "https://app.roboflow.eu",
        "OBJECT_DETECTION_URL": "https://serverless.roboflow.eu",
        "INSTANCE_SEGMENTATION_URL": "https://serverless.roboflow.eu",
        "DEDICATED_DEPLOYMENT_URL": "https://eu.roboflow.cloud",
    },
}

URL_DEFAULTS = {
    "API_URL": "https://api.roboflow.com",
    "APP_URL": "https://app.roboflow.com",
    "UNIVERSE_URL": "https://universe.roboflow.com",
    "INSTANCE_SEGMENTATION_URL": "https://serverless.roboflow.com",
    "SEMANTIC_SEGMENTATION_URL": "https://segment.roboflow.com",
    "OBJECT_DETECTION_URL": "https://serverless.roboflow.com",
    "CLIP_FEATURIZE_URL": "CLIP FEATURIZE URL NOT IN ENV",
    "OCR_URL": "OCR URL NOT IN ENV",
    "DEDICATED_DEPLOYMENT_URL": "https://roboflow.cloud",
}

_UNSET = object()
_WARNED_UNKNOWN_REGIONS: set[str] = set()


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


def _normalize_region(region) -> str:
    normalized_region = region.strip().lower() if isinstance(region, str) else ""
    if normalized_region in REGION_URL_DEFAULTS:
        return normalized_region

    warning_key = repr(region)
    if warning_key not in _WARNED_UNKNOWN_REGIONS:
        print(
            f"Warning: unknown Roboflow region {region!r}; falling back to 'us'.",
            file=sys.stderr,
        )
        _WARNED_UNKNOWN_REGIONS.add(warning_key)
    return "us"


def get_effective_region() -> str:
    """Return the configured Roboflow region, defaulting safely to US."""
    region = get_conditional_configuration_variable("ROBOFLOW_REGION", default="us")
    return _normalize_region(region)


def resolve_url(key: str, region: str | None = None) -> str:
    """Resolve a Roboflow URL using explicit overrides before region defaults."""
    if key not in URL_DEFAULTS:
        raise KeyError(f"Unknown Roboflow URL configuration key: {key}")

    explicit_url = get_conditional_configuration_variable(key, default=_UNSET)
    if explicit_url is not _UNSET:
        return explicit_url

    effective_region = get_effective_region() if region is None else _normalize_region(region)
    return REGION_URL_DEFAULTS[effective_region].get(key, URL_DEFAULTS[key])


CLASSIFICATION_MODEL = os.getenv("CLASSIFICATION_MODEL", "ClassificationModel")
INSTANCE_SEGMENTATION_MODEL = "InstanceSegmentationModel"
KEYPOINT_DETECTION_MODEL = "KeypointDetectionModel"
OBJECT_DETECTION_MODEL = os.getenv("OBJECT_DETECTION_MODEL", "ObjectDetectionModel")
SEMANTIC_SEGMENTATION_MODEL = "SemanticSegmentationModel"
PREDICTION_OBJECT = os.getenv("PREDICTION_OBJECT", "Prediction")

API_URL = resolve_url("API_URL")
APP_URL = resolve_url("APP_URL")
UNIVERSE_URL = resolve_url("UNIVERSE_URL")

INSTANCE_SEGMENTATION_URL = resolve_url("INSTANCE_SEGMENTATION_URL")
SEMANTIC_SEGMENTATION_URL = resolve_url("SEMANTIC_SEGMENTATION_URL")
OBJECT_DETECTION_URL = resolve_url("OBJECT_DETECTION_URL")

CLIP_FEATURIZE_URL = resolve_url("CLIP_FEATURIZE_URL")
OCR_URL = resolve_url("OCR_URL")

DEDICATED_DEPLOYMENT_URL = resolve_url("DEDICATED_DEPLOYMENT_URL")

DEMO_KEYS = ["coco-128-sample", "chess-sample-only-api-key"]

TYPE_CLASSICATION = "classification"
TYPE_OBJECT_DETECTION = "object-detection"
TYPE_INSTANCE_SEGMENTATION = "instance-segmentation"
TYPE_SEMANTIC_SEGMENTATION = "semantic-segmentation"
TYPE_KEYPOINT_DETECTION = "keypoint-detection"
TYPE_TEXT_IMAGE_PAIRS = "text-image-pairs"

TASK_DET = "det"
TASK_SEG = "seg"
TASK_SEM = "sem"
TASK_POSE = "pose"
TASK_CLS = "cls"
TASK_OBB = "obb"

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
