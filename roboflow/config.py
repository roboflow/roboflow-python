import json
import os
import sys
import warnings

URL_DEFAULTS = {
    "API_URL": "https://api.roboflow.com",
    "APP_URL": "https://app.roboflow.com",
    "UNIVERSE_URL": "https://universe.roboflow.com",
    "INSTANCE_SEGMENTATION_URL": "https://serverless.roboflow.com",
    "SEMANTIC_SEGMENTATION_URL": "https://segment.roboflow.com",
    "OBJECT_DETECTION_URL": "https://serverless.roboflow.com",
    "SERVERLESS_URL": "https://serverless.roboflow.com",
    "CLIP_FEATURIZE_URL": "CLIP FEATURIZE URL NOT IN ENV",
    "OCR_URL": "OCR URL NOT IN ENV",
    "DEDICATED_DEPLOYMENT_URL": "https://roboflow.cloud",
}

SUPPORTED_REGIONS = ("us", "eu")
SUPPORTED_ENVIRONMENTS = ("prod", "staging")
DEFAULT_REGION = "us"
DEFAULT_ENVIRONMENT = "prod"

# Per (region, environment) overrides of URL_DEFAULTS, which is US production. Hosts
# mirror the platform's environment configs and the dedicated-deployment ingresses.
SERVICE_URL_OVERRIDES: dict[tuple[str, str], dict[str, str]] = {
    ("us", "prod"): {},
    ("us", "staging"): {
        "API_URL": "https://api.roboflow.one",
        "APP_URL": "https://app.roboflow.one",
        "UNIVERSE_URL": "https://universe.roboflow.one",
        "OBJECT_DETECTION_URL": "https://serverless.roboflow.one",
        "INSTANCE_SEGMENTATION_URL": "https://serverless.roboflow.one",
        "SERVERLESS_URL": "https://serverless.roboflow.one",
        "SEMANTIC_SEGMENTATION_URL": "https://lambda-semantic-segmentation.staging.roboflow.com",
        "DEDICATED_DEPLOYMENT_URL": "https://staging.roboflow.cloud",
    },
    ("eu", "prod"): {
        "API_URL": "https://api.roboflow.eu",
        "APP_URL": "https://app.roboflow.eu",
        "OBJECT_DETECTION_URL": "https://serverless.roboflow.eu",
        "INSTANCE_SEGMENTATION_URL": "https://serverless.roboflow.eu",
        "SERVERLESS_URL": "https://serverless.roboflow.eu",
        "DEDICATED_DEPLOYMENT_URL": "https://eu.roboflow.cloud",
    },
    ("eu", "staging"): {
        "API_URL": "https://api.roboflow-eu.one",
        "APP_URL": "https://app.roboflow-eu.one",
        "UNIVERSE_URL": "https://universe.roboflow.one",
        "OBJECT_DETECTION_URL": "https://serverless.roboflow-eu.one",
        "INSTANCE_SEGMENTATION_URL": "https://serverless.roboflow-eu.one",
        "SERVERLESS_URL": "https://serverless.roboflow-eu.one",
        "DEDICATED_DEPLOYMENT_URL": "https://eu.staging.roboflow.cloud",
    },
}

# Hosted services with no deployment in a (region, environment). Their fallback would
# send data to another region or environment, so callers refuse unless the URL is
# set explicitly.
UNAVAILABLE_URL_KEYS: dict[tuple[str, str], tuple[str, ...]] = {
    ("eu", "prod"): ("SEMANTIC_SEGMENTATION_URL",),
    ("eu", "staging"): ("SEMANTIC_SEGMENTATION_URL",),
}

_UNSET = object()


class RegionWarning(UserWarning):
    """Emitted when ROBOFLOW_REGION or ROBOFLOW_ENVIRONMENT holds an unrecognized value."""


_WARNED_UNKNOWN_VALUES: set[tuple[str, str]] = set()


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


def _normalize_choice(setting: str, value, supported: tuple[str, ...], default: str) -> str:
    normalized = value.strip().lower() if isinstance(value, str) else ""
    if normalized in supported:
        return normalized

    warning_key = (setting, repr(value))
    if warning_key not in _WARNED_UNKNOWN_VALUES:
        _WARNED_UNKNOWN_VALUES.add(warning_key)
        # This runs while roboflow is imported, before the CLI parses its flags, so
        # mirror the CLI's --json detection: its stderr must stay machine-readable,
        # and `auth status` reports the problem as a JSON field instead.
        if not ("--json" in sys.argv or "-j" in sys.argv):
            warnings.warn(_unknown_value_message(setting, value, default), RegionWarning, stacklevel=3)
    return default


def _normalize_region(region) -> str:
    return _normalize_choice("region", region, SUPPORTED_REGIONS, DEFAULT_REGION)


def _normalize_environment(environment) -> str:
    return _normalize_choice("environment", environment, SUPPORTED_ENVIRONMENTS, DEFAULT_ENVIRONMENT)


def _unknown_value_message(setting: str, value, default: str) -> str:
    return f"unknown Roboflow {setting} {value!r}; falling back to {default!r}."


def unknown_region_message(region) -> str:
    return _unknown_value_message("region", region, DEFAULT_REGION)


def get_region_warning() -> str | None:
    """Return the fallback warning(s) when the configured region or environment is not recognized."""
    messages = []
    for setting, key, supported, default in (
        ("region", "ROBOFLOW_REGION", SUPPORTED_REGIONS, DEFAULT_REGION),
        ("environment", "ROBOFLOW_ENVIRONMENT", SUPPORTED_ENVIRONMENTS, DEFAULT_ENVIRONMENT),
    ):
        value = get_conditional_configuration_variable(key, default=default)
        normalized = value.strip().lower() if isinstance(value, str) else ""
        if normalized not in supported:
            messages.append(_unknown_value_message(setting, value, default))
    return " ".join(messages) or None


def get_effective_region() -> str:
    """Return the configured Roboflow region, defaulting safely to US."""
    region = get_conditional_configuration_variable("ROBOFLOW_REGION", default=DEFAULT_REGION)
    return _normalize_region(region)


def get_effective_environment() -> str:
    """Return the configured Roboflow environment, defaulting safely to production."""
    environment = get_conditional_configuration_variable("ROBOFLOW_ENVIRONMENT", default=DEFAULT_ENVIRONMENT)
    return _normalize_environment(environment)


def resolve_url(key: str, region: str | None = None, environment: str | None = None) -> str:
    """Resolve a Roboflow URL: explicit override, then region/environment default, then US production."""
    if key not in URL_DEFAULTS:
        raise KeyError(f"Unknown Roboflow URL configuration key: {key}")

    explicit_url = get_conditional_configuration_variable(key, default=_UNSET)
    if explicit_url is not _UNSET:
        return explicit_url

    effective_region = get_effective_region() if region is None else _normalize_region(region)
    effective_environment = get_effective_environment() if environment is None else _normalize_environment(environment)
    return SERVICE_URL_OVERRIDES[(effective_region, effective_environment)].get(key, URL_DEFAULTS[key])


def ensure_url_available_in_region(key: str) -> None:
    """Raise if ``key`` has no deployment in the effective region/environment and no explicit override."""
    target = (get_effective_region(), get_effective_environment())
    if key not in UNAVAILABLE_URL_KEYS.get(target, ()):
        return
    if get_conditional_configuration_variable(key, default=_UNSET) is not _UNSET:
        return
    label = f"{target[0].upper()} {target[1]}"
    raise RuntimeError(
        f"{key} has no Roboflow {label} deployment; the default {resolve_url(key)} would send "
        f"data outside {label}. Set {key} explicitly to override."
    )


def resolve_available_url(key: str) -> str:
    """Resolve ``key`` for a request, refusing when the service is not deployed in the effective region."""
    ensure_url_available_in_region(key)
    return resolve_url(key)


def region_conflict(region: str) -> str | None:
    """Return the ROBOFLOW_REGION environment value when it would override ``region``, else None.

    The environment variable wins over every saved or explicit choice when URLs are
    resolved, so authenticating against ``region`` would pair its credentials with
    requests sent to another platform.
    """
    value = os.getenv("ROBOFLOW_REGION")
    if value is None or _normalize_region(value) == region:
        return None
    return value


CREDENTIALS_REGION_KEY = "ROBOFLOW_CREDENTIALS_REGION"


def has_credentials(config) -> bool:
    """Whether a loaded config holds workspace credentials, not just preferences such as a region."""
    return isinstance(config, dict) and bool(config.get("workspaces"))


def credentials_region(config) -> str:
    """Region whose platform issued the stored credentials.

    Recorded at login and independent of ROBOFLOW_REGION, which ``auth set-region``
    changes without touching credentials. Configs written before the key existed can
    only hold US credentials.
    """
    stored = config.get(CREDENTIALS_REGION_KEY) if isinstance(config, dict) else None
    normalized = stored.strip().lower() if isinstance(stored, str) else ""
    return normalized if normalized in SUPPORTED_REGIONS else DEFAULT_REGION


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
SERVERLESS_URL = resolve_url("SERVERLESS_URL")

CLIP_FEATURIZE_URL = resolve_url("CLIP_FEATURIZE_URL")
OCR_URL = resolve_url("OCR_URL")

DEDICATED_DEPLOYMENT_URL = resolve_url("DEDICATED_DEPLOYMENT_URL")


def refresh_region_urls() -> None:
    """Re-resolve the URL constants after the region changes in a running process.

    Modules across the package bind these constants by value at import time
    (``from roboflow.config import API_URL``), so a login that switches region
    would otherwise keep talking to the previous platform until restart. Only
    bindings still holding the previous default are replaced, so values a caller
    patched deliberately are left alone.
    """
    module_globals = globals()
    previous_urls = {key: module_globals[key] for key in URL_DEFAULTS}
    current_urls = {key: resolve_url(key) for key in URL_DEFAULTS}
    for module_name, module in list(sys.modules.items()):
        if module is None or not (module_name == "roboflow" or module_name.startswith("roboflow.")):
            continue
        namespace = vars(module)
        for key, previous_url in previous_urls.items():
            if namespace.get(key, _UNSET) == previous_url:
                namespace[key] = current_urls[key]


DEMO_KEYS = ["coco-128-sample", "chess-sample-only-api-key"]

TYPE_CLASSICATION = "classification"
TYPE_OBJECT_DETECTION = "object-detection"
TYPE_INSTANCE_SEGMENTATION = "instance-segmentation"
TYPE_SEMANTIC_SEGMENTATION = "semantic-segmentation"
TYPE_KEYPOINT_DETECTION = "keypoint-detection"
TYPE_TEXT_IMAGE_PAIRS = "text-image-pairs"
TYPE_ACTION_RECOGNITION = "action-recognition"

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
