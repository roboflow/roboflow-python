from roboflow.config import TYPE_OBJECT_DETECTION
from roboflow.core.version import Version


def get_version(api_key="test-api-key", project_name="Test Project Name", version_number="1", type=TYPE_OBJECT_DETECTION, workspace_name="Test Workspace Name", **kwargs):
    version_data = {
        "id": f"test-workspace/test-project/2",
        "name": "augmented-416x416",
        "created": 1663104679.539,
        "images": 240,
        "splits": {"train": 210, "test": 10, "valid": 20},
        "preprocessing": {"resize": {"height": "416", "enabled": True, "width": "416", "format": "Stretch to"}, "auto-orient": {"enabled": True}},
        "augmentation": {"blur": {"enabled": True, "pixels": 1.5}, "image": {"enabled": True, "versions": 3}, "rotate": {"degrees": 15, "enabled": True}, "crop": {"enabled": True, "percent": 40, "min": 0}, "flip": {"horizontal": True, "enabled": True, "vertical": False}},
        "exports": []
    }
    version_data.update(kwargs)

    return Version(
                version_data,
                type,
                api_key,
                project_name,
                version_number,
                model_format=None,
                local=None,
                workspace=workspace_name,
                project_name=project_name
            )
