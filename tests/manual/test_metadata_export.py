"""
Test user_metadata export in roboflow-python SDK.

Related Linear Issues:
- DATAMAN-98: roboflow-python should export user_metadata on code
- DATAMAN-99: roboflow-python version export with user_metadata

This test validates that user_metadata is properly exported when downloading
dataset versions via the SDK.

Uses staging project: model-evaluation-workspace/donut-2-lcfx0/28
This project has tags and metadata on its images.

FINDINGS (2026-01-30):
=====================
1. COCO format export: ✅ WORKS
   - user_metadata is included in image.extra.user_metadata
   - Example: {"id": 0, ..., "extra": {"name": "...", "user_metadata": {"yummy": 0}}}

2. YOLOv8 format export:
   - Standard YOLO format (images + txt labels)
   - No dedicated metadata file (expected - YOLO format doesn't have metadata concept)

3. SDK Changes Needed: NONE
   - version.download() downloads ZIP from server
   - Server (roboflow-zip) already includes user_metadata in COCO JSON
   - SDK extracts ZIP locally
   - user_metadata is available in the downloaded files
"""
import os
import sys
import json
import glob

thisdir = os.path.dirname(os.path.abspath(__file__))
os.environ["ROBOFLOW_CONFIG_DIR"] = f"{thisdir}/data/.config"
# Use staging API
os.environ["API_URL"] = "https://api.roboflow.one"

rootdir = os.path.abspath(f"{thisdir}/../..")
sys.path.append(rootdir)

from roboflow import Roboflow


def _get_manual_api_key():
    api_key = os.getenv("ROBOFLOW_STAGING_API_KEY") or os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("Set ROBOFLOW_STAGING_API_KEY (or ROBOFLOW_API_KEY) to run this manual test")
    return api_key


def test_version_export_metadata():
    """
    Test that user_metadata is exported with version download.
    """
    rf = Roboflow(api_key=_get_manual_api_key())

    # Access the staging project with metadata
    project = rf.workspace("model-evaluation-workspace").project("donut-2-lcfx0")
    version = project.version(28)

    # Download in COCO format (metadata is in the JSON)
    print("Downloading version in COCO format...")
    dataset = version.download("coco", location=f"{thisdir}/metadata_test_coco", overwrite=True)
    print(f"\nDataset downloaded to: {dataset.location}")

    # Look for annotation files
    json_files = glob.glob(f"{dataset.location}/**/*.json", recursive=True)
    print(f"\nFound JSON files: {json_files}")

    # Check each JSON file for user_metadata in the extra field
    has_metadata = False
    for json_file in json_files:
        print(f"\n--- Inspecting: {json_file} ---")
        with open(json_file, 'r') as f:
            data = json.load(f)

        # COCO format has 'images' array
        if 'images' in data:
            print(f"Found {len(data['images'])} images in COCO format")
            for i, img in enumerate(data['images'][:3]):  # Check first 3
                print(f"\nImage {i}: {img.get('file_name', 'unknown')}")
                extra = img.get('extra', {})
                user_metadata = extra.get('user_metadata')
                if user_metadata:
                    print(f"  ✅ user_metadata (in extra): {user_metadata}")
                    has_metadata = True
                else:
                    print(f"  ❌ No user_metadata in extra field")
                    print(f"  Keys in image: {list(img.keys())}")
                    print(f"  Keys in extra: {list(extra.keys()) if extra else 'N/A'}")
        else:
            print(f"Keys in JSON: {list(data.keys())[:10]}")

    print("\n" + "=" * 60)
    if has_metadata:
        print("✅ TEST PASSED: user_metadata is exported in COCO format")
        print("   Location: image['extra']['user_metadata']")
    else:
        print("❌ TEST FAILED: user_metadata not found in exported files")
    print("=" * 60)


if __name__ == "__main__":
    test_version_export_metadata()
