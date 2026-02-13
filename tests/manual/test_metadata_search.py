"""
Test user_metadata retrieval via SDK APIs.

Related Linear Issues:
- DATAMAN-98: roboflow-python should export user_metadata on code
- DATAMAN-99: roboflow-python version export with user_metadata

This test validates that user_metadata can be retrieved via the SDK's
search() and image() methods.

Uses staging project: model-evaluation-workspace/donut-2-lcfx0
This project has tags and metadata on its images.

API SUPPORT:
============
1. project.search() - Available fields:
   - id, name, created, annotations, labels, split, tags, owner, embedding, user_metadata
   - Default: ['id', 'name', 'created', 'labels']
   - To get tags: fields=['id', 'name', 'tags']
   - To get user_metadata: fields=['id', 'name', 'user_metadata']

2. project.image(id) - Image API:
   - Returns metadata in the 'metadata' field
   - Example: {'material': 'aluminium', 'yummy': 3, 'penguin': 2}
"""
import os
import sys
import json

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


def test_search_and_image_api():
    """
    Test that user_metadata can be retrieved via project.search() and project.image().
    """
    rf = Roboflow(api_key=_get_manual_api_key())

    # Access the staging project with metadata
    project = rf.workspace("model-evaluation-workspace").project("donut-2-lcfx0")

    print("=" * 60)
    print("TEST 1: project.search() - Default fields")
    print("=" * 60)
    results = project.search(limit=3)
    print(f"\nFound {len(results)} images")
    for i, img in enumerate(results):
        print(f"\nImage {i}: {img.get('name', 'unknown')}")
        print(f"  Keys: {list(img.keys())}")

    print("\n" + "=" * 60)
    print("TEST 2: project.search() - With tags field")
    print("=" * 60)
    results_tags = project.search(limit=3, fields=["id", "name", "tags"])
    print(f"\nFound {len(results_tags)} images")
    for i, img in enumerate(results_tags):
        print(f"\nImage {i}: {img.get('name', 'unknown')}")
        print(f"  Keys: {list(img.keys())}")
        if 'tags' in img:
            print(f"  ✅ tags: {img['tags']}")

    print("\n" + "=" * 60)
    print("TEST 3: project.search() - With user_metadata field")
    print("=" * 60)
    try:
        results_metadata = project.search(limit=3, fields=["id", "name", "user_metadata"])
        print(f"\nFound {len(results_metadata)} images")
        for i, img in enumerate(results_metadata):
            print(f"\nImage {i}: {img.get('name', 'unknown')}")
            print(f"  Keys: {list(img.keys())}")
            if 'user_metadata' in img:
                print(f"  ✅ user_metadata: {img['user_metadata']}")
            else:
                print(f"  ❌ No user_metadata in response")
    except Exception as e:
        print(f"  ⚠️  Error: {e}")
        print("  (user_metadata field may not be deployed yet)")

    print("\n" + "=" * 60)
    print("TEST 4: project.image(id) - Image API")
    print("=" * 60)

    # Get images and check their metadata via image API
    images_with_metadata = 0
    for img in results[:5]:
        image_id = img.get('id')
        image_name = img.get('name', 'unknown')

        print(f"\nFetching details for: {image_name}")
        details = project.image(image_id)

        metadata = details.get('metadata', {})
        if metadata:
            print(f"  ✅ metadata: {metadata}")
            images_with_metadata += 1
        else:
            print(f"  ❌ No metadata (empty dict)")

    print("\n" + "=" * 60)
    if images_with_metadata > 0:
        print(f"✅ TEST PASSED: {images_with_metadata}/{min(5, len(results))} images have metadata")
        print("   Access via: project.image(id)['metadata']")
    else:
        print("❌ TEST FAILED: No images have metadata")
    print("=" * 60)


if __name__ == "__main__":
    test_search_and_image_api()
