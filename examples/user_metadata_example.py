"""
Roboflow User Metadata Example
==============================

This example demonstrates how to access user_metadata (custom metadata fields)
attached to images in your Roboflow projects.

User metadata allows you to store custom key-value pairs on images, such as:
- capture_location: "warehouse-A"
- camera_id: "cam-001"
- quality_score: 0.95
- is_validated: True

There are three ways to access user_metadata:

1. Search API - Query images and retrieve metadata in bulk
2. Image API - Get metadata for a specific image by ID
3. Version Export - Download datasets with metadata included in annotation files

Requirements:
    pip install roboflow

Usage:
    python user_metadata_example.py
"""

import os
import json
import glob


from roboflow import Roboflow


# =============================================================================
# Configuration
# =============================================================================
# change as needed
API_KEY = os.getenv("ROBOFLOW_API_KEY")
WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE")
PROJECT = os.getenv("ROBOFLOW_PROJECT")
VERSION = os.getenv("ROBOFLOW_VERSION")


def example_search_with_metadata(project):
    """
    Example 1: Search API with user_metadata

    Use the `fields` parameter to request user_metadata in search results.
    This is efficient for retrieving metadata for multiple images at once.

    Available fields: id, name, created, annotations, labels, split, tags,
                      owner, embedding, user_metadata
    """
    print("=" * 70)
    print("Example 1: Search API with user_metadata")
    print("=" * 70)

    # Search for images and include user_metadata in results
    results = project.search(limit=5, fields=["id", "name", "tags", "user_metadata"])

    print(f"\nFound {len(results)} images\n")

    for img in results:
        print(f"Image: {img['name']}")
        print(f"  ID: {img['id']}")
        print(f"  Tags: {img.get('tags', [])}")

        metadata = img.get("user_metadata")
        if metadata:
            print("  User Metadata:")
            for key, value in metadata.items():
                print(f"    - {key}: {value}")
        else:
            print("  User Metadata: (none)")
        print()

    return results


def example_image_by_id(project, image_id):
    """
    Example 2: Image API - Get metadata for a specific image

    Use project.image(id) to retrieve full details including metadata.
    The metadata is returned in the 'metadata' field.
    """
    print("=" * 70)
    print("Example 2: Image API - Get metadata by image ID")
    print("=" * 70)

    # Get image details by ID
    image = project.image(image_id)

    print(f"\nImage: {image['name']}")
    print(f"  ID: {image['id']}")
    print(f"  Split: {image.get('split', 'N/A')}")
    print(f"  Tags: {image.get('tags', [])}")

    metadata = image.get("metadata", {})
    if metadata:
        print("  Metadata:")
        for key, value in metadata.items():
            print(f"    - {key}: {value}")
    else:
        print("  Metadata: (none)")
    print()


def example_version_export(version, export_path="./dataset_export"):
    """
    Example 3: Version Export with user_metadata

    When exporting a dataset version in COCO format, user_metadata is included
    in the annotation JSON file under each image's 'extra' field.

    Location in COCO JSON: images[].extra.user_metadata
    """
    print("=" * 70)
    print("Example 3: Version Export with user_metadata (COCO format)")
    print("=" * 70)

    # Download the dataset in COCO format
    print(f"\nDownloading dataset to: {export_path}")
    dataset = version.download("coco", location=export_path, overwrite=True)

    # Read the annotation file to show user_metadata
    json_files = glob.glob(f"{dataset.location}/**/_annotations.coco.json", recursive=True)

    if json_files:
        print(f"\nInspecting: {json_files[0]}")
        with open(json_files[0], "r") as f:
            coco_data = json.load(f)

        print("\nImages with user_metadata in export:\n")
        for img in coco_data.get("images", [])[:5]:  # Show first 5
            extra = img.get("extra", {})
            user_metadata = extra.get("user_metadata")

            print(f"  {img['file_name']}")
            if user_metadata:
                print(f"    user_metadata: {user_metadata}")
            else:
                print("    user_metadata: (none)")
    print()


def main():
    # =========================================================================
    # Initialize Roboflow
    # =========================================================================
    print("\nInitializing Roboflow...\n")
    rf = Roboflow(api_key=API_KEY)

    project = rf.workspace(WORKSPACE).project(PROJECT)
    version = project.version(VERSION)

    # =========================================================================
    # Example 1: Search with user_metadata
    # =========================================================================
    results = example_search_with_metadata(project)

    # =========================================================================
    # Example 2: Get image by ID
    # =========================================================================
    # Find an image with metadata from search results
    image_with_metadata = next((img for img in results if img.get("user_metadata")), results[0] if results else None)

    if image_with_metadata:
        example_image_by_id(project, image_with_metadata["id"])

    # =========================================================================
    # Example 3: Version export (COCO format)
    # =========================================================================
    example_version_export(version, export_path="./dataset_with_metadata")

    print("=" * 70)
    print("Examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
