import json
import os
from typing import Any, Dict, List, Optional

import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder

from roboflow.adapters.rfapi import RoboflowError
from roboflow.config import API_URL

_BASE = f"{API_URL}/vision-events"


def _auth_headers(api_key: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def write_event(api_key: str, event: Dict[str, Any]) -> dict:
    """Create a single vision event.

    Args:
        api_key: Roboflow API key.
        event: Event payload dict (eventId, eventType, useCaseId, timestamp, etc.).

    Returns:
        Parsed JSON response with ``eventId`` and ``created``.

    Raises:
        RoboflowError: On non-201 response status codes.
    """
    response = requests.post(_BASE, json=event, headers=_auth_headers(api_key))
    if response.status_code != 201:
        raise RoboflowError(response.text)
    return response.json()


def write_batch(api_key: str, events: List[Dict[str, Any]]) -> dict:
    """Create multiple vision events in a single request.

    Args:
        api_key: Roboflow API key.
        events: List of event payload dicts (max 100 per the server).

    Returns:
        Parsed JSON response with ``created`` count and ``eventIds``.

    Raises:
        RoboflowError: On non-201 response status codes.
    """
    response = requests.post(
        f"{_BASE}/batch",
        json={"events": events},
        headers=_auth_headers(api_key),
    )
    if response.status_code != 201:
        raise RoboflowError(response.text)
    return response.json()


def query(api_key: str, query_params: Dict[str, Any]) -> dict:
    """Query vision events with filters and pagination.

    Args:
        api_key: Roboflow API key.
        query_params: Query payload (useCaseId, eventType, startTime, endTime,
            cursor, limit, customMetadataFilters, etc.).

    Returns:
        Parsed JSON response with ``events``, ``nextCursor``, ``hasMore``,
        and ``lookbackDays``.

    Raises:
        RoboflowError: On non-200 response status codes.
    """
    response = requests.post(
        f"{_BASE}/query",
        json=query_params,
        headers=_auth_headers(api_key),
    )
    if response.status_code != 200:
        raise RoboflowError(response.text)
    return response.json()


def list_use_cases(api_key: str, status: Optional[str] = None) -> dict:
    """List all use cases for a workspace.

    Args:
        api_key: Roboflow API key.
        status: Optional status filter (default server-side: "active").

    Returns:
        Parsed JSON response with ``useCases`` list and ``lookbackDays``.

    Raises:
        RoboflowError: On non-200 response status codes.
    """
    params: Dict[str, str] = {}
    if status is not None:
        params["status"] = status
    response = requests.get(
        f"{_BASE}/use-cases",
        params=params,
        headers=_auth_headers(api_key),
    )
    if response.status_code != 200:
        raise RoboflowError(response.text)
    return response.json()


def get_custom_metadata_schema(api_key: str, use_case_id: str) -> dict:
    """Get the custom metadata schema for a use case.

    Args:
        api_key: Roboflow API key.
        use_case_id: Use case identifier.

    Returns:
        Parsed JSON response with ``fields`` mapping field names to their types.

    Raises:
        RoboflowError: On non-200 response status codes.
    """
    response = requests.get(
        f"{_BASE}/custom-metadata-schema/{use_case_id}",
        headers=_auth_headers(api_key),
    )
    if response.status_code != 200:
        raise RoboflowError(response.text)
    return response.json()


def create_use_case(api_key: str, name: str) -> dict:
    """Create a new vision event use case.

    Args:
        api_key: Roboflow API key.
        name: Human-readable name for the use case.

    Returns:
        Parsed JSON response with ``id`` and ``name``.

    Raises:
        RoboflowError: On non-201 response status codes.
    """
    response = requests.post(
        f"{_BASE}/use-cases",
        json={"name": name},
        headers=_auth_headers(api_key),
    )
    if response.status_code != 201:
        raise RoboflowError(response.text)
    return response.json()


def rename_use_case(api_key: str, use_case_id: str, name: str) -> dict:
    """Rename an existing vision event use case.

    Args:
        api_key: Roboflow API key.
        use_case_id: Use case identifier.
        name: New name for the use case.

    Returns:
        Parsed JSON response with ``id`` and ``name``.

    Raises:
        RoboflowError: On non-200 response status codes.
    """
    response = requests.put(
        f"{_BASE}/use-cases/{use_case_id}",
        json={"name": name},
        headers=_auth_headers(api_key),
    )
    if response.status_code != 200:
        raise RoboflowError(response.text)
    return response.json()


def archive_use_case(api_key: str, use_case_id: str) -> dict:
    """Archive a vision event use case.

    Args:
        api_key: Roboflow API key.
        use_case_id: Use case identifier.

    Returns:
        Parsed JSON response with ``success``.

    Raises:
        RoboflowError: On non-200 response status codes.
    """
    response = requests.post(
        f"{_BASE}/use-cases/{use_case_id}/archive",
        headers=_auth_headers(api_key),
    )
    if response.status_code != 200:
        raise RoboflowError(response.text)
    return response.json()


def unarchive_use_case(api_key: str, use_case_id: str) -> dict:
    """Unarchive a vision event use case.

    Args:
        api_key: Roboflow API key.
        use_case_id: Use case identifier.

    Returns:
        Parsed JSON response with ``success``.

    Raises:
        RoboflowError: On non-200 response status codes.
    """
    response = requests.post(
        f"{_BASE}/use-cases/{use_case_id}/unarchive",
        headers=_auth_headers(api_key),
    )
    if response.status_code != 200:
        raise RoboflowError(response.text)
    return response.json()


def upload_image(
    api_key: str,
    image_path: str,
    name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> dict:
    """Upload an image for use in vision events.

    Args:
        api_key: Roboflow API key.
        image_path: Local filesystem path to the image file.
        name: Optional custom image name.
        metadata: Optional flat dict of metadata to attach.

    Returns:
        Parsed JSON response with ``sourceId`` (and optionally ``url``).

    Raises:
        RoboflowError: On non-201 response status codes.
    """
    filename = name or os.path.basename(image_path)
    with open(image_path, "rb") as f:
        fields: Dict[str, Any] = {
            "file": (filename, f, "application/octet-stream"),
        }
        if name is not None:
            fields["name"] = name
        if metadata is not None:
            fields["metadata"] = json.dumps(metadata)
        m = MultipartEncoder(fields=fields)
        headers = _auth_headers(api_key)
        headers["Content-Type"] = m.content_type
        response = requests.post(f"{_BASE}/upload", data=m, headers=headers)

    if response.status_code != 201:
        raise RoboflowError(response.text)
    return response.json()
