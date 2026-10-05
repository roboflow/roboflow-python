"""Shared helpers for the project and workspace search surfaces."""

from typing import List, Optional

# The search API accepts exactly these media types. Mixed selection is `["image", "video"]`.
VALID_MEDIA_TYPES = ("image", "video")

_VALID_LIST = ", ".join(repr(t) for t in VALID_MEDIA_TYPES)


def normalize_media_types(media_types: Optional[List[str]]) -> Optional[List[str]]:
    """Validate and normalize a ``media_types`` search selection.

    Returns ``None`` for an omitted selection so callers leave ``mediaTypes`` off the
    request body entirely and inherit the API default of images only.

    Args:
        media_types: List of media types to search, e.g. ``["video"]`` or
            ``["image", "video"]``. ``None`` means "use the API default".

    Returns:
        A lowercased, de-duplicated list preserving the caller's order, or ``None``.

    Raises:
        ValueError: If the selection is not a non-empty list of valid media types.
    """
    if media_types is None:
        return None

    # A bare string is the most common mistake, so name the fix instead of iterating characters.
    if isinstance(media_types, str):
        raise ValueError(f"media_types must be a list, not a string - use media_types=[{media_types!r}]")

    if not isinstance(media_types, (list, tuple)) or len(media_types) == 0:
        raise ValueError(f"media_types must be a non-empty list containing any of: {_VALID_LIST}")

    normalized: List[str] = []
    for media_type in media_types:
        if not isinstance(media_type, str):
            raise ValueError(f"media_types entries must be strings, got {type(media_type).__name__!r}")
        lowered = media_type.lower()
        if lowered not in VALID_MEDIA_TYPES:
            raise ValueError(f"invalid media type {media_type!r} - media_types must only contain: {_VALID_LIST}")
        if lowered not in normalized:
            normalized.append(lowered)

    return normalized


def parse_media_types_option(raw: Optional[str]) -> Optional[List[str]]:
    """Parse a comma-separated CLI ``--media-types`` value into a normalized list.

    Args:
        raw: Raw option value, e.g. ``"video"`` or ``"image,video"``. ``None`` or an
            empty string means "use the API default".

    Returns:
        A normalized media type list, or ``None`` when nothing was requested.

    Raises:
        ValueError: If the value contains anything other than valid media types.
    """
    if raw is None:
        return None
    entries = [entry.strip() for entry in raw.split(",") if entry.strip()]
    if not entries:
        raise ValueError(f"--media-types must name at least one of: {_VALID_LIST}")
    return normalize_media_types(entries)
