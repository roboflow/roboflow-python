import base64
import os
import urllib

import cv2
import numpy as np
import requests


def check_image_path(image_path):
    """
    Check whether a local OR hosted image path is valid
    :param image_path: local path or URL of image
    :returns: Boolean
    """
    return os.path.exists(image_path) or check_image_url(image_path)


def check_image_url(url):
    """
    Check whether a hosted image path is valid
    :param url: URL of image
    :returns: Boolean
    """
    if urllib.parse.urlparse(url).scheme not in (
        "http",
        "https",
    ):
        return False

    r = requests.head(url)
    return r.status_code == requests.codes.ok


def mask_image(image, encoded_mask, transparency=60):
    """
    Overlay a translucent mask on top of an image with CV2
    :param image: a CV2 image / numpy.ndarray matrix
    :param encoded_mask: a base64 encoded single channel image
    :param transparency: alpha transparency of masks for semantic overlays
    :returns: CV2 image / numpy.ndarray matrix
    """
    np_data = np.fromstring(base64.b64decode(encoded_mask), np.uint8)
    mask = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)

    # Fallback in case the API returns an incorrectly sized mask
    # This should not happen in practice, but we saw it in testing
    image_size = image.shape[1::-1]
    mask_size = mask.shape[1::-1]
    if mask_size != image_size:
        mask = cv2.resize(mask, image_size)

    # Mask the original image
    masked = cv2.bitwise_and(image, image, mask=mask)

    # Overlay a translucent version of the original image on top of the masked image to give it a translucent effect
    alpha = transparency / 100
    return cv2.addWeighted(masked, alpha, image, 1 - alpha, 0)


def validate_image_path(image_path):
    """
    Validate whether a local OR hosted image path is valid
    :param image_path: local path or URL of image
    :returns: None
    :raises Exception: Image path is not valid
    """
    if not check_image_path(image_path):
        raise Exception(f"Image does not exist at {image_path}!")
