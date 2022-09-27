import os
import urllib

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


def validate_image_path(image_path):
    """
    Validate whether a local OR hosted image path is valid
    :param image_path: local path or URL of image
    :returns: None
    :raises Exception: Image path is not valid
    """
    if not check_image_path(image_path):
        raise Exception(f"Image does not exist at {image_path}!")
