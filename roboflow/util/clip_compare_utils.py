import base64
import glob
import io
import json

import requests
from PIL import Image

from roboflow.config import CLIP_FEATURIZE_URL


def base64_encode(image_path):
    """`rf.predict` requires images formatted and base64 encoded
    @params:
        iamge_path: (str) = name reference to a given image for encoding

    returns: a base64 encoded string formatted for travel to an HTTP endpoint
    """

    image = Image.open(image_path)
    buffered = io.BytesIO()
    image_rgb = image.convert("RGB")
    image_rgb.save(buffered, quality=90, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str.decode("ascii")


def clip_encode(image1: str, image2: str) -> float:
    """This function base64 encodes two images and send them to the clip endpoint for further encoding and comparison
    @params

        image1: (str) = name referenceto a given image for encoding
        image2: (str) = name referenceto a given image for encoding

        returns: (float) = a value between 1 and 0, with 1 being images were identical
    """
    image1 = base64_encode(image1)
    image2 = base64_encode(image2)

    headers = {"Content-Type": "text/plain"}
    data = json.dumps({"image1": image1, "image2": image2})

    r = requests.post(CLIP_FEATURIZE_URL, data=data, headers=headers)

    return float(r.json()["similarity"])
