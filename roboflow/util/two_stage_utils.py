import base64
import io

import requests

from roboflow.config import OCR_URL


def ocr_infer(image):
    # Convert to JPEG Buffer
    buffered = io.BytesIO()
    image.save(buffered, quality=90, format="PNG")

    # Base 64 Encode
    img_str = base64.b64encode(buffered.getvalue())
    img_str = img_str.decode("ascii")

    # Construct the URL
    upload_url = "".join([OCR_URL])

    # POST to the API
    r = requests.post(
        upload_url,
        data=img_str,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    # Output result
    return r.json()
