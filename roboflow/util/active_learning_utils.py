import base64
import io
import json

import requests
from PIL import Image


# a for loop that counts the number of occurances within an array
def count_class_occurances(predictions, target_class):
    count = 0

    for prediction in predictions:
        if prediction["class"] in target_class:
            count += 1

    return count


# compares counts and returns False if counts below requirement
def count_comparisons(
    predictions, required_objects_count, required_class_count, target_class
):
    if (
        len(predictions) < required_objects_count
        or target_class
        and count_class_occurances(predictions, target_class) < required_class_count
    ):
        return False
    else:
        return True


# checks box size and returns False if not within requirements
def check_box_size(prediction, minimum_size_requirement, maximum_size_requirement):
    if (
        prediction["height"] * prediction["width"] < maximum_size_requirement
        and prediction["height"] * prediction["width"] > minimum_size_requirement
    ):
        return True
    else:
        return False


# clip_encode requires images to be in a PIL image format, rf.predict handles this and only requires the file location
def base64_encode(image_path):
    image = Image.open(image_path)
    buffered = io.BytesIO()
    image_rgb = image.convert("RGB")
    image_rgb.save(buffered, quality=90, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str.decode("ascii")


def clip_encode(image1, image2, CLIP_FEATURIZE_URL):
    image1 = base64_encode(image1)
    image2 = base64_encode(image2)

    if CLIP_FEATURIZE_URL == "CLIP FEATURIZE URL NOT IN ENV":
        raise Exception(
            "You need to ad CLIP_FEATURE_URL to your env vars. To learn more about this active learning feature, contact Roboflow sales https://roboflow.com/sales. You can remove the similarity keys from your conditionals to use other active learning functionality."
        )

    url = CLIP_FEATURIZE_URL
    headers = {"Content-Type": "text/plain"}
    data = json.dumps({"image1": image1, "image2": image2})

    r = requests.post(url, data=data, headers=headers)

    return float(r.json()["similarity"])
