import os
import re

from .image_utils import load_labelmap

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
ANNOTATION_EXTENSIONS = {".txt", ".json", ".xml"}
LABELMAPS_EXTENSIONS = {".labels"}


def parsefolder(folder):
    if not os.path.exists(folder):
        raise Exception(f"folder does not exist. {folder}")
    files = _list_files(folder)
    images = [f for f in files if f["extension"] in IMAGE_EXTENSIONS]
    _decide_split(images)
    annotations = [f for f in files if f["extension"] in ANNOTATION_EXTENSIONS]
    labelmaps = [f for f in files if f["extension"] in LABELMAPS_EXTENSIONS]
    labelmaps = _load_labelmaps(folder, labelmaps)
    _map_labelmaps_to_annotations(annotations, labelmaps)
    _map_annotations_to_images(images, annotations)
    return {
        "location": folder,
        "images": images,
    }


def _alphanumkey(s):
    s = os.path.splitext(s)[0]
    # Split the string into two parts: all characters before the last digit sequence, and the last digit sequence
    match = re.match(r"(.*?)(\d*)$", s)
    if match:
        alpha_part = match.group(1)
        num_part = match.group(2)
        num_part = int(num_part) if num_part else 0
        return (alpha_part, num_part)
    else:
        return (s, 0)


def _list_files(folder):
    filedescriptors = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(root, file)
            filedescriptors.append(_describe_file(file_path.split(folder)[1]))
    filedescriptors = sorted(filedescriptors, key=lambda x: _alphanumkey(x["file"]))
    _add_indices(filedescriptors)
    return filedescriptors


def _add_indices(files):
    for i, f in enumerate(files):
        f["index"] = i


def _describe_file(f):
    name = f.split("/")[-1]
    dirname = os.path.dirname(f)
    fullkey, extension = os.path.splitext(f)
    key = os.path.splitext(name)[0]
    return {
        "file": f,
        "dirname": dirname,
        "name": name,
        "extension": extension.lower(),
        "key": key.lower(),
        "fullkey": fullkey.lower(),
    }


def _map_annotations_to_images(images, annotations):
    imgmap = {i["fullkey"]: i for i in images}
    countmapped = 0
    for ann in annotations:
        image = imgmap.get(ann["fullkey"])
        if image:
            image["annotationfile"] = ann
            countmapped += 1
    if countmapped >= 0:
        return
    imgmap = {i["key"]: i for i in images}
    for ann in annotations:
        image = imgmap.get(ann["key"])
        if image:
            image["annotationfile"] = ann


def _map_labelmaps_to_annotations(annotations, labelmaps):
    if not labelmaps:
        return
    labelmapmap = {lm["dirname"]: lm for lm in labelmaps}
    if len(labelmapmap) < len(labelmaps):
        print("warning: unexpectedly found multiple labelmaps per directory")
        print([lm["file"] for lm in labelmaps])
    for ann in annotations:
        labelmap = labelmapmap.get(ann["dirname"])
        if labelmap:
            ann["labelmap"] = labelmap["labelmap"]


def _load_labelmaps(folder, labelmaps):
    for labelmap in labelmaps:
        try:
            labelmap["labelmap"] = load_labelmap(f"{folder}/{labelmap['file']}")
        except Exception:
            # raise Exception(f"failed to load labelmap {labelmap['file']}")
            pass
    return [lm for lm in labelmaps if lm.get("labelmap")]


def _decide_split(images):
    for i in images:
        fullkey = i["fullkey"]
        if "valid" in fullkey:
            i["split"] = "valid"
        elif "train" in fullkey:
            i["split"] = "train"
        elif "test" in fullkey:
            i["split"] = "test"
        else:
            i["split"] = "train"
