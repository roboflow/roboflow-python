import os

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
ANNOTATION_EXTENSIONS = {".txt", ".json", ".xml"}


def parsefolder(folder):
    if not os.path.exists(folder):
        raise Exception(f"folder does not exist. {folder}")
    files = _list_files(folder)
    images = [f for f in files if f["extension"] in IMAGE_EXTENSIONS]
    annotations = [f for f in files if f["extension"] in ANNOTATION_EXTENSIONS]
    _map_annotations_to_images(images, annotations)
    return {
        "location": folder,
        "images": images,
    }


def _list_files(folder):
    filedescriptors = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(root, file)
            filedescriptors.append(_describe_file(file_path.split(folder)[1]))
    return filedescriptors


def _describe_file(f):
    name = f.split("/")[-1]
    fullkey, extension = os.path.splitext(f)
    key = os.path.splitext(name)[0]
    return {
        "file": f,
        "split": "train",  # TODO: decide split
        "name": name,
        "extension": extension.lower(),
        "key": key.lower(),
        "fullkey": fullkey.lower(),
    }


def _map_annotations_to_images(images, annotations):
    imgmap = {i["fullkey"]: i for i in images}
    for ann in annotations:
        image = imgmap.get(ann["fullkey"])
        if image:
            image["annotationfile"] = ann
