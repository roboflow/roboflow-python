import json
import os
import re
from collections import defaultdict

from tqdm import tqdm

from .image_utils import load_labelmap

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
ANNOTATION_EXTENSIONS = {".txt", ".json", ".xml", ".csv", ".jsonl"}
LABELMAPS_EXTENSIONS = {".labels", ".yaml", ".yml"}


def parsefolder(folder):
    folder = folder.strip()
    if folder.endswith("/"):
        folder = folder[:-1]
    if not os.path.exists(folder):
        raise Exception(f"folder does not exist. {folder}")
    files = _list_files(folder)
    images = [f for f in files if f["extension"] in IMAGE_EXTENSIONS]
    _add_indices(images)
    _decide_split(images)
    annotations = [f for f in files if f["extension"] in ANNOTATION_EXTENSIONS]
    labelmaps = [f for f in files if f["extension"] in LABELMAPS_EXTENSIONS]
    labelmaps = _load_labelmaps(folder, labelmaps)
    _map_labelmaps_to_annotations(annotations, labelmaps)
    if not _map_annotations_to_images_1to1(images, annotations):
        annotations = _loadAnnotations(folder, annotations)
        _map_annotations_to_images_1tomany(images, annotations)
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
    return filedescriptors


def _add_indices(files):
    for i, f in enumerate(files):
        f["index"] = i


def _describe_file(f):
    name = f.split("/")[-1]
    dirname = os.path.dirname(f)
    fullkey, extension = os.path.splitext(f)
    fullkey2 = fullkey.replace("/labels", "").replace("/images", "")
    key = os.path.splitext(name)[0]
    return {
        "file": f,
        "dirname": dirname,
        "name": name,
        "extension": extension.lower(),
        "key": key.lower(),
        "fullkey": fullkey.lower(),
        "fullkey2": fullkey2.lower(),
    }


def _map_annotations_to_images_1to1(images, annotations):
    imgmap = {i["fullkey"]: i for i in images}
    countmapped = 0
    for ann in annotations:
        image = imgmap.get(ann["fullkey"])
        if image:
            image["annotationfile"] = ann
            countmapped += 1
    if countmapped > 0:
        return True
    imgmap = {i["fullkey2"]: i for i in images}
    for ann in annotations:
        image = imgmap.get(ann["fullkey2"])
        if image:
            image["annotationfile"] = ann
            countmapped += 1
    return countmapped > 0


def _map_annotations_to_images_1tomany(images, annotationFiles):
    annotationsByDirname = _list_map(annotationFiles, "dirname")
    imgRefMap, annotationMap = _build_image_and_annotation_maps(annotationFiles)

    for image in tqdm(images):
        dirname = image["dirname"]
        annotationsInSameDir = annotationsByDirname.get(dirname, [])
        if annotationsInSameDir:
            for annotationFile in annotationsInSameDir:
                format = annotationFile["parsedType"]
                filtered_annotations = _filterIndividualAnnotations(
                    image, annotationFile, format, imgRefMap, annotationMap
                )
                if filtered_annotations:
                    image["annotationfile"] = filtered_annotations
                    break


def _build_image_and_annotation_maps(annotationFiles):
    imgRefMap = {}
    annotationMap = defaultdict(list)
    for annFile in annotationFiles:
        filename, dirname, parsed, parsedType = (
            annFile["file"],
            annFile["dirname"],
            annFile["parsed"],
            annFile["parsedType"],
        )
        if parsedType == "coco":
            for imageRef in parsed["images"]:
                imgRefMap[f"{filename}/{imageRef['file_name']}"] = imageRef
            for annotation in parsed["annotations"]:
                annotationMap[f"{dirname}/{annotation['image_id']}"].append(annotation)
    return imgRefMap, annotationMap


def _filterIndividualAnnotations(image, annotation, format, imgRefMap, annotationMap):
    parsed = annotation["parsed"]
    if format == "coco":
        imgReference = imgRefMap.get(f"{annotation['file']}/{image['name']}")
        if imgReference:
            # workaround to make Annotations.js correctly identify this as coco in the backend
            fake_annotation = {
                "id": 999999999,
                "image_id": 999999999,
                "category_id": 0,
                "area": 1,
                "segmentation": [],
                "iscrowd": 0,
            }
            _annotation = {"name": "annotation.coco.json"}
            annotations_for_image = annotationMap.get(f"{image['dirname']}/{imgReference['id']}", [])
            _annotation["rawText"] = json.dumps(
                {
                    "info": parsed["info"],
                    "licenses": parsed["licenses"],
                    "categories": parsed["categories"],
                    "images": [imgReference],
                    "annotations": annotations_for_image or [fake_annotation],
                }
            )
            return _annotation
    elif format == "createml":
        imgReferences = [i for i in parsed if i["image"] == image["name"]]
        if len(imgReferences) > 1:
            print(f"warning: found multiple image entries for image {image['file']} in {annotation['file']}")
        if imgReferences:
            imgReference = imgReferences[0]
            _annotation = {
                "name": "annotation.createml.json",
                "rawText": json.dumps([imgReference]),
            }
            return _annotation
    elif format == "csv":
        imgLines = [ld["line"] for ld in parsed["lines"] if ld["file_name"] == image["name"]]
        if imgLines:
            headers = parsed["headers"]
            _annotation = {
                "name": "annotation.csv",
                "rawText": "".join([headers] + imgLines),
            }
            return _annotation
        else:
            return None
    elif format == "jsonl":
        jsonlLines = [json.dumps(line) for line in parsed if line["image"] == image["name"]]
        if jsonlLines:
            _annotation = {"name": "annotation.jsonl", "rawText": "\n".join(jsonlLines)}
            return _annotation
    return None


def _loadAnnotations(folder, annotations):
    valid_extensions = {".json", ".csv", ".jsonl"}
    annotations = [a for a in annotations if a["extension"] in valid_extensions]
    for ann in annotations:
        extension = ann["extension"]
        if extension == ".json":
            with open(f"{folder}{ann['file']}") as f:
                parsed = json.load(f)
                parsedType = _guessAnnotationFileFormat(parsed, extension)
                if parsedType:
                    ann["parsed"] = parsed
                    ann["parsedType"] = parsedType
        elif extension == ".jsonl":
            ann["parsed"] = _read_jsonl(f"{folder}{ann['file']}")
            ann["parsedType"] = "jsonl"
        elif extension == ".csv":
            ann["parsedType"] = "csv"
            ann["parsed"] = _parseAnnotationCSV(f"{folder}{ann['file']}")
    return annotations


def _read_jsonl(path):
    data = []
    with open(path) as file:
        for linenum, line in enumerate(file, 1):
            if not line:
                continue
            try:
                json_object = json.loads(line.strip())
                data.append(json_object)
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line in {path}:{linenum}")
    return data


def _parseAnnotationCSV(filename):
    # TODO: use a proper CSV library?
    with open(filename) as f:
        lines = f.readlines()
    headers = lines[0]
    lines = [{"file_name": ld.split(",")[0].strip(), "line": ld} for ld in lines[1:]]
    return {
        "headers": headers,
        "lines": lines,
    }


def _guessAnnotationFileFormat(parsed, extension):
    if extension == ".json":
        if isinstance(parsed, dict):
            if isinstance(parsed.get("annotations"), list) and isinstance(parsed.get("images"), list):
                return "coco"
        elif isinstance(parsed, list):
            return "createml"
    return None


def _map_labelmaps_to_annotations(annotations, labelmaps):
    if not labelmaps:
        return
    labelmapmap = {lm["dirname"]: lm for lm in labelmaps}
    rootLabelmap = labelmapmap.get("/")
    if len(labelmapmap) < len(labelmaps):
        print("warning: unexpectedly found multiple labelmaps per directory")
        print([lm["file"] for lm in labelmaps])
    for ann in annotations:
        labelmap = labelmapmap.get(ann["dirname"]) or rootLabelmap
        if labelmap:
            ann["labelmap"] = labelmap["labelmap"]


def _load_labelmaps(folder, labelmaps):
    for labelmap in labelmaps:
        try:
            labelmap["labelmap"] = load_labelmap(f"{folder}{labelmap['file']}")
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


def _list_map(my_list, key):
    d = {}
    for i in my_list:
        d.setdefault(i[key], []).append(i)
    return d
