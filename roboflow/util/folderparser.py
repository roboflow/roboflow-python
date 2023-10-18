import os

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]


def parsefolder(folder):
    if not os.path.exists(folder):
        raise Exception(f"folder does not exist. {folder}")
    files = _list_files(folder)
    images = [{"file": f, "key": _imgkey(f)} for f in files if _is_image(f)]
    return None


def _list_files(folder):
    file_paths = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path.split(folder)[1])
    return file_paths


def _is_image(f):
    return any([f.lower().endswith(ext) for ext in IMAGE_EXTENSIONS])


def _imgkey(f):
    name = f.split("/")[-1]
    key = os.path.splitext(name)[0]
    return key
