import json
import os
import shutil
import zipfile
from typing import Callable

import yaml

from roboflow.util.versions import print_warn_for_wrong_dependencies_versions


def process(model_type: str, model_path: str, filename: str) -> str:
    processor = _get_processor_function(model_type)
    return processor(model_type, model_path, filename)


def _get_processor_function(model_type: str) -> Callable:
    if model_type.startswith("yolo11"):
        model_type = model_type.replace("yolo11", "yolov11")

    supported_models = [
        "yolov5",
        "yolov7-seg",
        "yolov8",
        "yolov9",
        "yolonas",
        "paligemma",
        "paligemma2",
        "yolov10",
        "florence-2",
        "yolov11",
    ]

    if not any(supported_model in model_type for supported_model in supported_models):
        raise (ValueError(f"Model type {model_type} not supported. Supported models are {supported_models}"))

    if model_type.startswith(("paligemma", "paligemma2", "florence-2")):
        if any(model in model_type for model in ["paligemma", "paligemma2", "florence-2"]):
            supported_hf_types = [
                "florence-2-base",
                "florence-2-large",
                "paligemma-3b-pt-224",
                "paligemma-3b-pt-448",
                "paligemma-3b-pt-896",
                "paligemma2-3b-pt-224",
                "paligemma2-3b-pt-448",
                "paligemma2-3b-pt-896",
                "paligemma2-3b-pt-224-peft",
                "paligemma2-3b-pt-448-peft",
                "paligemma2-3b-pt-896-peft",
            ]
            if model_type not in supported_hf_types:
                raise RuntimeError(
                    f"{model_type} not supported for this type of upload."
                    f"Supported upload types are {supported_hf_types}"
                )
        return _process_huggingface

    if "yolonas" in model_type:
        return _process_yolonas

    return _process_yolo


def _process_yolo(model_type: str, model_path: str, filename: str) -> str:
    if "yolov8" in model_type:
        try:
            import torch
            import ultralytics

        except ImportError:
            raise RuntimeError(
                "The ultralytics python package is required to deploy yolov8"
                " models. Please install it with `pip install ultralytics`"
            )

        print_warn_for_wrong_dependencies_versions([("ultralytics", "==", "8.0.196")], ask_to_continue=True)

    elif "yolov10" in model_type:
        try:
            import torch
            import ultralytics

        except ImportError:
            raise RuntimeError(
                "The ultralytics python package is required to deploy yolov10"
                " models. Please install it with `pip install ultralytics`"
            )

    elif "yolov5" in model_type or "yolov7" in model_type or "yolov9" in model_type:
        try:
            import torch
        except ImportError:
            raise RuntimeError(
                f"The torch python package is required to deploy {model_type} models."
                " Please install it with `pip install torch`"
            )

    elif "yolov11" in model_type:
        try:
            import torch
            import ultralytics

        except ImportError:
            raise RuntimeError(
                "The ultralytics python package is required to deploy yolov11"
                " models. Please install it with `pip install ultralytics`"
            )

        print_warn_for_wrong_dependencies_versions([("ultralytics", ">=", "8.3.0")], ask_to_continue=True)

    model = torch.load(os.path.join(model_path, filename))

    if isinstance(model["model"].names, list):
        class_names = model["model"].names
    else:
        class_names = []
        for i, val in enumerate(model["model"].names):
            class_names.append((val, model["model"].names[val]))
        class_names.sort(key=lambda x: x[0])
        class_names = [x[1] for x in class_names]

    if "yolov8" in model_type or "yolov10" in model_type or "yolov11" in model_type:
        # try except for backwards compatibility with older versions of ultralytics
        if "-cls" in model_type or model_type.startswith("yolov10") or model_type.startswith("yolov11"):
            nc = model["model"].yaml["nc"]
            args = model["train_args"]
        else:
            nc = model["model"].nc
            args = model["model"].args
        try:
            model_artifacts = {
                "names": class_names,
                "yaml": model["model"].yaml,
                "nc": nc,
                "args": {k: val for k, val in args.items() if ((k == "model") or (k == "imgsz") or (k == "batch"))},
                "ultralytics_version": ultralytics.__version__,
                "model_type": model_type,
            }
        except Exception:
            model_artifacts = {
                "names": class_names,
                "yaml": model["model"].yaml,
                "nc": nc,
                "args": {
                    k: val for k, val in args.__dict__.items() if ((k == "model") or (k == "imgsz") or (k == "batch"))
                },
                "ultralytics_version": ultralytics.__version__,
                "model_type": model_type,
            }
    elif "yolov5" in model_type or "yolov7" in model_type or "yolov9" in model_type:
        # parse from yaml for yolov5

        with open(os.path.join(model_path, "opt.yaml")) as stream:
            opts = yaml.safe_load(stream)

        model_artifacts = {
            "names": class_names,
            "nc": model["model"].nc,
            "args": {
                "imgsz": opts["imgsz"] if "imgsz" in opts else opts["img_size"],
                "batch": opts["batch_size"],
            },
            "model_type": model_type,
        }
        if hasattr(model["model"], "yaml"):
            model_artifacts["yaml"] = model["model"].yaml

    with open(os.path.join(model_path, "model_artifacts.json"), "w") as fp:
        json.dump(model_artifacts, fp)

    torch.save(model["model"].state_dict(), os.path.join(model_path, "state_dict.pt"))

    list_files = [
        "results.csv",
        "results.png",
        "model_artifacts.json",
        "state_dict.pt",
    ]

    zip_file_name = "roboflow_deploy.zip"

    with zipfile.ZipFile(os.path.join(model_path, zip_file_name), "w") as zipMe:
        for file in list_files:
            if os.path.exists(os.path.join(model_path, file)):
                zipMe.write(
                    os.path.join(model_path, file),
                    arcname=file,
                    compress_type=zipfile.ZIP_DEFLATED,
                )
            else:
                if file in ["model_artifacts.json", "state_dict.pt"]:
                    raise (ValueError(f"File {file} not found. Please make sure to provide a valid model path."))

    return zip_file_name


def _process_huggingface(
    model_type: str, model_path: str, filename: str = "fine-tuned-paligemma-3b-pt-224.f16.npz"
) -> str:
    # Check if model_path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {model_path} does not exist.")
    model_files = os.listdir(model_path)
    print(f"Model files found in {model_path}: {model_files}")

    files_to_deploy = []

    # Find first .npz file in model_path
    npz_filename = next((file for file in model_files if file.endswith(".npz")), None)
    if any([file.endswith(".safetensors") for file in model_files]):
        print(f"Found .safetensors file in model path. Deploying PyTorch {model_type} model.")
        necessary_files = [
            "preprocessor_config.json",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "tokenizer.json",
        ]
        for file in necessary_files:
            if file not in model_files:
                print("Missing necessary file", file)
                res = input("Do you want to continue? (y/n)")
                if res.lower() != "y":
                    exit(1)
        for file in model_files:
            files_to_deploy.append(file)
    elif npz_filename is not None:
        print(f"Found .npz file {npz_filename} in model path. Deploying JAX PaliGemma model.")
        files_to_deploy.append(npz_filename)
    else:
        raise FileNotFoundError(f"No .npz or .safetensors file found in model path {model_path}.")

    if len(files_to_deploy) == 0:
        raise FileNotFoundError(f"No valid files found in model path {model_path}.")
    print(f"Zipping files for deploy: {files_to_deploy}")

    import tarfile

    tar_file_name = "roboflow_deploy.tar"

    with tarfile.open(os.path.join(model_path, tar_file_name), "w") as tar:
        for file in files_to_deploy:
            tar.add(os.path.join(model_path, file), arcname=file)

    print("Uploading to Roboflow... May take several minutes.")

    return tar_file_name


def _process_yolonas(model_type: str, model_path: str, filename: str = "weights/best.pt") -> str:
    try:
        import torch
    except ImportError:
        raise RuntimeError(
            "The torch python package is required to deploy yolonas models. Please install it with `pip install torch`"
        )

    model = torch.load(os.path.join(model_path, filename), map_location="cpu")
    class_names = model["processing_params"]["class_names"]

    opt_path = os.path.join(model_path, "opt.yaml")
    if not os.path.exists(opt_path):
        raise RuntimeError(
            f"You must create an opt.yaml file at {os.path.join(model_path, '')} of the format:\n"
            f"imgsz: <resolution of model>\n"
            f"batch_size: <batch size of inference model>\n"
            f"architecture: <one of [yolo_nas_s, yolo_nas_m, yolo_nas_l]."
            f"s, m, l refer to small, medium, large architecture sizes, respectively>\n"
        )
    with open(os.path.join(model_path, "opt.yaml")) as stream:
        opts = yaml.safe_load(stream)
    required_keys = ["imgsz", "batch_size", "architecture"]
    for key in required_keys:
        if key not in opts:
            raise RuntimeError(f"{opt_path} lacks required key {key}. Required keys: {required_keys}")

    model_artifacts = {
        "names": class_names,
        "nc": len(class_names),
        "args": {
            "imgsz": opts["imgsz"] if "imgsz" in opts else opts["img_size"],
            "batch": opts["batch_size"],
            "architecture": opts["architecture"],
        },
        "model_type": model_type,
    }

    with open(os.path.join(model_path, "model_artifacts.json"), "w") as fp:
        json.dump(model_artifacts, fp)

    shutil.copy(os.path.join(model_path, filename), os.path.join(model_path, "state_dict.pt"))

    list_files = [
        "results.json",
        "results.png",
        "model_artifacts.json",
        "state_dict.pt",
    ]

    zip_file_name = "roboflow_deploy.zip"

    with zipfile.ZipFile(os.path.join(model_path, zip_file_name), "w") as zipMe:
        for file in list_files:
            if os.path.exists(os.path.join(model_path, file)):
                zipMe.write(
                    os.path.join(model_path, file),
                    arcname=file,
                    compress_type=zipfile.ZIP_DEFLATED,
                )
            else:
                if file in ["model_artifacts.json", filename]:
                    raise (ValueError(f"File {file} not found. Please make sure to provide a valid model path."))

    return zip_file_name
