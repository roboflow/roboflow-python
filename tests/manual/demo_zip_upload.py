"""Manual validation for the zip upload flow on Workspace.upload_dataset.

Edit the constants below, then uncomment the scenario you want to run.
"""

from __future__ import annotations

import os
import sys
import time

thisdir = os.path.dirname(os.path.abspath(__file__))
rootdir = os.path.abspath(f"{thisdir}/../..")
sys.path.insert(0, rootdir)

from roboflow import Roboflow  # noqa: E402
from roboflow.adapters import rfapi  # noqa: E402

# ---- edit these -----------------------------------------------------------
# Reads from env by default; set directly if you prefer.
API_KEY = os.environ.get("ROBOFLOW_API_KEY", "<YOUR_API_KEY>")
WORKSPACE = os.environ.get("ROBOFLOW_WORKSPACE", "rodrigo-xn5xn")
PROJECT = os.environ.get("ROBOFLOW_PROJECT", "small-od")

ZIP_PATH = os.path.expanduser("~/Downloads/instance-seg.coco-segmentation.zip")
DIR_PATH = os.path.expanduser("~/Downloads/instance-seg.coco-segmentation")
# For the `status` scenario, paste the task_id returned by the `no_wait` run
TASK_ID = ""
# ---------------------------------------------------------------------------


def _batch(tag: str) -> str:
    return f"zip-demo-{tag}-{int(time.time())}"


def scenario_zip_path(workspace) -> None:
    print(f"\n=== scenario: zip_path  (file={ZIP_PATH}) ===")
    result = workspace.upload_dataset(
        dataset_path=ZIP_PATH,
        project_name=PROJECT,
        batch_name=_batch("zip"),
    )
    print(f"result: {result}")


def scenario_dir_default(workspace) -> None:
    """Directory without use_zip_upload — legacy per-image flow (returns None)."""
    print(f"\n=== scenario: dir_default  (dir={DIR_PATH}) ===")
    result = workspace.upload_dataset(
        dataset_path=DIR_PATH,
        project_name=PROJECT,
        batch_name=_batch("dir-peritem"),
    )
    print(f"result: {result}  (expected: None -- per-image flow)")


def scenario_dir_zip_opt_in(workspace) -> None:
    """Directory with use_zip_upload=True — SDK zips client-side."""
    print(f"\n=== scenario: dir_zip_opt_in  (dir={DIR_PATH}) ===")
    result = workspace.upload_dataset(
        dataset_path=DIR_PATH,
        project_name=PROJECT,
        batch_name=_batch("dir-zip"),
        use_zip_upload=True,
    )
    print(f"result: {result}")


def scenario_no_wait(workspace) -> None:
    print(f"\n=== scenario: no_wait  (file={ZIP_PATH}) ===")
    result = workspace.upload_dataset(
        dataset_path=ZIP_PATH,
        project_name=PROJECT,
        batch_name=_batch("nowait"),
        wait=False,
    )
    print(f"result: {result}")
    print(f"-> paste this task_id into TASK_ID and run scenario_status: {result['task_id']}")


def scenario_status(workspace) -> None:
    print(f"\n=== scenario: status  (task_id={TASK_ID}) ===")
    status = rfapi.get_zip_upload_status(API_KEY, workspace.url, TASK_ID)
    print(f"status: {status}")


def scenario_with_tags_and_split(workspace) -> None:
    print(f"\n=== scenario: tags + split  (file={ZIP_PATH}) ===")
    result = workspace.upload_dataset(
        dataset_path=ZIP_PATH,
        project_name=PROJECT,
        batch_name=_batch("tagged"),
        split="train",
        tags=["reviewed", "batch-q4"],
    )
    print(f"result: {result}")


def scenario_prediction_per_image(workspace) -> None:
    """Prediction upload always uses per-image flow (zip flow doesn't support it)."""
    print(f"\n=== scenario: prediction_per_image  (dir={DIR_PATH}) ===")
    result = workspace.upload_dataset(
        dataset_path=DIR_PATH,
        project_name=PROJECT,
        batch_name=_batch("pred"),
        is_prediction=True,
    )
    print(f"result: {result}  (expected: None -- per-image flow)")


if __name__ == "__main__":
    rf = Roboflow(api_key=API_KEY)
    workspace = rf.workspace(WORKSPACE)

    # Uncomment the scenario you want to run:
    # scenario_zip_path(workspace)
    # scenario_dir_default(workspace)
    scenario_dir_zip_opt_in(workspace)
    # scenario_no_wait(workspace)
    # scenario_status(workspace)
    # scenario_with_tags_and_split(workspace)
    # scenario_prediction_per_image(workspace)
