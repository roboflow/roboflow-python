"""Manual demo/smoke test for the image metadata SDK wrappers (DATAMAN-337).

Usage (staging):
    API_URL=https://api.roboflow.one ROBOFLOW_API_KEY=<staging-key> \
        .venv/bin/python tests/manual/demo_image_metadata.py

Optional env: RF_WORKSPACE (default model-evaluation-workspace),
RF_PROJECT (default penguin-finder).

Exercises workspace.update_image_metadata, project.update_image_metadata,
batch_update_image_metadata (no-wait + wait=True with a bogus id), server-side
validation errors, and cleans up everything it wrote.
"""

import os
import time

import roboflow
from roboflow.adapters.rfapi import RoboflowError

WORKSPACE = os.environ.get("RF_WORKSPACE", "model-evaluation-workspace")
PROJECT = os.environ.get("RF_PROJECT", "penguin-finder")
TAG = f"smoke-{int(time.time())}"

rf = roboflow.Roboflow()
workspace = rf.workspace(WORKSPACE)
project = workspace.project(PROJECT)

ids = [r["id"] for r in project.search(fields=["id"], limit=2)]
assert len(ids) == 2, f"need 2 images in {WORKSPACE}/{PROJECT}, got {len(ids)}"
print(f"image ids: {ids}, tag: {TAG}")

# --- Single image (workspace) ---
print("=== workspace.update_image_metadata ===")
r = workspace.update_image_metadata(ids[0], metadata={"smoke_key": "v1"}, add_tags=[TAG])
print(r)
assert r == {"success": True}

# --- Single image (project alias) ---
print("=== project.update_image_metadata ===")
r = project.update_image_metadata(ids[0], metadata={"smoke_project": True})
print(r)
assert r == {"success": True}

# --- Batch, fire-and-forget ---
print("=== batch (no wait) + get_async_task ===")
r = workspace.batch_update_image_metadata([{"imageId": ids[0], "addTags": [TAG]}])
print(r)
assert "taskId" in r and "url" in r
for _ in range(20):
    status = workspace.get_async_task(r["taskId"])
    if status["status"] not in ("created", "running"):
        break
    time.sleep(3)
print(status)
assert status["status"] == "completed"

# --- Batch, wait=True, with one bogus id -> partial success ---
print("=== batch (wait=True) with bogus id ===")
updates = [{"imageId": i, "metadata": {"smoke_batch": "yes"}} for i in ids]
updates.append({"imageId": "bogus-does-not-exist", "addTags": [TAG]})
final = workspace.batch_update_image_metadata(updates, wait=True, timeout=300)
print(final["status"], final["result"])
assert final["result"]["succeeded"] == 2
assert final["result"]["failedItems"][0]["imageId"] == "bogus-does-not-exist"

# --- Server-side validation surfaces as RoboflowError ---
print("=== validation errors ===")
for kwargs, expect in [
    ({"add_tags": ["bad tag spaces"]}, "Invalid tag"),
    ({}, "At least one of"),
]:
    try:
        workspace.update_image_metadata(ids[0], **kwargs)
        raise AssertionError(f"expected RoboflowError containing {expect!r}")
    except RoboflowError as e:
        assert expect in str(e)
        print(f"ok: {expect!r} surfaced")

# --- Cleanup ---
print("=== cleanup ===")
cleanup = [
    {
        "imageId": i,
        "removeMetadata": ["smoke_key", "smoke_project", "smoke_batch"],
        "removeTags": [TAG],
    }
    for i in ids
]
final = workspace.batch_update_image_metadata(cleanup, wait=True, timeout=300)
print(final["status"], final["result"])
assert final["result"]["succeeded"] == 2

print("\nALL CHECKS PASSED")
