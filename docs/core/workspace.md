:::roboflow.core.workspace

## Search Workflow Eval Dataset Cases

```python
evals = workspace.workflow_evals()
query = {"groundTruth": {"output": "boxes", "className": "car", "count": {"min": 1}}}
cursor = None
while True:
    page = evals.search_dataset("DATASET_ID", query=query, limit=24, cursor=cursor)
    for case in page["items"]:
        print(case["id"], case["name"])
    cursor = page.get("nextCursor")
    if not cursor:
        break
```

`query` also accepts `semantic`, Source `filename`, `tags`, `width`/`height` ranges,
and ground-truth `state`, `slices`, `sliceMatch` and exact `scalar: {output, value}`.
Source and semantic filters require `image:read` in addition to `workflow-evals:read`
and only match Cases backed by platform Sources. Ground-truth filters also cover
artifact-only Cases. Class/count filters require an output name.

Pages include Dataset revisions, coverage counts and a possibly unknown (`None`)
`total`. Follow `nextCursor` even when `items` is empty, keeping the same query and
limit (1–48, default 24). On HTTP 409, restart without a cursor. Preview URLs expire.
This search does not execute a Workflow or create embeddings.

## Publish a Workflow with Deploy Gates

A workflow with a deploy gate publishes only a saved version that passed every gate, so
`update_workflow` raises `WorkflowDeployGateError` (`code == "deploy_check_required"`) and saves
nothing. Save a version, run its gates and deploy it:

```python
gates = workspace.workflow_deploy_gates()
workflow_id = workspace.get_workflow("my-workflow")["workflow"]["id"]

version_id = gates.save_version(workflow_id, definition)["version"]["id"]
gates.run(workflow_id, version_id)
verdicts = gates.wait_for_version(workflow_id, version_id)
if verdicts["publishable"]:
    gates.deploy(workflow_id, version_id)
else:
    for gate in verdicts["gates"]:
        print(gate["type"], gate["status"], gate["reason"])
```

`save_version` and `run` send an `Idempotency-Key`; pass `idempotency_key=` to retry one safely.
On an Auto-Publish workflow a version that passes publishes itself, so `deploy` is only needed on a
Protected one. Scopes: `workflow:update` to save and deploy; `workflow:update`,
`workflow-evals:run`, `workflow:read` and `model:infer` to run; `workflow:read` to read verdicts.
Add `workflow-evals:read` to see a check's Eval Run results. Folder-scoped keys are refused.

To gate a workflow on a Workflow Eval:

```python
gates.set_eval_gate(workflow_id, "EVAL_ID", check_thresholds={"code-reading": 0.98})
```

Checks left out need every applicable Case to pass.
