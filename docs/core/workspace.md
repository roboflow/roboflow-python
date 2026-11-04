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
