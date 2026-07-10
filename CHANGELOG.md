# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

### Added

- Upload raw rf-detr PyTorch-Lightning checkpoints (e.g. `checkpoint_best_ema.pth`):
  `upload_model` detects them and rebuilds a deploy-ready bundle via rf-detr's
  `export_for_roboflow` (requires `rfdetr>=1.8.0`)
  ([#488](https://github.com/roboflow/roboflow-python/pull/488))

## 1.3.11

### Added

- `roboflow api-key` CLI command group and SDK methods to create, list, get,
  update, protect, and revoke workspace API keys — including scoped keys, folder
  restrictions, and custom metadata (scoping/metadata require the Advanced API
  Keys plan feature).

## 1.3.10

### Added

- Weight upload support for yolo26-sem semantic segmentation models via
  `version.deploy()` and `workspace.deploy_model()`

## 1.3.9

### Added — Model evaluations SDK & CLI

Wraps the public `/{workspace}/model-evals` REST surface
([roboflow/roboflow#11636](https://github.com/roboflow/roboflow/pull/11636))
so users can read evaluation results — mAP, confidence sweep, per-class
performance, confusion matrix, vector clusters, per-image stats,
recommendations — from Python and from the CLI without hitting the API
directly. Companion docs:
[roboflow-dev-reference#18](https://github.com/roboflow/roboflow-dev-reference/pull/18).

**SDK (`roboflow/core/model_eval.py`):**
- `Workspace.evals(project=None, version=None, model=None, status=None, limit=None)` — list evals as `ModelEval` instances pre-populated with metadata from the list response.
- `Workspace.eval(eval_id)` — fetch a single eval (returns a `ModelEval` with `.summary` populated when status is `done`).
- `ModelEval.refresh()` — re-fetch the eval header.
- `ModelEval.map_results()`, `.confidence_sweep()`, `.performance_by_class(split=None)`, `.confusion_matrix(split=None, confidence=None)`, `.vector_analysis(confidence=None)`, `.image_predictions(split=None, confidence=None, limit=None, offset=None)`, `.recommendations()` — one method per panel; each returns the raw JSON dict.

**CLI (`roboflow/cli/handlers/eval.py`):**
- `roboflow eval list [--project P] [--version V] [--model M] [--status S] [--limit N]`
- `roboflow eval get <eval_id>`
- `roboflow eval map-results <eval_id>`
- `roboflow eval confidence-sweep <eval_id>`
- `roboflow eval performance-by-class <eval_id> [--split S]`
- `roboflow eval confusion-matrix <eval_id> [--split S] [--confidence N]`
- `roboflow eval vector-analysis <eval_id> [--confidence N]`
- `roboflow eval image-predictions <eval_id> [--split S] [--confidence N] [--limit N] [--offset N]`
- `roboflow eval recommendations <eval_id>`

Exit codes are stable per error class so shell scripts and AI agents can
react without parsing message strings: `3` for `model_eval_not_found`
(404), `4` for `model_eval_not_done` (409), `5` for `invalid_split` /
`invalid_confidence` (400). Every command supports `--json` for
structured output.

**Low-level (`roboflow.adapters.rfapi`):**
- `list_model_evals`, `get_model_eval`, `get_model_eval_map_results`, `get_model_eval_confidence_sweep`, `get_model_eval_performance_by_class`, `get_model_eval_confusion_matrix`, `get_model_eval_vector_analysis`, `get_model_eval_image_predictions`, `get_model_eval_recommendations`.
- New typed exceptions `ModelEvalNotFoundError`, `ModelEvalNotDoneError`, `InvalidSplitError`, `InvalidConfidenceError` (all subclasses of `RoboflowError`) so callers can distinguish "eval doesn't exist" from "eval still running" from "bad argument" without parsing strings.

The endpoints require the `model-eval:read` scope. The base URL is
configurable via `API_URL` (set to `https://localapi.roboflow.one` to
test against a local API server).

### Fixed
- rf-detr model upload: accept checkpoints whose `args` is a plain dict (e.g. EMA checkpoints) when extracting class names, instead of raising `TypeError` from `vars()`.

### Changed
- Pin `typer<0.26` and declare `click` explicitly: typer 0.26 vendors its own click and drops the external dependency, which broke the CLI and its type checks.

## 1.3.7

### Added — Soft-delete / Trash support

Mirrors the soft-delete and Trash features added to the Roboflow web app
([roboflow/roboflow#11131](https://github.com/roboflow/roboflow/pull/11131)).
Deleting a project, version, or workflow now moves it to Trash with a
30-day retention window (and cancels any in-flight training jobs); items
can be restored within that window. Companion docs:
[roboflow-dev-reference#5](https://github.com/roboflow/roboflow-dev-reference/pull/5).

**SDK (`roboflow/`):**
- `Project.delete()` / `Project.restore()` — soft-delete and restore by slug.
- `Version.delete()` / `Version.restore()` — same shape on a version handle.
- `Workspace.trash()` — list everything currently in a workspace's Trash, grouped by `projects` / `versions` / `workflows`.
- `Workspace.restore_from_trash(item_type, item_id, parent_id=None)` — restore an item by id when you don't have a live SDK handle (or for workflows, which don't have a first-class object yet).

**CLI (`roboflow/cli/`):**
- `roboflow project delete` / `roboflow project restore`
- `roboflow version delete` / `roboflow version restore`
- `roboflow workflow delete` / `roboflow workflow restore`
- `roboflow trash list`

Destructive commands prompt for confirmation interactively and accept
`--yes` / `-y` for scripted use. Every command supports `--json` for
structured output and emits actionable error hints with stable exit codes.

**Low-level (`roboflow.adapters.rfapi`):**
- `delete_project`, `delete_version`, `delete_workflow`, `list_trash`, `restore_trash_item`.
- `RoboflowError` messages now extract the `error` field from JSON response bodies (e.g. "Not authorized to view trash") instead of the raw response text.

**Permanent deletion is intentionally web-UI-only.** Emptying Trash or
immediately deleting a single Trash item destroys data irrecoverably, so
those actions are not exposed on the SDK or CLI — they live only in the
Roboflow app's Trash view, which has an explicit confirmation dialog.
Items left in Trash are cleaned up automatically after 30 days.

### Fixed — Workflows created via SDK/CLI now execute successfully

`Workspace.create_workflow()` and `roboflow workflow create --definition`
auto-wrap bare workflow definitions in `{"specification": ...}` before
POSTing to the backend, matching what the web app does
([#460](https://github.com/roboflow/roboflow-python/pull/460)). Previously,
the user-facing flat shape (`{version, inputs, steps, outputs}`) was sent
verbatim, so `POST /infer/workflows/...` against the resulting workflow
returned `HTTP 502` with `MalformedWorkflowResponseError: Workflow
specification not found in Roboflow API response`.

Workflows already wrapped (top-level `specification` key) are passed
through unchanged. Non-workflow dicts and non-JSON strings are also
passed through verbatim so custom payloads aren't second-guessed.

> **Note:** workflows that were stored with the bare shape *before* this
> fix will still 502 until re-saved. Run `roboflow workflow update
> <url> --definition <file>` once per affected workflow to migrate.

### Changed — Image upload no longer re-encodes images

`upload_image` now uploads original image bytes instead of re-encoding to
JPEG client-side ([#464](https://github.com/roboflow/roboflow-python/pull/464)).

### Backward compatibility

Purely additive on the public API surface. The new endpoints require
`project:update`, `version:update`, or `workflow:update` scopes — most
existing keys already have these.

## 1.1.50

- Added support for Palligema2 model uploads via `upload_model` command with the following model types:
  - `paligemma2-3b-pt-224`
  - `paligemma2-3b-pt-448`
  - `paligemma2-3b-pt-896`
