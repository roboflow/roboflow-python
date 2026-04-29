# Changelog

All notable changes to this project will be documented in this file.

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
