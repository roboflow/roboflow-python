# Roboflow CLI

The `roboflow` command line tool provides access to the Roboflow platform for managing computer vision projects, datasets, models, and deployments. It's designed for both human developers and AI coding agents.

> **Full reference:** [docs.roboflow.com/deploy/sdks/python-cli](https://docs.roboflow.com/deploy/sdks/python-cli)

## Install & authenticate

```bash
pip install roboflow
export ROBOFLOW_API_KEY=rf_xxxxx    # recommended for scripts and agents
roboflow auth login                  # or interactive login
```

### Select a Roboflow region

Roboflow uses the US platform by default. To authenticate with the EU
data-residency platform, select the region during login:

```bash
roboflow auth login --region eu
# The backwards-compatible alias accepts the same option:
roboflow login --region eu
```

The selection is saved in the Roboflow config file. You can change it later
or inspect the effective endpoints with:

```bash
roboflow auth set-region eu
roboflow auth status
```

For CI and other non-interactive environments, set `ROBOFLOW_REGION=eu`.
`ROBOFLOW_REGION` accepts `us` or `eu` (case-insensitive); an environment
value takes precedence over the saved region. Explicit per-URL environment or
config values such as `API_URL` continue to take precedence over the region.

| Endpoint | `us` (default) | `eu` |
|----------|----------------|------|
| API | `https://api.roboflow.com` | `https://api.roboflow.eu` |
| App / CLI authentication | `https://app.roboflow.com` | `https://app.roboflow.eu` |
| Object detection | `https://serverless.roboflow.com` | `https://serverless.roboflow.eu` |
| Instance segmentation | `https://serverless.roboflow.com` | `https://serverless.roboflow.eu` |
| Dedicated deployment | `https://roboflow.cloud` | `https://eu.roboflow.cloud` |
| Universe | `https://universe.roboflow.com` | `https://universe.roboflow.com` |
| Semantic segmentation | `https://segment.roboflow.com` | `https://segment.roboflow.com` |

Roboflow Universe remains a single global product, so its URL stays on
`.com` in the EU region. Semantic segmentation also remains on its current
`.com` endpoint. EU and US use separate authentication backends; obtain EU
API keys from `https://app.roboflow.eu` and log in again after switching if
your existing credentials were issued by the other region.

## Global flags

| Flag | Short | Description |
|------|-------|-------------|
| `--json` | `-j` | Structured JSON output (for agents and piping) |
| `--api-key` | `-k` | API key override |
| `--workspace` | `-w` | Workspace override |
| `--quiet` | `-q` | Suppress progress bars and status messages |
| `--version` | | Show version |

Flags work in any position: `roboflow project list --json` and `roboflow --json project list` are equivalent.

## Quick examples

### Create a project and upload images

```bash
roboflow project create my-project --type object-detection
roboflow image upload photo.jpg -p my-project
roboflow image upload ./dataset-folder/ -p my-project   # smart: detects directory
```

### Download a dataset

```bash
roboflow version download my-workspace/my-project/3 -f yolov8
roboflow download my-workspace/my-project/3 -f coco   # alias
```

### Run inference

```bash
roboflow infer photo.jpg -m my-project/3
```

### Train, monitor, cancel, stop

```bash
# Start training (any architecture). For NAS sweeps, use a NAS parent modelType:
roboflow train start -p my-project -v 3 --type rfdetr-base
roboflow train start -p my-project -v 3 --type rfdetr-nas-parent      # NAS sweep
roboflow train start -p my-project -v 3 --type rfdetr-nas-base-parent # NAS Base sweep
roboflow train start -p my-project -v 3 --type rfdetr-nas-seg-parent  # NAS instance-segmentation

# Cancel an in-flight training (any architecture; NAS-aware):
roboflow train cancel my-project/3
# Pass --continue-if-no-refund to cancel even past the refund window:
roboflow train cancel my-project/3 --continue-if-no-refund

# Graceful early-stop:
roboflow train stop my-project/3

# Run-level training results bundle (NAS leaderboard for NAS runs,
# minimal bundle for non-NAS):
roboflow train results my-project/3
```

NAS sweeps require the version's validation split to have at least 15 images;
the server returns `code: "insufficient_validation_images_for_nas"` otherwise.

### Train recipes — custom hyperparameters & augmentation (v2)

```bash
# Inspect a model type's tunable hyperparameter schema, allowed online
# augmentation/preprocessing steps, and a ready-to-edit recipe template:
roboflow train recipe -p my-project -v 3 -m rfdetr-medium

# Start a training from an edited recipe: take the `template` field, tweak
# it (hyperparameters, online augmentation), and submit it. The server
# dense-fills any defaults the recipe leaves out:
roboflow --json train recipe -p my-project -v 3 -m rfdetr-medium | jq .template > recipe.json
# ... edit recipe.json (e.g. set .hyperparameters.lr) ...
roboflow train start -p my-project -v 3 -t rfdetr-medium --train-recipe @recipe.json
```

--train-recipe accepts inline JSON or a curl-style @file reference; it creates
the training through the v2 trainings API and prints
the new `trainingId` instead of blocking — handy for launching sweeps and
polling status separately. --epochs is folded into the recipe's
hyperparameters unless the recipe already sets epochs.

### NAS models — list, star, deploy

```bash
# Get a NAS run's modelGroup from training results:
roboflow --json train results my-project/3 | jq -r .modelGroup
# → rfdetrNasGroup-3

# List every model from one NAS run, with hardware/latency/mAP columns:
roboflow model list -p my-project --group rfdetrNasGroup-3

# Star a NAS-trained model (triggers TRT compile for its recommended hardware):
#   --json train results … gives you the modelId per row.
roboflow model star <modelId>
roboflow model star <modelId> --unstar
```

`model star` is NAS-only by server-side design; non-NAS modelTypes return
`code: "MODEL_NOT_NAS"`.

### Update image metadata and tags

```bash
# Single image: set metadata + add tags
roboflow image metadata <image_id> -m '{"camera": "cam1"}' --tags "review,v2"

# Remove metadata keys
roboflow image metadata <image_id> --remove-metadata "old_key"

# Remove tags
roboflow image metadata <image_id> --remove-tags "draft"

# Batch: update multiple images (async), poll for completion
roboflow image metadata img1,img2,img3 --tags "processed" --poll

# Batch with timeout
roboflow image metadata img1,img2 -m '{"status": "done"}' --poll --timeout 600

# Tag alias works identically (hidden command)
roboflow image tag <image_id> --tags "review" --remove-tags "draft"
```

Single image ID updates synchronously. Multiple comma-separated IDs use the
batch async endpoint (up to 1000 images). Use `--poll` to block until
completion; without it the command returns the `taskId` immediately.

### Search and export

```bash
roboflow search "tag:reviewed" --limit 100
roboflow search "class:person" --export -f coco -l ./export/
```

### Browse resources

```bash
roboflow workspace list
roboflow project list
roboflow project get my-project
roboflow version list -p my-project
roboflow model list -p my-project
```

### Manage folders

```bash
roboflow folder list
roboflow folder create "Training Data" --projects proj1,proj2
roboflow folder get <folder-id>
roboflow folder update <folder-id> --name "New Name"
roboflow folder delete <folder-id>
```

### Annotation batches and jobs

```bash
roboflow annotation batch list -p my-project
roboflow annotation batch get <batch-id> -p my-project
roboflow annotation job list -p my-project
roboflow annotation job create -p my-project --name "Label round 1" \
  --batch <batch-id> --num-images 100 --labeler a@co.com --reviewer b@co.com
```

### RFDM devices (v2 deployments)

Workspace-scoped device management — backed by the external Deployments API
(`/:workspace/devices/v2/*`). Read commands need the `device:read` scope on
your api_key; `create` needs `device:update`.

```bash
roboflow device list
roboflow device get <device-id>
roboflow device create "Factory floor cam" --type edge --tags floor-1,vision

# Observe — config is sensitive (may include credentials).
roboflow device config <device-id>
roboflow device config-history <device-id> --limit 20

# Streams the device runs.
roboflow device streams <device-id>
roboflow device stream <device-id> <stream-id>

# Logs (5 req/min/IP) and aggregated telemetry (60 req/min).
roboflow device logs <device-id> --severity ERROR --limit 200
roboflow device telemetry <device-id> --time-period 7d

# Lifecycle events (stream start/stop, errors, config changes…).
roboflow device events <device-id> --entity-type stream --direction backward
```

### Workflows

```bash
roboflow workflow list
roboflow workflow get my-workflow
roboflow workflow create --name "My Workflow" --definition workflow.json
roboflow workflow update my-workflow --definition updated.json
roboflow workflow version list my-workflow
roboflow workflow fork other-ws/their-workflow
```

### Fork a Universe project (async)

```bash
# Fork a public Universe project into the default (or --workspace) workspace.
# By default this blocks until the async task completes (up to --timeout seconds).
roboflow project fork https://universe.roboflow.com/leo-ueno-uduc7/license-plate-recognition
roboflow project fork leo-ueno-uduc7/license-plate-recognition --workspace my-ws

# Return immediately with a {taskId, url} payload instead of waiting.
roboflow project fork leo-ueno-uduc7/license-plate-recognition --no-wait

# Poll the resulting task later (works for any async task that returns a taskId).
roboflow asynctasks get  <task-id>
roboflow asynctasks wait <task-id> --timeout 600
```

### Create a dataset version

```bash
roboflow version create -p my-project --settings settings.json
```

### Delete and restore (soft delete / Trash)

```bash
# Move a project to Trash — any in-flight training jobs are cancelled automatically.
# Items stay in Trash for 30 days, then are permanently cleaned up.
roboflow project delete my-workspace/my-project
roboflow project restore my-workspace/my-project

# Same flow for versions (also cancels in-flight training on the version).
roboflow version delete my-workspace/my-project/3
roboflow version restore my-workspace/my-project/3

# Same flow for workflows.
roboflow workflow delete my-workflow
roboflow workflow restore my-workflow

# Inspect what's currently in Trash.
roboflow trash list

# Skip the confirmation prompt for scripts.
roboflow project delete my-workspace/my-project --yes
```

Permanent deletion (emptying Trash or skipping the retention window for a
single item) is intentionally not available from the SDK or CLI — those
actions destroy data irrecoverably and live only in the web UI's Trash
view. Items left in Trash are cleaned up automatically after 30 days.

### Inspect model evaluations

```bash
# List evals in the workspace; filter by project, version, model, or status.
roboflow eval list --status done --limit 10

# Read a single eval's metadata + summary metrics.
roboflow eval get <eval-id>

# Pull each panel — pipe to jq for structured access.
roboflow eval map-results <eval-id> --json | jq '.splits.test.map50'
roboflow eval performance-by-class <eval-id> --split test
roboflow eval confusion-matrix <eval-id> --split test --confidence 30
roboflow eval confidence-sweep <eval-id> --json
roboflow eval vector-analysis <eval-id> --confidence 20 --json
roboflow eval image-predictions <eval-id> --split test --limit 200
roboflow eval recommendations <eval-id> --json
```

Exit codes are stable per error class so scripts and agents can react
without parsing message strings: `3` for `model_eval_not_found` (404),
`4` for `model_eval_not_done` (409 — eval still running), `5` for
`invalid_split` / `invalid_confidence` (400). Requires the
`model-eval:read` scope on the api key.

### Workspace stats and billing

```bash
roboflow workspace usage
roboflow workspace plan
roboflow workspace stats --start-date 2026-01-01 --end-date 2026-03-31
```

### Search Roboflow Universe

```bash
roboflow universe search "hard hats" --type dataset --limit 5
```

### Video inference

```bash
roboflow video infer -p my-project -v 3 -f video.mp4 --fps 10
roboflow video status <job-id>
```

### Shell completion

The fastest path: let the CLI install completion for you. Auto-detects your shell from `$SHELL`.

```bash
roboflow completion install
```

This writes the completion script to a per-user location and updates your shell rc file (`~/.bashrc` or `~/.zshrc`) so completion works in new shells. Idempotent — safe to re-run. Delegates to `typer.completion.install` under the hood.

Supported shells: `bash`, `zsh`, `fish`. Windows / PowerShell is not supported.

Override detection or scope to one shell:

```bash
roboflow completion install --shell zsh
roboflow completion install --shell bash
roboflow completion install --shell fish
```

Hidden commands (legacy aliases, snake_case shims, not-yet-implemented stubs) are filtered from completion automatically.

To uninstall, delete the completion script (location depends on your shell — typer writes to `~/.bash_completions/roboflow.sh`, `~/.zfunc/_roboflow`, or `~/.config/fish/completions/roboflow.fish`) and remove any `source ...` line typer added to your `~/.bashrc`.

#### Advanced: print the script yourself

If you want full control, generate the raw script and source it however you like:

```bash
# Zsh
eval "$(roboflow completion zsh)"

# Bash (requires bash >= 4.4)
eval "$(roboflow completion bash)"

# Fish
roboflow completion fish | source
```

## JSON output for agents

Every command supports `--json` for structured output that's safe to pipe:

```bash
# stdout: JSON data, stderr: JSON errors, exit codes: 0/1/2/3
roboflow --json project list | python3 -c "import sys,json; print(json.load(sys.stdin))"
roboflow --json project get nonexistent 2>/dev/null   # stderr gets the error JSON
```

Error schema is consistent: `{"error": {"message": "...", "hint": "..."}}`

## Resource shorthand

Resources can be addressed with compact identifiers:

| Shorthand | Resolves to |
|-----------|-------------|
| `my-project` | default workspace + project |
| `my-ws/my-project` | explicit workspace + project |
| `my-project/3` | default workspace + project + version 3 |
| `my-ws/my-project/3` | explicit workspace + project + version 3 |

Version numbers are always numeric — that's how `x/y` is disambiguated between `workspace/project` and `project/version`.

## All command groups

| Command | Description |
|---------|-------------|
| `auth` | Login, logout, status, set region or default workspace |
| `api-key` | List, create, update, protect, disable, revoke workspace API keys |
| `workspace` | List and inspect workspaces |
| `project` | List, get, create projects |
| `version` | List, get, download, export dataset versions |
| `image` | Upload, get, search, metadata, tag, delete, annotate images |
| `model` | List, get, upload trained models |
| `train` | Start model training |
| `infer` | Run inference on images |
| `search` | Search workspace images (RoboQL), export results |
| `deployment` | Manage dedicated deployments |
| `device` | List, get, create, and observe RFDM devices (v2 deployment API) |
| `eval` | Inspect model evaluation runs (mAP, confusion matrix, recommendations, ...) |
| `workflow` | Manage workflows |
| `folder` | Manage workspace folders |
| `annotation` | Annotation batches and jobs |
| `asynctasks` | Inspect async background tasks (e.g. project forks) |
| `trash` | List items in Trash |
| `universe` | Search Roboflow Universe |
| `video` | Video inference |
| `batch` | Batch processing jobs *(coming soon)* |
| `completion` | Install or generate shell completion scripts (bash, zsh, fish) |

Run `roboflow <command> --help` for details on any command.

## Backwards compatibility

All legacy command names still work:

| Legacy | Current |
|--------|---------|
| `roboflow login` | `roboflow auth login` |
| `roboflow whoami` | `roboflow auth status` |
| `roboflow upload <file>` | `roboflow image upload <file>` |
| `roboflow import <dir>` | `roboflow image upload <dir>` |
| `roboflow download <url>` | `roboflow version download <url>` |
| `roboflow search-export` | `roboflow search --export` |
| `roboflow train` | `roboflow train start` |
| `roboflow deployment add` | `roboflow deployment create` |
| `roboflow deployment machine_type` | `roboflow deployment machine-type` |
