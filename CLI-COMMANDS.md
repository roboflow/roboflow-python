# Roboflow CLI

The `roboflow` command line tool provides access to the Roboflow platform for managing computer vision projects, datasets, models, and deployments. It's designed for both human developers and AI coding agents.

> **Full reference:** [docs.roboflow.com/deploy/sdks/python-cli](https://docs.roboflow.com/deploy/sdks/python-cli)

## Install & authenticate

```bash
pip install roboflow
export ROBOFLOW_API_KEY=rf_xxxxx    # recommended for scripts and agents
roboflow auth login                  # or interactive login
```

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

### Workflows

```bash
roboflow workflow list
roboflow workflow get my-workflow
roboflow workflow create --name "My Workflow" --definition workflow.json
roboflow workflow update my-workflow --definition updated.json
roboflow workflow version list my-workflow
roboflow workflow fork other-ws/their-workflow
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

# Inspect what's currently in Trash.
roboflow trash list

# Skip the confirmation prompt for scripts.
roboflow project delete my-workspace/my-project --yes
```

Permanent deletion (emptying Trash or skipping the retention window for a
single item) is intentionally not available from the SDK or CLI — those
actions destroy data irrecoverably and live only in the web UI's Trash
view. Items left in Trash are cleaned up automatically after 30 days.

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
| `auth` | Login, logout, status, set default workspace |
| `workspace` | List and inspect workspaces |
| `project` | List, get, create projects |
| `version` | List, get, download, export dataset versions |
| `image` | Upload, get, search, tag, delete, annotate images |
| `model` | List, get, upload trained models |
| `train` | Start model training |
| `infer` | Run inference on images |
| `search` | Search workspace images (RoboQL), export results |
| `deployment` | Manage dedicated deployments |
| `workflow` | Manage workflows |
| `folder` | Manage workspace folders |
| `annotation` | Annotation batches and jobs |
| `trash` | List items in Trash |
| `universe` | Search Roboflow Universe |
| `video` | Video inference |
| `batch` | Batch processing jobs *(coming soon)* |
| `completion` | Generate shell completion scripts (bash, zsh, fish) |

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
