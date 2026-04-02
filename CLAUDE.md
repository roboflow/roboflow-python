# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running Tests
```bash
python -m unittest
```

### Linting and Code Quality
```bash
# Format code with ruff
make style

# Check code quality (includes ruff and mypy)
make check_code_quality

# Individual commands
ruff format roboflow
ruff check roboflow --fix
mypy roboflow
```

### Building Documentation
```bash
# Install documentation dependencies
python -m pip install mkdocs mkdocs-material mkdocstrings mkdocstrings[python]

# Serve documentation locally
mkdocs serve
```

### Installing Development Environment
```bash
# Create virtual environment
python3 -m venv env
source env/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

## Architecture Overview

The Roboflow Python SDK follows a hierarchical object model that mirrors the Roboflow platform structure:

### Core Components

1. **Roboflow** (`roboflow/__init__.py`) - Entry point and authentication
   - Handles API key management and workspace initialization
   - Provides `login()` for CLI authentication
   - Creates workspace connections

2. **Workspace** (`roboflow/core/workspace.py`) - Manages Roboflow workspaces
   - Lists and accesses projects
   - Handles dataset uploads and model deployments
   - Manages workspace-level operations

3. **Project** (`roboflow/core/project.py`) - Represents a computer vision project
   - Manages project metadata and versions
   - Handles image/annotation uploads
   - Supports different project types (object-detection, classification, etc.)

4. **Version** (`roboflow/core/version.py`) - Dataset version management
   - Downloads datasets in various formats
   - Deploys models
   - Provides access to trained models for inference

5. **Model Classes** (`roboflow/models/`) - Type-specific inference models
   - `ObjectDetectionModel` - Bounding box predictions
   - `ClassificationModel` - Image classification
   - `InstanceSegmentationModel` - Pixel-level segmentation
   - `SemanticSegmentationModel` - Class-based segmentation
   - `KeypointDetectionModel` - Keypoint predictions

### API Adapters

- **rfapi** (`roboflow/adapters/rfapi.py`) - Low-level API communication
- **deploymentapi** (`roboflow/adapters/deploymentapi.py`) - Model deployment operations

### CLI Package (`roboflow/cli/`)

The CLI is a modular package with auto-discovered handler modules. `roboflow/roboflowpy.py` is a backwards-compatibility shim that delegates to `roboflow.cli.main`.

**Package structure:**
- `__init__.py` — Root parser with global flags (`--json`, `--workspace`, `--api-key`, `--quiet`), auto-discovery via `pkgutil.iter_modules`, custom `_CleanHelpFormatter`, and `_reorder_argv` for flexible flag positioning
- `_output.py` — `output(args, data, text)` for JSON/text output, `output_error(args, msg, hint, exit_code)` for structured errors, `suppress_sdk_output()` to silence SDK noise, `stub()` for unimplemented commands
- `_table.py` — `format_table(rows, columns)` for columnar list output
- `_resolver.py` — `resolve_resource(shorthand)` for parsing `project`, `ws/project`, `ws/project/3`
- `handlers/` — One file per command group (auto-discovered). `_aliases.py` registers backwards-compat top-level commands (loaded last)

**Adding a new command:**
1. Create `roboflow/cli/handlers/mycommand.py`
2. Export `register(subparsers)` — it will be auto-discovered
3. Use lazy imports for heavy dependencies (inside handler functions, not at module top level)
4. Use `output()` for all output, `output_error()` for all errors
5. Wrap SDK calls in `with suppress_sdk_output():` to prevent "loading..." noise
6. Add tests in `tests/cli/test_mycommand_handler.py`

**Agent experience requirements for all CLI commands:**
- Support `--json` for structured output (stable schema)
- No interactive prompts when all required flags are provided
- Structured error output: `{"error": {"message": "...", "hint": "..."}}` on stderr
- Exit codes: 0 = success, 1 = error, 2 = auth error, 3 = not found
- Actionable error messages: always tell the user what went wrong AND what to do

**Documentation policy:** `CLI-COMMANDS.md` in this repo is a quickstart only. The full command reference lives in `roboflow-product-docs` (published to docs.roboflow.com). When adding commands, update both.

### Key Design Patterns

1. **Hierarchical Access**: Always access objects through their parent (Workspace → Project → Version → Model)
2. **API Key Flow**: API key is passed down through the object hierarchy
3. **Format Flexibility**: Supports multiple dataset formats (YOLO, COCO, Pascal VOC, etc.)
4. **Batch Operations**: Upload and download operations support concurrent processing
5. **CLI Noun-Verb Pattern**: Commands follow `roboflow <noun> <verb>` (e.g. `roboflow project list`). Common operations have top-level aliases (`login`, `upload`, `download`)
6. **CLI Auto-Discovery**: Handler modules in `roboflow/cli/handlers/` are loaded automatically — no registration list to maintain
7. **Backwards Compatibility**: Legacy command names and flag signatures are preserved as hidden aliases

## Project Configuration

- **Python Version**: 3.8+
- **Main Dependencies**: See `requirements.txt`
- **Entry Point**: `roboflow=roboflow.roboflowpy:main` (shim delegates to `roboflow.cli.main`)
- **Code Style**: Enforced by ruff with Google docstring convention
- **Type Checking**: mypy configured for Python 3.8

## Important Notes

- API keys are stored in `~/.config/roboflow/config.json` (Unix) or `~/roboflow/config.json` (Windows)
- The SDK supports both hosted inference (Roboflow platform) and local inference (via Roboflow Inference)
- Pre-commit hooks automatically run formatting and linting checks
- Test files intentionally excluded from linting: `tests/manual/debugme.py`
