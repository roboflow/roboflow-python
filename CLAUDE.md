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

### CLI Interface

The `roboflow` command line tool (`roboflow/roboflowpy.py`) provides:
- Authentication: `roboflow login`
- Dataset operations: `roboflow download`, `roboflow upload`, `roboflow import`
- Inference: `roboflow infer`
- Project/workspace management: `roboflow project`, `roboflow workspace`

### Key Design Patterns

1. **Hierarchical Access**: Always access objects through their parent (Workspace → Project → Version → Model)
2. **API Key Flow**: API key is passed down through the object hierarchy
3. **Format Flexibility**: Supports multiple dataset formats (YOLO, COCO, Pascal VOC, etc.)
4. **Batch Operations**: Upload and download operations support concurrent processing

## Project Configuration

- **Python Version**: 3.8+
- **Main Dependencies**: See `requirements.txt`
- **Entry Point**: `roboflow=roboflow.roboflowpy:main`
- **Code Style**: Enforced by ruff with Google docstring convention
- **Type Checking**: mypy configured for Python 3.8

## Important Notes

- API keys are stored in `~/.config/roboflow/config.json` (Unix) or `~/roboflow/config.json` (Windows)
- The SDK supports both hosted inference (Roboflow platform) and local inference (via Roboflow Inference)
- Pre-commit hooks automatically run formatting and linting checks
- Test files intentionally excluded from linting: `tests/manual/debugme.py`