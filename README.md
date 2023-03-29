# Roboflow Python

---
![roboflow logo](https://media.roboflow.com/homepage/cv_pipeline_compact.png?updatedAt=1679939317160)


  <div align="center">
      <a href="https://youtube.com/roboflow">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/youtube.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949634652"
            width="3%"
          />
      </a>
      <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
      <a href="https://roboflow.com">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/roboflow-app.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949746649"
            width="3%"
          />
      </a>
      <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
      <a href="https://www.linkedin.com/company/roboflow-ai/">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/linkedin.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949633691"
            width="3%"
          />
      </a>
      <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
      <a href="https://docs.roboflow.com">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/knowledge.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949634511"
            width="3%"
          />
      </a>
      <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
      <a href="https://disuss.roboflow.com">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/forum.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949633584"
            width="3%"
          />
      <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
      <a href="https://blog.roboflow.com">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/blog.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949633605"
            width="3%"
          />
      </a>
      </a>
  </div>

  <br>

**Roboflow** streamlines your computer vision pipeline - upload data, label it, download datasets, train models, deploy models, and repeat.

The **Roboflow Python Package** is a python wrapper around the core Roboflow web application and REST API.

We also maintain an open source set of CV utililities and notebook tutorials in Python:

* :fire: https://github.com/roboflow/supervision :fire:
* :fire: https://github.com/roboflow/notebooks :fire:

## Installation

To install this package, please use `Python 3.6` or higher.

Install from PyPi (Recommended):

```bash
pip install roboflow
```

Install from Source:

```bash
git clone https://github.com/roboflow-ai/roboflow-python.git
cd roboflow-python
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```

## Authentication

```python
import roboflow
roboflow.login()
```

## Quickstart

### Datasets

Download any of over 200,000 public computer vision datasets from [Roboflow Universe](universe.roboflow.com). Label and download your own datasets on app.roboflow.com.

```python
import roboflow
dataset = roboflow.download_dataset(dataset_url="universe.roboflow.com/...", model_format="yolov8")
#ex. dataset = roboflow.download_dataset(dataset_url="https://universe.roboflow.com/joseph-nelson/bccd/dataset/1", model_format="yolov8")
print(dataset.location)
```

### Models

Predict with any of over 50,000 public computer vision models. Train your own computer vision models on app.roboflow.com or train upload your model from open source models - see https://github.com/roboflow/notebooks

```python
img_url = "https://media.roboflow.com/quickstart/aerial_drone.jpeg?updatedAt=1678743716455"
universe_model_url = "https://universe.roboflow.com/brad-dwyer/aerial-solar-panels/model/6"

model = roboflow.load_model(model_url=universe_model_url)
pred = model.predict(img_url, hosted=True)
pred.plot()
```

## Library Structure

The Roboflow python library is structured by the core Roboflow application objects.

Workspace (workspace.py) --> Project (project.py) --> Version (version.py)

```python
from roboflow import Roboflow
rf = Roboflow()
workspace = rf.workspace("WORKSPACE_URL")
project = workspace.project("PROJECT_URL")
version = project.version("VERSION_NUMBER")
```

The workspace, project, and version parameters are the same that you will find in the URL addresses at app.roboflow.com and universe.roboflow.com.

Within the workspace object you can perform actions like making a new project, listing your projects, or performing active learning where you are using predictions from one project's model to upload images to a new project.

Within the project object, you can retrieve metadata about the project, list versions, generate a new dataset version with preprocessing and augmentation settings, train a model in your project, and upload images and annotations to your project.

Within the version object, you can download the dataset version in any model format, train the version on Roboflow, and deploy your own external model to Roboflow.

## Contributing

If you want to extend our Python library or if you find a bug, please open a PR!

Also be sure to test your code the `unittest` command at the `/root` level directory.

Run tests:

```bash
python -m unittest
```

When creating new functions, please follow the [Google style Python docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html). See example below:

```python
def example_function(param1: int, param2: str) -> bool:
    """Example function that does something.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        The return value. True for success, False otherwise.

    """
```

We provide a `Makefile` to format and ensure code quality. **Be sure to run them before creating a PR**.

```bash
# format code with `black` and `isort`
make style

# check code with flake8
make check_code_quality
```

**Note** These tests will be run automatically when you commit thanks to git hooks.