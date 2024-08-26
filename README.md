<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="100%"
        src="https://github.com/roboflow/roboflow-python/assets/37276661/528ed065-d5ac-4f9a-942e-0d211b8d97de"
      >
    </a>
  </p>

  <br>

  [notebooks](https://github.com/roboflow/notebooks) | [inference](https://github.com/roboflow/inference) | [autodistill](https://github.com/autodistill/autodistill) | [collect](https://github.com/roboflow/roboflow-collect)  | [supervision](https://github.com/roboflow/supervision)

  <br>

  [![version](https://badge.fury.io/py/roboflow.svg)](https://badge.fury.io/py/roboflow)
  [![downloads](https://img.shields.io/pypi/dm/roboflow)](https://pypistats.org/packages/roboflow)
  [![license](https://img.shields.io/pypi/l/roboflow)](https://github.com/roboflow/roboflow-python/blob/main/LICENSE.md)
  [![python-version](https://img.shields.io/pypi/pyversions/roboflow)](https://badge.fury.io/py/roboflow)

</div>

# Roboflow Python Package

[Roboflow](https://roboflow.com) provides everything you need to build and deploy computer vision models. `roboflow-python` is the official Roboflow Python package. `roboflow-python` enables you to interact with models, datasets, and projects hosted on Roboflow.

With this Python package, you can:

1. Create and manage projects;
2. Upload images, annotations, and datasets to manage in Roboflow;
3. Start training vision models on Roboflow;
4. Run inference on models hosted on Roboflow, or Roboflow models self-hosted via [Roboflow Inference](https://github.com/roboflow/inference), and more.

The Python package is documented on the [official Roboflow documentation site](https://docs.roboflow.com/api-reference/introduction). If you are developing a feature for this Python package, or need a full Python library reference, refer to the [package developer documentation](https://roboflow.github.io/roboflow-python/).

## 💻 Installation

You will need to have `Python 3.8` or higher set up to use the Roboflow Python package.

Run the following command to install the Roboflow Python package:

```bash
pip install roboflow
```

For desktop features, use:

```bash
pip install "roboflow[desktop]"
```


<details>
  <summary>Install from source</summary>

  You can also install the Roboflow Python package from source using the following commands:

  ```bash
  git clone https://github.com/roboflow-ai/roboflow-python.git
  cd roboflow-python
  python3 -m venv env
  source env/bin/activate
  pip install .
  ```
</details>

<details>
  <summary>Command line tool</summary>

  By installing roboflow python package you can use some of its functionality in the command line (without having to write python code).
  See [CLI-COMMANDS.md](CLI-COMMANDS.md)
</details>


## 🚀 Getting Started

To use the Roboflow Python package, you first need to authenticate with your Roboflow account. You can do this by running the following command:

```python
import roboflow
roboflow.login()
```

<details>
<summary>Authenticate with an API key</summary>

You can also authenticate with an API key by using the following code:

```python
import roboflow

rf = roboflow.Roboflow(api_key="")
```

[Learn how to retrieve your Roboflow API key](https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key).

</details>

## Quickstart

Below are some common methods used with the Roboflow Python package, presented concisely for reference. For a full library reference, refer to the [Roboflow API reference documentation](https://docs.roboflow.com/api-reference).

```python
import roboflow

roboflow.login()

rf = roboflow.Roboflow()

# create a project
rf.create_project(
    project_name="project name",
    project_type="project-type",
    license="project-license" # "private" for private projects
)

workspace = rf.workspace("WORKSPACE_URL")
project = workspace.project("PROJECT_URL")
version = project.version("VERSION_NUMBER")

# upload a dataset
workspace.upload_dataset(
    dataset_path="./dataset/",
    num_workers=10,
    dataset_format="yolov8", # supports yolov8, yolov5, and Pascal VOC
    project_license="MIT",
    project_type="object-detection"
)

# upload model weights
version.deploy(model_type="yolov8", model_path=f”{HOME}/runs/detect/train/”)

# upload model weights - yolov10
# Before attempting to upload YOLOv10 models install ultralytics like this:
# pip install git+https://github.com/THU-MIG/yolov10.git
version.deploy(model_type="yolov10", model_path=f”{HOME}/runs/detect/train/”, filename="weights.pt")

# run inference
model = version.model

img_url = "https://media.roboflow.com/quickstart/aerial_drone.jpeg"

predictions = model.predict(img_url, hosted=True).json()

print(predictions)
```

## Library Structure

The Roboflow Python library is structured using the same Workspace, Project, and Version ontology that you will see in the Roboflow application.

```python
import roboflow

roboflow.login()

rf = roboflow.Roboflow()

workspace = rf.workspace("WORKSPACE_URL")
project = workspace.project("PROJECT_URL")
version = project.version("VERSION_NUMBER")
```

The workspace, project, and version parameters are the same as those you will find in the URL addresses at app.roboflow.com and universe.roboflow.com.

Within the workspace object you can perform actions like making a new project, listing your projects, or performing active learning where you are using predictions from one project's model to upload images to a new project.

Within the project object, you can retrieve metadata about the project, list versions, generate a new dataset version with preprocessing and augmentation settings, train a model in your project, and upload images and annotations to your project.

Within the version object, you can download the dataset version in any model format, train the version on Roboflow, and deploy your own external model to Roboflow.

## 🏆 Contributing

We would love your input on how we can improve the Roboflow Python package! Please see our [contributing guide](https://github.com/roboflow/roboflow-python/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!

<br>

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
