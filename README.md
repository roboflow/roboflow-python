# Roboflow Python

---
![roboflow logo](https://media.roboflow.com/homepage/cv_pipeline_compact.png?updatedAt=1679939317160)

[Roboflow](https://roboflow.com) provides everything you need to build and deploy computer vision models. `roboflow-python` is the official Roboflow Python package. `roboflow-python` enables you to interact with models, datasets, and projects hosted on Roboflow.

With this Python package, you can:

1. Create and manage projects;
2. Upload images, annotations, and datasets to manage in Roboflow;
3. Start training vision models on Robfolow;
4. Run inference on models hosted on Roboflow, or Roboflow models self-hosted via [Roboflow Inference](https://github.com/roboflow/inference), and more.

## 💻 Installation

You will need to have `Python 3.6` or higher set up to use the Roboflow Python package.

Run the following command to install the Roboflow Python package:

```bash
pip install roboflow
```

<details>
  <summary>Install from source</summary>

  You can also install the Roboflow Python package from source using the following commands:

  ```bash
  git clone https://github.com/roboflow-ai/roboflow-python.git
  cd roboflow-python
  python3 -m venv env
  source env/bin/activate
  pip3 install -r requirements.txt
  ```
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

### Datasets

Download any of over 200,000 public computer vision datasets from [Roboflow Universe](universe.roboflow.com). Label and download your own datasets on app.roboflow.com.

```python
dataset = roboflow.download_dataset(dataset_url="universe.roboflow.com/...", model_format="yolov8")
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

The Roboflow Python library is structured by the core Roboflow application objects.

Workspace (workspace.py) --> Project (project.py) --> Version (version.py)

```python
import roboflow

roboflow.login()

rf = roboflow.Roboflow()

workspace = rf.workspace("WORKSPACE_URL")
project = workspace.project("PROJECT_URL")
version = project.version("VERSION_NUMBER")
```

The workspace, project, and version parameters are the same that you will find in the URL addresses at app.roboflow.com and universe.roboflow.com.

Within the workspace object you can perform actions like making a new project, listing your projects, or performing active learning where you are using predictions from one project's model to upload images to a new project.

Within the project object, you can retrieve metadata about the project, list versions, generate a new dataset version with preprocessing and augmentation settings, train a model in your project, and upload images and annotations to your project.

Within the version object, you can download the dataset version in any model format, train the version on Roboflow, and deploy your own external model to Roboflow.

## 🏆 Contributing

We would love your input on how we can improve the Roboflow Python package! Please see our [contributing guide](https://github.com/roboflow/roboflow-python) to get started. Thank you 🙏 to all our contributors!

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