# Roboflow Python Library

---
![roboflow logo](https://i.imgur.com/lXCoVt5.png)

[Website](https://docs.roboflow.com/python) • [Docs](https://docs.roboflow.com/python) • [Blog](https://blog.roboflow.com)
• [Twitter](https://twitter.com/roboflow) • [Linkedin](https://www.linkedin.com/company/roboflow-ai)
• [Universe](https://universe.roboflow.com)

**Roboflow** makes managing, preprocessing, augmenting, and versioning datasets for computer vision seamless. This is
the official Roboflow python package that interfaces with the [Roboflow API](https://docs.roboflow.com). Key features of
Roboflow:

- Import and Export image datasets into any supported [format](https://roboflow.com/formats)
- [Preprocess](https://docs.roboflow.com/image-transformations/image-preprocessing)
  and [augment](https://docs.roboflow.com/image-transformations/image-augmentation) data using Roboflow's dataset
  management tools
- Train computer vision models using [Roboflow Train](https://docs.roboflow.com/train) and deploy
  to [production](https://docs.roboflow.com/inference)
- Use [community curated projects](https://universe.roboflow.com/) to start building your own vision-powered products

## Installation

To install this package, please use `Python 3.6` or higher. We provide three different ways to install the Roboflow
package to use within your own projects.

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

## Quickstart

```python
import roboflow

# Instantiate Roboflow object with your API key
rf = roboflow.Roboflow(api_key=YOUR_API_KEY_HERE)

# List all projects for your workspace
workspace = rf.workspace()

# Load a certain project (workspace url is optional)
project = rf.project("PROJECT_ID")

# List all versions of a specific project
project.versions()

# Upload image to dataset
project.upload("UPLOAD_IMAGE.jpg")

# Retrieve the model of a specific project
model = project.version("1").model

# predict on a local image
prediction = model.predict("YOUR_IMAGE.jpg")

# Predict on a hosted image
prediction = model.predict("YOUR_IMAGE.jpg", hosted=True)

# Plot the prediction
prediction.plot()

# Convert predictions to JSON
prediction.json()

# Save the prediction as an image
prediction.save(output_path='predictions.jpg')
```

## Using this package for a specific project

If you have a specific project from your workspace you'd like to run in a notebook, follow along with this tutorial: [Downloading Datasets from Roboflow for Training (Python)](https://www.youtube.com/watch?v=76E6esnez8E).

Selecting the format you'd like your project to be exported as while choosing the `show download code` option will display code snippets you can use in either Jupyter or your terminal. These code snippets will include your `api_key`, project, and workspace names.

![Alt Text](https://media.giphy.com/media/I5g06mUnVzdX7iT2Gf/giphy.gif)

## Developing locally

### Using Docker

To set the Docker container up for the first time:

```bash
# Clone this repo
git clone git@github.com:roboflow-ai/roboflow-python.git && cd roboflow-python

# Copy the environment variables template
# Be sure to update the values with your account's information

# Build our development image
docker build -t roboflow-python -f Dockerfile.dev .

# Run container and map current folder in it
docker run --rm -it \
  -v $(pwd)/:/workspace/ \
  --env-file .env \
  roboflow-python

# Run tests
python -m unittest
```

#### Change Python version

You can pass the build arg `PYTHON_VERSION` to dynamically change python version at build time

```bash
docker build  -t roboflow-python --build-arg PYTHON_VERSION=3.9 -f Dockerfile.dev .
```

Will use `python:3.9-slim`

**Note** If you are using [VSCode](https://code.visualstudio.com/) we recommend you read the ["Developing inside a Container"](https://code.visualstudio.com/docs/remote/containers) tutorial.

### Using Virtualenv

```bash
# Clone this repo
git clone git@github.com:roboflow-ai/roboflow-python.git && cd roboflow-python

# create virtual env
virtualenv local_dev

# activate virtual env
source local_dev/bin/activate

# install dependencies
pip3 install -e ".[dev]"
```

### Testing

Make sure you have your `virtualenv` spun up before running tests. Execute the `unittest` command at the `/root` level directory.

Run tests:

```bash
python -m unittest
```

### Contributing

1. Increment the pip package minor version number in `setup.py`
1. Manually add any new dependencies to `requirements.txt` with a version such as `chardet==4.0.0` and list of dependencies in `setup.py` (Be careful not to overwrite any packages that might screw up backwards dependencies for object detection, etc.)

### Code Quality

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
