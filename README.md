# Roboflow Python Library

---
![roboflow logo](https://i.imgur.com/lXCoVt5.png)

[Website](https://roboflow.com) • [Docs](https://docs.roboflow.com) • [Blog](https://blog.roboflow.com)
• [Twitter](https://twitter.com/roboflow) • [Linkedin](https://www.linkedin.com/company/roboflow-ai)
• [Universe](https://universe.roboflow.com)

**Roboflow** makes managing, preprocessing, augmenting, and versioning datasets for computer vision seamless. This is
the official Roboflow python package that interfaces with the [Roboflow API](https://docs.roboflow.com). Key features of
Roboflow:

- Import and Export image datasets into any supported [formats](https://roboflow.com/formats)
- [Preprocess](https://docs.roboflow.com/image-transformations/image-preprocessing)
  and [augment](https://docs.roboflow.com/image-transformations/image-augmentation) data using Roboflow's dataset
  management tools
- Train computer vision models using [Roboflow Train](https://docs.roboflow.com/train) and deploy
  to [production](https://docs.roboflow.com/inference)
- Use [community curated projects](https://universe.roboflow.com/) to start building your own vision-powered products

## Installation:

To install this package, please use `Python 3.6` or higher. We provide three different ways to install the Roboflow
package to use within your own projects.

Install from Pypi (Recommended):

```
pip install roboflow
```

Install from Conda:

```
conda install roboflow
```

Install from Source:

```
git clone https://github.com/roboflow-ai/roboflow-python.git
cd roboflow-python
python3 -m venv
source venv/bin/activate 
pip3 install -r requirements.txt
```

## Quickstart

```python
import roboflow

# Authenticate Roboflow
rf = roboflow.auth("YOUR_API_KEY_HERE")

# List all projects for your workspace
workspace = rf.workspace()

# Load a certain project
project = rf.project(workspace.url, "YOUR_PROJECT NAME")

# Upload image to dataset
project.upload("UPLOAD_IMAGE.jpg")

# Choose a specific trained model from the project
model = project.models()[0]

# predict on a local image
prediction = model.predict("YOUR_IMAGE.jpg")
# Plot the prediction
prediction.plot()
# Save the prediction as an image
prediction.save(output_path='predictions.jpg')
```
