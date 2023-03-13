import setuptools
from setuptools import find_packages
import re

with open("./roboflow/__init__.py", "r") as f:
    content = f.read()
    # from https://www.py4u.net/discuss/139845
    version = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', content).group(1)


with open("README.md", "r") as fh:
    long_description = fh.read()

# we are using the packages in `requirements.txt` for now, 
# not 100% ideal but will do
with open("requirements.txt", "r") as fh:
    install_requires = fh.read().split('\n')

setuptools.setup(
    name="roboflow",
    version=version,
    author="Roboflow",
    author_email="jacob@roboflow.com",
    description="python client for the Roboflow application",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/roboflow-ai/roboflow-python",
    install_requires=install_requires,
    packages=find_packages(exclude=("tests",)),
    extras_require={
        "dev": ["flake8", "black==22.3.0", "isort", "responses", "twine", "wheel"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
