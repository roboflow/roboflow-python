import re

import setuptools
from setuptools import find_packages

with open("./roboflow/__init__.py") as f:
    content = f.read()
_search_version = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', content)
assert _search_version
version = _search_version.group(1)


with open("README.md") as fh:
    long_description = fh.read()

with open("requirements-slim.txt") as fh:
    install_requires = fh.read().split("\n")

setuptools.setup(
    name="roboflow-slim",
    version=version,
    author="Roboflow",
    author_email="support@roboflow.com",
    description="Lightweight Roboflow SDK for vision events, workspace management, and CLI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/roboflow-ai/roboflow-python",
    install_requires=install_requires,
    packages=find_packages(exclude=("tests",)),
    extras_require={
        "dev": [
            "mypy",
            "responses",
            "ruff",
            "types-pyyaml",
            "types-requests",
            "types-setuptools",
            "types-tqdm",
            "wheel",
        ],
    },
    entry_points={
        "console_scripts": [
            "roboflow=roboflow.roboflowpy:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
