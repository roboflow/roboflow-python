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

with open("requirements.txt") as fh:
    install_requires = [
        line.strip() for line in fh.read().split("\n") if line.strip() and not line.strip().startswith("#")
    ]

setuptools.setup(
    name="roboflow",
    version=version,
    author="Roboflow",
    author_email="support@roboflow.com",
    description="Official Python package for working with the Roboflow API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/roboflow-ai/roboflow-python",
    install_requires=install_requires,
    packages=find_packages(exclude=("tests",)),
    # create optional [desktop]
    extras_require={
        # Keep OpenCV mandatory in desktop extra across versions
        "desktop": [
            "opencv-python>=4.10.0.84",
        ],
        "dev": [
            "mypy",
            "responses",
            "ruff",
            "twine",
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
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
