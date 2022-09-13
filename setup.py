import setuptools
from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="roboflow",  # Replace with your own username
    version="0.2.15",
    author="Roboflow",
    author_email="jacob@roboflow.com",
    description="python client for the Roboflow application",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/roboflow-ai/roboflow-python",
    install_requires=[
        "certifi==2021.5.30",
        "chardet==4.0.0",
        "cycler==0.10.0",
        "glob2",
        "idna==2.10",
        "kiwisolver==1.3.1",
        "matplotlib",
        "numpy>=1.18.5",
        "opencv-python-headless>=4.5.1.48",
        "Pillow>=7.1.2",
        "pyparsing==2.4.7",
        "python-dateutil",
        "python-dotenv",
        "requests",
        "requests_toolbelt",
        "six",
        "urllib3==1.26.6",
        "wget",
        "tqdm>=4.41.0",
        "PyYAML>=5.3.1",
        "wget",
    ],
    packages=find_packages(exclude=("tests",)),
    extras_require={
        "dev": ["flake8", "black==22.3.0", "isort"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
