import setuptools
from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="roboflow",  # Replace with your own username
    version="0.1.3",
    author="Palash Shah",
    author_email="ps9cmk@virginia.edu",
    description="Ergonomic machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/roboflow-ai/roboflow-python",
    install_requires=[
        "certifi==2021.5.30",
        "chardet==4.0.0",
        "cycler==0.10.0",
        "idna==2.10",
        "kiwisolver==1.3.1",
        "matplotlib",
        "numpy==1.19.5",
        "opencv-python==4.5.3.56",
        "pillow",
        "pyparsing==2.4.7",
        "python-dateutil",
        "python-dotenv",
        "requests==2.25.1",
        "six",
        "urllib3==1.26.6",
        "wget",
        "tqdm",
        "pyyaml",
        "wget"
    ],
    packages=find_packages(exclude=('tests',)),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)