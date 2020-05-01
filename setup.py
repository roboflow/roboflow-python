import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="roboflow", # Replace with your own username
    version="0.0.1",
    author="Roboflow",
    author_email="devs@roboflow.ai",
    description="Loader for Roboflow datasets.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/roboflow-ai/roboflow-python",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 1 - Planning",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Recognition"
    ],
    keywords="roboflow datasets dataset download convert annotation annotations computer vision object detection classification",
    python_requires='>=3.6',
)
