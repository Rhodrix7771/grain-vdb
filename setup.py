from setuptools import setup, find_packages

setup(
    name="grainvdb",
    version="1.0.0",
    author="Adam Sussman",
    description="Metal-Core Vector Intelligence Engine for Apple Silicon",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "sanic",
        "orjson",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Topic :: Database :: Database Engines",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
)
