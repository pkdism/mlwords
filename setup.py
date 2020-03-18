import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mlwords",
    version="0.0.1",
    author="Pawan Dwivedi",
    author_email="pawan.dwivedi94@gmail.com",
    description="MLwords",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pkdism/mlwords",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
