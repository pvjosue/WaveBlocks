import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Waveblocks",
    version="0.1.0",
    author="Josue Page Vizcaino, Josef Kamysek, Erik Riedel",
    author_email="pv.josue@gmail.com, josef@kamysek.com, erik.riedel@tum.de",
    description="Waveblocks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pvjosue/WaveBlocks",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
)
