import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Waveblocks",
    version="0.0.1",
    author="Josue Page Vizcaino, Josef Kamysek, Erik Riedel",
    author_email="pv.josue@gmail.com, josef@kamysek.com, erik.riedel@tum.de",
    description="Waveblocks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.lrz.de/IP/WaveBlocks",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
)
