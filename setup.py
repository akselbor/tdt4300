import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="tdt4300", # Replace with your own username
    version="0.0.1",
    author="Aksel Borgen",
    author_email="akselborgen@outlook.com",
    description="Helper library used for the TDT4300 exam",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/akselbor/tdt4300",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements,
    python_requires='>=3.6',
)