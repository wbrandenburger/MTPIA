# -*- coding: utf-8 -*-
#
# you can install this to a local test virtualenv like so:
#   virtualenv venv
#   ./venv/bin/pip install --editable .
#   ./venv/bin/pip install --editable .[dev]  # with dev requirements, too

import dldawn

import glob
import setuptools
import sys

with open("README.md") as fd:
    long_description = fd.read()

if sys.platform == "win32":
    data_files = []
else:
    data_files = []

included_packages = ["dldawn"] + ["dldawn." + p for p in setuptools.find_packages("dldawn")]

setuptools.setup(
    name="dldawn",
    version=dldawn.__version__,
    maintainer=dldawn.__maintainer__,
    maintainer_email=dldawn.__email__,
    author=dldawn.__author__,
    author_email=dldawn.__email__,
    license=dldawn.__license__,
    url="https://github.com/wbrandenburger/MTPIA",
    install_requires=[
        # - python development packages - 
        "pylint"
        # - python deep learning packages -
        "tensorflow==1.*"
        "keras==2.2.5"
        # - python evaluation packages -
        # "pyqt==4.11.4"
        "numpy"
        "matplotlib"
        "jupyter"
        "opencv-python"
        "pillow"
        "scikit-learn"
        "scikit-image"
        "scipy"
        "h5py"
        "eventlet"
        "flask-socketio"
        "seaborn"
        "pandas"
        "ffmpeg"
        # "imageio=2.1.2"
        "moviepy"
        # - python project packages - 
        "colorama>=0.4",
        "click>=7.0.0",
        "stevedore>=1.30",
        # "requests>=2.11.1",
        # "filetype>=1.0.1",
        # "pyparsing>=2.2.0",
        # "configparser>=3.0.0",
        # "arxiv2bib>=1.0.7",
        # "PyYAML>=3.12",
        # "chardet>=3.0.2",
        # "beautifulsoup4>=4.4.1",
        # "bibtexparser>=0.6.2",
        # "python-slugify>=1.2.6",
        # "habanero>=0.6.0",
        # "isbnlib>=3.9.1,<4.0.0",
        # "prompt_toolkit>=2.0.5",
        # "tqdm>=4.1",
        # "pygments>=2.2.0",
        # "python-doi>=0.1.1",
        # "lxml",
        # "validators>=0.13.0"
    ],
    python_requires=">=3",
    classifiers=[
        "Development Status :: 1 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: System Administrators",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Utilities",
    ],
    extras_require=dict(
        # List additional groups of dependencies here (e.g. development
        # dependencies). You can install these using the following syntax,
        # for example:
        # $ pip install -e .[develop]
        optional=[
        ],
        develop=[
            # "sphinx",
            # "sphinx-click",
            # "sphinx_rtd_theme",
            # "pytest",
            # "pytest-cov==2.5.0",
        ]
    ),
    description=(
        "Visualization tool for exploring remote sensing data and view processing results"
    ),
    long_description=long_description,
    keywords=[
        "visualization", "remote sensing", "images", "aerial", "satellite", "viewer", "explorer", "science", "research", "command-line", "tui"
    ],
    package_data=dict(
        dldawn=[
        ],
    ),
    data_files=data_files,
    packages=included_packages,
    entry_points={
        "console_scripts": [
            "dldawn=dldawn.commands.default:run",
        ],
    },
    platforms=["linux", "osx", "windows"],
)
