# pydrostat

pydrostat is a package for simulating and controlling muscular hydrostats, a biological structure which moves without the use of skeletal support such as tongues, elephant trunks, and octopus arms.

# Installation

Start by ensuring you have python 3.11 or greater and upgrade pip. Replace `XX` with the python version.

- Windows: `py -3.XX -m pip install --upgrade pip`
- macOS/Linux: `python3.XX -m pip install --upgrade pip`

Instantiate a virtual environment in the top level of the cloned repository. The python version must be at least 3.11.

- Windows: `py -3.XX -m venv venv`
- macOS: `/usr/local/opt/python@3.XX/bin/python3 -m venv venv`
- Linux: `python3.XX -m venv venv`

Instantiate the virtual environment

- Windows: `venv\Scripts\activate`
- macOS/Linux: `source venv\bin\activate`

Upgrade pip.

- Windows: `py -3.XX -m pip install --upgrade pip`
- macOS/Linux: `python3.XX -m pip install --upgrade pip`

Install dependencies

`pip install -r requirements.txt`

In order to run unit tests, install the package in editable mode using the following command in the top level directory.

`pip install -e .`

Tests can then be run by using the command `pytest` in the terminal. The test directory, `tests` should mirror the package directory `src>pydrostat` where each module's tests are located in a test file `test_<module_name>.py`.

# Dev Guide

In general, we will try to follow [Google's Python style guide](https://google.github.io/styleguide/pyguide.html).

Interfaces will have the naming convention `<x>_interface.py` with class name `I<X>`.
