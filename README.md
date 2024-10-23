# pydrostat

pydrostat is a package for simulating and controlling muscular hydrostats, an biological structure which moves without the use of skeletal support such as tongues, elephant trunks, and octopus arms.


# Dev Guide
In order to run unit tests, install the package in editable mode using the following command in the top level directory.

`pip install -e .`

Tests can then be run by using the command `pytest` in the terminal. The test directory, `tests` should mirror the package directory `src>pydrostat` where each module's tests are located in a test file `test_<module_name>.py`.