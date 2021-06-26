[![Build Status](https://travis-ci.com/Trave11er/RCWA.svg?branch=master)](https://travis-ci.com/Trave11er/RCWA)
[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)

# RCWA
Simple implementation of transfer matrix method (TMM) and
rigorous coupled wave analysis (RCWA) based on the notes from [Computational Electromagnetics course by Raymond Rumpf](https://empossible.net/academics/emp5337/) ([formerly here](http://emlab.utep.edu/ee5390cem.htm))


## Getting Started
Open a terminal in the root rcwa directory and execute
```
python setup.py install
```
Optionally, run the test suite by executing
```
pytest
```
To run a tmm computation execute
```
tmm path-to-input-toml-file
```
Analogously, for a rcwa run
```
rcwa path-to-input-toml-file
```
which will read the provided input files in .toml format.
### Dependencies

python, numpy, toml, pytest

## Authors

* **Gleb Siroki**

## License

This project is licensed under the MIT License - see the LICENSE.md file for details
