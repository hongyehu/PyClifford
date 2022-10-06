<img src="/doc/logo.png" alt="Alt text" height="80" width="320">

[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=for-the-badge)](http://unitary.fund)


[![license](https://img.shields.io/badge/license-New%20BSD-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)  [![version](https://img.shields.io/badge/version-0.1.0-green.svg)](https://semver.org)

# About

This is a `python` based clifford circuit simulation package which not only offers the fast simulation but also supports analytical level manipulation of pauli operators and stabilizer states. And we are working on quantum circuit (strong/weak) simulations that include a few T-gates.

## Dependence of `PyClifford`:
- Numba
- Numpy
- QuTip
- Matplotlib

**For MacOS user:** you can create a virtual environment containing necessary dependences with `conda env create -f env/miniClifford.yml`


# To-Do and Issues:

1. Currently there is no conflict detection for CliffordGate class. So users can assign conflicting Clifford map by using both `gate.set_generator()` and `gate.set_forward_map()`. We need to add a conflict detection.

