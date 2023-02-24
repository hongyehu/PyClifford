<img src="/doc/logo.png" alt="Alt text" height="80" width="320">

[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=for-the-badge)](http://unitary.fund)


[![license](https://img.shields.io/badge/license-New%20BSD-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)  [![version](https://img.shields.io/badge/version-0.1.0-green.svg)](https://semver.org)

# About

This is a `python` based clifford circuit simulation package which not only offers the fast simulation but also supports analytical level manipulation of pauli operators and stabilizer states. And we are working on quantum circuit (strong/weak) simulations that include a few T-gates. To get started, in [PyClifford Jupyter Book](https://hongyehu.github.io/PyCliffordPages/intro.html), we have made several jupyter notebooks to illustrate the basics of PyClifford.

Also, there are several application shows cases below that are helpful.

## Application show cases:
<img src="/doc/show_cases.png" alt="Alt text" height="400" width="570">

(We are still working on making detailed examples in those jupyter notebooks)
 1. Solving strong disordered Hamiltonian by Spectrum Bifurcation Renormalization Group (*Condensed Matter Physics*) [Link](/doc/SBRG.ipynb)
 2. Measurement induced phase transition (*Condensed Matter Physics/Quantum Error Correction*) [Link](/dev/demo-MIPT.ipynb)
 3. Classical shadow tomography with shallow layers (*Quantum Computation & Information*) [Link](/dev/demo-CST.ipynb)
 4. Clifford ansatz for quantum chemistry (*Quantum Computation & Quantum Chemistry*) [Link](/dev/demo-QChem.ipynb)

## Structure of `PyClifford`:
The structure of `PyClifford` is illustrated below. Pauli strings, stabilizer states, and Clifford maps are represented by binary or integer strings. All the low-level calculation is in the `utils.py` with JIT compliation. Then `paulialg.py` handles all the Pauli algebra and manipulation of Pauli string lists. On top of that, we have built `stabilizer.py` to handle stabilizer states and Clifford maps. Finally, `circuit.py` gives user an easy access to all the functions.

In addition, we are interested in developing `PyCliffordExt` as an extension to `PyClifford`, where we would like to include few-T gate into the package. If you are interested in its physics or contributing to the code, please feel free to [contact us](https://scholar.harvard.edu/hongyehu/home)!

<img src="/doc/structure_of_code.png" alt="Alt text" height="280" width="399">

## Dependence of `PyClifford`:
- Numba
- Numpy
- QuTip
- Matplotlib
### Dependence of `PyCliffordExt`:
- Qiskit 0.39.4
- Pyscf 2.0.1



<!--**For MacOS user:** you can create a virtual environment containing necessary dependences with `conda env create -f env/miniClifford.yml`-->




