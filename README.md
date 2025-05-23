<img src="/doc/logo.png" alt="Alt text" height="80" width="320">

[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=for-the-badge)](http://unitary.fund)


[![license](https://img.shields.io/badge/license-New%20BSD-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)  [![version](https://img.shields.io/badge/version-0.1.1-green.svg)](https://semver.org)

# About

This is a `python` based Clifford circuit simulation package that not only offers fast simulation but also supports analytical-level manipulation of Pauli operators and stabilizer states. And we are working on quantum circuit (strong/weak) simulations that include a few T-gates. To get started, in the [examples](https://github.com/hongyehu/PyClifford/tree/main/examples) folder (or older version [PyClifford Jupyter Book](https://hongyehu.github.io/PyCliffordPages/intro.html)), we have made several Jupyter notebooks to illustrate the basics of PyClifford. It is very intuitive to use and get started!

<!-- Installation of PyClifford also installs `TorchClifford`, a parallel package that contains the same functionality but in a vectorized and GPU-friendly format. To use TorchClifford, simply `import torchclifford` rather than `import pyclifford`. -->


Also, there are several application show cases below that are helpful.

## 🛠️ Installation 🛠️

You can install **PyClifford** locally by following these simple steps:

### 1. Set up your Python environment

It’s recommended to use a virtual environment to avoid dependency conflicts.

<details>
<summary><strong>Using conda (recommended)</strong></summary>

```bash
conda create --name pyclifford python=3.11
conda activate pyclifford
```
</details>

<details>
<summary><strong>Using venv (if not using conda)</strong></summary>
 
```bash
python3 -m venv pyclifford-env
source pyclifford-env/bin/activate
```
</details>

###  2. Install dependencies

Make sure you have the following packages installed:
```bash
pip install numpy matplotlib numba
```
You may also need to install a compatible version of setuptools:
```bash
pip install setuptools==68.2.2
```

### 3. Clone and install PyClifford

```bash
git clone https://github.com/hongyehu/PyClifford.git
cd PyClifford
pip install .
```

### 4. Test the installation

Open a Python shell and try:
```bash
import pyclifford as pc
print(pc.__version__)
```

Suppose the version is >=**v0.1.1** ; then, it is the updated version. See the release notes later for the changes.
And now try:
```bash
import pyclifford as pc
print(pc.pauli("X"))
```
You should see output corresponding to the Pauli X operator! Congratulations! 🎉

## :fire: New Release (v.0.1.1) :fire:

**We're excited to introduce our latest feature!:** :sparkles: 
- A new `examples` folder containing step-by-step examples.

## 🧪 Simple first step: Creating a GHZ State 🧪

Here's how to create a 4-qubit GHZ state using `PyClifford`:

```python
import pyclifford as pc

# Create a 4-qubit quantum circuit
N = 4
circ = pc.Circuit()

# Apply gates: H on qubit 0, then a chain of CNOTs
circ.append(pc.H(0))
circ.append(pc.CNOT(0, 1))
circ.append(pc.CNOT(1, 2))
circ.append(pc.CNOT(2, 3))

# Initialize the |0000⟩ state
state = pc.zero_state(N)

# Apply the circuit
state = circ.forward(state)

# Print the final state
print("State after circuit:", state)
```

Or one can simply do

```python
import pyclifford as pc

# Create a 4-qubit quantum circuit
N = 4
circ = pc.Circuit(pc.H(0),pc.CNOT(0, 1),pc.CNOT(1, 2),pc.CNOT(2, 3))

# Initialize the |0000⟩ state
state = pc.zero_state(N)

# Apply the circuit
state = circ.forward(state)

# Print the final state
print("State after circuit:", state)
```


Now, let's do a **mid-circuit** measurement!

```python
import pyclifford as pc

# Create a 4-qubit quantum circuit
N = 4
circ = pc.Circuit()

# Apply gates: H on qubit 0, then a chain of CNOTs
circ.append(pc.H(0))
circ.append(pc.CNOT(0, 1))
circ.append(pc.Measurement(1)) # measure qubit-1 in z-basis
circ.append(pc.H(1))
circ.append(pc.CNOT(1, 2))
circ.append(pc.CNOT(2, 3))

# Initialize the |0000⟩ state
state = pc.zero_state(N)

# Apply the circuit
state = circ.forward(state)

# Print the final state
print("State after circuit:", state)
# Print mid-circuit measurement result
print("mid-circuit measurement:", circ.measure_result)
```

## 💬 Quotes from our first few users 💬:
 - Prof. Zhen Bi (Pappalardo Fellow 17'@MIT, Assistant Professor@PennState): 
  > "*PyClifford is an exceptional tool that offers researchers in quantum condensed matter a wide range of capabilities, including an intuitive programming language for simulating and analyzing Clifford circuits, quantum measurement, and evaluation of entanglement quantities, all of which are crucial in advancing our understanding of the quantum world. Its continuous updates and enhancements by a well-coordinated team of experts make it a reliable and powerful resource that can keep pace with the latest research developments and drive new discoveries in the field.*"

## 🚀 Application show cases 🚀:
<img src="/doc/show_cases.png" alt="Alt text" height="400" width="560">

(We are still working on making detailed examples in those Jupyter notebooks)
 1. Solving a strong disordered Hamiltonian by Spectrum Bifurcation Renormalization Group (*Condensed Matter Physics*) [Link](/doc/SBRG.ipynb)
 2. Measurement induced phase transition (*Condensed Matter Physics/Quantum Error Correction*) [Link](/dev/demo-MIPT.ipynb)
 3. Classical shadow tomography with shallow layers (*Quantum Computation & Information*) [Link](/dev/demo-CST.ipynb)
 4. Clifford ansatz for quantum chemistry (*Quantum Computation & Quantum Chemistry*) [Link](/dev/demo-QChem.ipynb)

## 🧱 Structure of `PyClifford` 🧱:
The structure of `PyClifford` is illustrated below. Pauli strings, stabilizer states, and Clifford maps are represented by binary or integer strings. All the low-level calculation is in the `utils.py` with JIT compliation. Then `paulialg.py` handles all the Pauli algebra and manipulation of Pauli string lists. On top of that, we have built `stabilizer.py` to handle stabilizer states and Clifford maps. Finally, `circuit.py` gives user an easy access to all the functions.

In addition, we are interested in developing `PyCliffordExt` as an extension to `PyClifford`, where we would like to include few-T gate into the package. If you are interested in its physics or contributing to the code, please feel free to [contact us](https://scholar.harvard.edu/hongyehu/home)!

<img src="/doc/structure_of_code.png" alt="Alt text" height="280" width="399">

## 🧩 Dependence of `PyClifford` 🧩:
- Numba
- Numpy
- Matplotlib
### Dependence of `PyCliffordExt`:
- Qiskit 0.39.4
- Pyscf 2.0.1

## Release Note:
<details>
<summary><strong>📦 Major Update: v0.1.1 (May 2025)</strong></summary>
<br>
🚀 Major Update Summary
 
- Unified circuit architecture to support both unitary and measurement operations, and more flexible circuit class to compose any type of operation.
- Unified measurement and post-selection framework using `log2` scale for stability.
- Removed `QuTiP` dependency in favor of pure NumPy implementations.
- Added caching for measurements and random unitaries to support classical shadows.
🗂 File-by-File Updates
<br>
### `__init__.py`

- Refactored imports and renamed core classes: `Circuit`, `Layer`, `Measurement`.
- Removed obsolete classes: `CliffordCircuit`, `MeasurementLayer`.
- Added new gate primitives: `SWAP`, `CZ`, `CX`, `measurement_layer`.
<br>
### `utils.py`

- Introduced `stabilizer_postselect`, unified treatment of projection and post-selection with `log2prob`.
- Removed deprecated methods: `stabilizer_projection_trace`, `stabilizer_postselection`, and `decompose`.
- Explained the encoder-decoder logic for Pauli decomposition using `.transform_by(state.to_map().inverse())`.
<br>
### `paulialg.py`

- Removed `qutip` dependency and replaced with `to_numpy` methods.
- Vectorized `PauliList.to_numpy` for efficiency.
- Constructors now accept `**kwargs` to support inheritance.
<br>
### `stabilizer.py`

- Fixed `.copy()` issue by improving constructor signature handling `r`.
- Removed `.set_r()`, now set `r` via constructor or direct assignment.
- Standardized error handling with Python exceptions.
- Unified implementation of `.expect()` and `.get_prob()` using `stabilizer_postselect`.
- Introduced fugacity for reweighting Pauli observable expectations.
<br>
### `circuit.py`

- Major refactor of `Circuit` and `Layer` classes to support measurements and flexible composition.
- Replaced `.take()` with `.append()`.
- `forward` and `backward` now return `(state, log2prob)` tuple uniformly.
- Introduced `Measurement` class with cached outcomes, enabling classical shadows via forward/backward structure.
- Removed `.N` and constructor no longer requires qubit count.
- Added `.reset()` methods to clear cached values.
- Random unitary caching in `CliffordGate` to support deterministic inverses in backward pass.
</details>


## 🔮 What will be in the next release (June-July 2025) 🔮: 
- Generalized stabilizer states
- Noisy circuit simulation of classical shadows
- One-line calculation of Pauli weight used for classical post-processing. (Tensor network-based algorithm)


<!--**For MacOS user:** you can create a virtual environment containing necessary dependences with `conda env create -f env/miniClifford.yml`-->

## 📚 Citation 📚

We are currently preparing a documentation paper for **PyClifford**, expected in **July 2025**. In the meantime, if you use this package in your research, please consider citing the related works that contributed to its development:

- **Topological and symmetry-enriched random quantum critical points**  
  Carlos M. Duque, Hong-Ye Hu, Yi-Zhuang You, Vedika Khemani, Ruben Verresen, Romain Vasseur  
  *Phys. Rev. B 103, L100207 (2021)*  
  [DOI: 10.1103/PhysRevB.103.L100207](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.103.L100207)

- **Classical shadow tomography with locally scrambled quantum dynamics**  
  Hong-Ye Hu, Soonwon Choi, Yi-Zhuang You  
  *Phys. Rev. Research 5, 023027 (2023)*  
  [DOI: 10.1103/PhysRevResearch.5.023027](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.5.023027)

- **Demonstration of robust and efficient quantum property learning with shallow shadows**  
  Hong-Ye Hu, Andi Gu, Swarnadeep Majumder, Hang Ren, Yipei Zhang, Derek S. Wang, Yi-Zhuang You, Zlatko Minev, Susanne F. Yelin, Alireza Seif  
  *Nature Communications 16, 2943 (2025)*  
  [DOI: 10.1038/s41467-025-57349-w](https://www.nature.com/articles/s41467-025-57349-w)

## Changes
We are planning to move **TorchClifford** to a separate package and keep **PyClifford** lightweight, as users' suggestions. For users of **TorchClifford**, please keep an eye for the update.
