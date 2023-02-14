 # Development Log:
  - Feb 9th (Hong-Ye Hu):
  1. Add `stabilizer.postselection()` method
  2. Add `MeasureLayer()` class. This will support mid-circuit measurement
  3. Add `Circuit()` class. If there is no `MeasurementLayer`, `self.unitary=True`, and it should be the same as the `CliffordCircuit` class. If measurement is added, it is not unitary. `forward(obj)` will perform circuit unitary $U$ and measurements, and `backward()` will perform $U^{\dagger}$ and `postselection` on the states.
  4. Add pyclifford quantum chemistry interface. The code is in `qchem.py` file. With PySCF and Qiskit, user can generate quantum chemistry Hamiltonian in pyclifford format, and calculated mean-field (Hatree-Fock) energy and exact energy.


 # To Do List:
 1. Check `Circuit()` is correct and replace `CliffordCircuit`.
 2. Currently there is no conflict detection for CliffordGate class. So users can assign conflicting Clifford map by using both `gate.set_generator()` and `gate.set_forward_map()`. We need to add a conflict detection. (YZYou: there is a priority that when the gate implements the unitary transformation, generator will be used first, otherwise, clifford map.)
 3. Speed up `expectation()` method: possible solution: 1) njit(parallel) 2) use Heisenberg evolution, and calculate expetation of Pauli strings in the zero state 3) test performance of Nvidia `cuNumerics`, to change numpy arrays.
