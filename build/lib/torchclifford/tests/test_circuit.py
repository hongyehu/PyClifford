import torch
import numpy as np

from ..circuit import *
from ..stabilizer import Pauli, random_clifford_map, random_clifford_state


device = 'cpu'


def test_cliffordgate():
    qubits = np.random.randint(4, 10)
    pauli_vals = torch.tensor(np.random.randint(4, size=qubits), device=device)
    p = pauli(pauli_vals)
    clifford_map = random_clifford_map(qubits)
    inv_clifford_map = clifford_map.inverse()

    cg = CliffordGate(0, 1)
    cg.set_generator(p)
    cg.set_forward_map(clifford_map)
    cg.set_backward_map(inv_clifford_map)

    assert cg.generator.__repr__() == p.copy().__repr__()
    assert torch.allclose(cg.forward_map.gs, clifford_map.gs)
    assert torch.allclose(cg.forward_map.ps, clifford_map.ps)
    assert torch.allclose(cg.backward_map.gs, inv_clifford_map.gs)
    assert torch.allclose(cg.backward_map.ps, inv_clifford_map.ps)

    cg_copy = cg.copy()
    assert cg.generator.__repr__() == cg_copy.generator.__repr__()
    assert torch.allclose(cg.forward_map.gs, cg_copy.forward_map.gs)
    assert torch.allclose(cg.forward_map.ps, cg_copy.forward_map.ps)
    assert torch.allclose(cg.backward_map.gs, cg_copy.backward_map.gs)
    assert torch.allclose(cg.backward_map.ps, cg_copy.backward_map.ps)

    cg2 = CliffordGate(2, 3)
    assert not cg.independent_from(cg_copy)
    assert cg.independent_from(cg2)

    cg2 = CliffordGate(*tuple(range(qubits)))
    psi = random_clifford_state(qubits)
    psi_copy = psi.copy()
    cg2.set_forward_map(clifford_map)
    cg2.set_backward_map(inv_clifford_map)
    psi_copy = cg2.backward(cg2.forward(psi))
    assert torch.allclose(psi.gs, psi_copy.gs)
    assert torch.allclose(psi.ps, psi_copy.ps)

    cg2 = CliffordGate(*tuple(range(qubits)))
    cg2.set_generator(p)
    psi_copy = cg2.backward(cg2.forward(psi))
    assert torch.allclose(psi.gs, psi_copy.gs)
    assert torch.allclose(psi.ps, psi_copy.ps)

    extra_qubits = np.random.randint(1, 4)
    cg2 = CliffordGate(*tuple(range(qubits)))
    psi = random_clifford_state(qubits+extra_qubits)
    psi_copy = psi.copy()
    cg2.set_forward_map(clifford_map)
    cg2.set_backward_map(inv_clifford_map)
    psi_copy = cg2.backward(cg2.forward(psi))
    assert torch.allclose(psi.gs, psi_copy.gs)
    assert torch.allclose(psi.ps, psi_copy.ps)

    cg2 = CliffordGate(*tuple(range(qubits)))
    cg2.set_generator(p)
    psi_copy = cg2.backward(cg2.forward(psi))
    assert torch.allclose(psi.gs, psi_copy.gs)
    assert torch.allclose(psi.ps, psi_copy.ps)


def test_cliffordlayer():
    qubits = np.random.randint(4, 10)
    pauli_vals = torch.tensor(np.random.randint(4, size=qubits), device=device)
    clifford_map = random_clifford_map(qubits)
    cg = CliffordGate(*tuple(range(qubits)))
    cg.set_forward_map(clifford_map)
    pauli_vals = torch.tensor(np.random.randint(4, size=qubits), device=device)
    cg2 = CliffordGate(*tuple(range(qubits)))
    clifford_map = random_clifford_map(qubits)
    cg2.set_forward_map(clifford_map)
    cg3 = CliffordGate(qubits+1)
    clifford_map = random_clifford_map(1)
    cg3.set_forward_map(clifford_map)
    cl = CliffordLayer(cg, cg2)
    cl2 = cl.copy()
    for gate, gate2 in zip(cl.gates, cl2.gates):
        assert torch.allclose(gate.forward_map.gs, gate2.forward_map.gs)
        assert torch.allclose(gate.forward_map.ps, gate2.forward_map.ps)
    assert not cl2.independent_from(cg)
    assert cl2.independent_from(cg3)
    psi = random_clifford_state(qubits)
    psi_copy = cl.backward(cl.forward(psi))
    assert torch.allclose(psi.gs, psi_copy.gs)
    assert torch.allclose(psi.ps, psi_copy.ps)


def test_cliffordcircuit():
    qubits = np.random.randint(4, 10)
    pauli_vals = torch.tensor(np.random.randint(4, size=qubits), device=device)
    clifford_map = random_clifford_map(qubits)
    cg = CliffordGate(*tuple(range(qubits)))
    cg.set_forward_map(clifford_map)
    pauli_vals = torch.tensor(np.random.randint(4, size=qubits), device=device)
    cg2 = CliffordGate(*tuple(range(qubits)))
    clifford_map = random_clifford_map(qubits)
    cg2.set_forward_map(clifford_map)
    circ = CliffordCircuit()
    circ.take(cg)
    circ.take(cg2)
    pauli_vals = torch.tensor(np.random.randint(4, size=qubits), device=device)
    clifford_map = random_clifford_map(qubits)
    cg = CliffordGate(*tuple(range(qubits)))
    cg.set_forward_map(clifford_map)
    pauli_vals = torch.tensor(np.random.randint(4, size=qubits), device=device)
    cg2 = CliffordGate(*tuple(range(qubits)))
    clifford_map = random_clifford_map(qubits)
    cg2.set_forward_map(clifford_map)
    circ2 = CliffordCircuit()
    circ2.take(cg)
    circ2.take(cg2)

    circ_true = circ.copy()
    circ2_true = circ2.copy()
    circ_comp = circ2.compose(circ)
    psi = random_clifford_state(qubits)
    psi_forward = circ_comp.forward(psi)
    psi_true_forward = circ_true.forward(circ2_true.forward(psi))
    assert torch.allclose(psi_forward.gs, psi_true_forward.gs)
    assert torch.allclose(psi_forward.ps, psi_true_forward.ps)
    psi_true = psi.copy()
    psi = circ2.backward(circ2.forward(psi))
    assert torch.allclose(psi.gs, psi_true.gs)
