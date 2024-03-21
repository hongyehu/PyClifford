import torch
import numpy as np
import scipy
import qutip

from ..stabilizer import *
from ..paulialg import pauli, paulis, Pauli, PauliList, PauliPolynomial


device = 'cpu'
#device = 'cuda:0'


iden = np.array([[1, 0], [0, 1]])
z = np.array([[1, 0], [0, -1]])
x = np.array([[0, 1], [1, 0]])
y = np.array([[0, -1j], [1j, 0]])

def one_hot_to_pauli(op):
    if np.allclose(op, [0, 0]):
        return iden
    elif np.allclose(op, [1, 0]):
        return x
    elif np.allclose(op, [0, 1]):
        return z
    elif np.allclose(op, [1, 1]):
        return y

def character_to_pauli(op):
    if op == 0:
        return iden
    elif op == 1:
        return x
    elif op == 2:
        return y
    elif op == 3:
        return z

def integer_to_character(integer):
    if integer == 0:
        return 'I'
    elif integer == 1:
        return 'X'
    elif integer == 2:
        return 'Y'
    elif integer == 3:
        return 'Z'

def map_pauli_operator(op):
    if op == 'I':
        return iden
    elif op == 'X':
        return x
    elif op == 'Y':
        return y
    elif op == 'Z':
        return z
    elif op == '-':
        return -1
    elif op == 'i':
        return 1j
    else:
        return 1


def test_clifford_rotation_map():
    nqubits = np.random.randint(1, 5)
    pauli_vals = torch.tensor(np.random.randint(4, size=(nqubits,)), device=device, dtype=torch.float32)

    pauli_op = 1
    for val in pauli_vals:
        pauli_op = np.kron(pauli_op, character_to_pauli(val))
    U = torch.tensor(scipy.linalg.expm((1j*np.pi/4)*pauli_op), dtype=torch.complex64)

    a = clifford_rotation_map(pauli_vals)
    input_strings = torch.eye(2*nqubits, dtype=torch.float32)

    for input_pauli_string, output_pauli_string, phase in zip(input_strings, a.gs.cpu(), a.ps.cpu()):
        input_operator, output_operator = 1, 1
        for i in range(nqubits):
            input_op = input_pauli_string[2*i:2*i+2]
            output_op = output_pauli_string[2*i:2*i+2]
            input_operator = np.kron(input_operator, one_hot_to_pauli(input_op))
            output_operator = np.kron(output_operator, one_hot_to_pauli(output_op))
        output_operator = torch.tensor(output_operator, dtype=torch.complex64) * 1j**phase
        input_operator = torch.matmul(torch.conj(U).T, torch.matmul(torch.tensor(input_operator, dtype=torch.complex64), U))
        assert torch.allclose(output_operator, input_operator, atol=1e-5, rtol=1e-5)


def test_zero_state():
    nqubits = np.random.randint(1, 5)
    state = torch.zeros((2**nqubits, 1), device=device, dtype=torch.float32)
    state[0, 0] = 1
    a = zero_state(nqubits, device=device)
    inputs, input_strings = torch.eye(2*nqubits, device=device, dtype=torch.float32), []
    for i in range(nqubits):
        input_strings.append(inputs[2*i+1, :])
    for i in range(nqubits):
        input_strings.append(inputs[2*i, :])

    stabilizer_list = a.__repr__().split('(')[1][1:-1]
    stabilizer_list = stabilizer_list.split('\n')
    stabilizer_operator_list = []
    for stabilizer in stabilizer_list:
        stabilizer_operator = 1
        for char in stabilizer:
            stabilizer_operator = np.kron(stabilizer_operator, map_pauli_operator(char))
        stabilizer_operator_list.append(torch.tensor(stabilizer_operator, device=device, dtype=torch.float32))

    for stabilizer_operator in stabilizer_operator_list:
        assert torch.allclose(state, torch.matmul(stabilizer_operator, state))

    for input_pauli_string, output_pauli_string, phase in zip(input_strings, a.gs.cpu(), a.ps.cpu()):
        input_operator, output_operator = 1, 1
        for i in range(nqubits):
            input_op = input_pauli_string[2*i:2*i+2].cpu()
            output_op = output_pauli_string[2*i:2*i+2]
            input_operator = np.kron(input_operator, one_hot_to_pauli(input_op))
            output_operator = np.kron(output_operator, one_hot_to_pauli(output_op))
        output_operator = torch.tensor(output_operator) * 1j**phase
        np.allclose(output_operator, input_operator)


def test_ghz_state():
    nqubits = np.random.randint(1, 5)
    state = torch.zeros((2**nqubits, 1), device=device, dtype=torch.float32)
    state[0, 0] = 1
    state[-1, -1] = 1
    state = state/torch.sqrt(torch.tensor(2))
    a = ghz_state(nqubits, device=device)
    inputs, input_strings = torch.eye(2*nqubits, device=device), []
    for i in range(nqubits):
        input_strings.append(inputs[2*i+1, :])
    for i in range(nqubits):
        input_strings.append(inputs[2*i, :])

    stabilizer_list = a.__repr__().split('(')[1][1:-1]
    stabilizer_list = stabilizer_list.split('\n')
    stabilizer_operator_list = []
    for stabilizer in stabilizer_list:
        stabilizer_operator = 1
        for char in stabilizer:
            stabilizer_operator = np.kron(stabilizer_operator, map_pauli_operator(char))
        stabilizer_operator_list.append(stabilizer_operator)

    for stabilizer_operator in stabilizer_operator_list:
        assert np.allclose(state.cpu(), np.matmul(stabilizer_operator, state.cpu()))

    for input_pauli_string, output_pauli_string, phase in zip(input_strings, a.gs.cpu(), a.ps.cpu()):
        input_operator, output_operator = 1, 1
        for i in range(nqubits):
            input_op = input_pauli_string[2*i:2*i+2].cpu()
            output_op = output_pauli_string[2*i:2*i+2]
            input_operator = np.kron(input_operator, one_hot_to_pauli(input_op))
            output_operator = np.kron(output_operator, one_hot_to_pauli(output_op))
        output_operator = torch.tensor(output_operator) * 1j**phase
        np.allclose(output_operator, input_operator)


def test_maximally_mixed_state():
    nqubits = np.random.randint(1, 5)
    a = maximally_mixed_state(nqubits, device=device)
    inputs, input_strings = torch.eye(2*nqubits, dtype=torch.float32), []
    for i in range(nqubits):
        input_strings.append(inputs[2*i+1, :])
    for i in range(nqubits):
        input_strings.append(inputs[2*i, :])
    for true_pauli, pauli, phases in zip(input_strings, a.gs, a.ps):
        assert np.allclose(true_pauli, pauli.cpu())
        assert phases == 0


def test_inverse():
    nqubits = np.random.randint(1, 5)
    pauli_vals = torch.tensor(np.random.randint(4, size=(nqubits,)), device=device)
    pauli_characters = ''
    for val in pauli_vals:
        pauli_characters += integer_to_character(val)

    a = clifford_rotation_map(pauli_characters, device=device)
    true_a_inverse = clifford_rotation_map('-'+pauli_characters, device=device)
    a_inverse = a.inverse()
    assert np.allclose(a_inverse.gs, true_a_inverse.gs)
    assert np.allclose(a_inverse.ps, true_a_inverse.ps)


def test_compose():
    nqubits = np.random.randint(1, 5)
    cmap = random_clifford_map(nqubits, device=device)
    left_inverse = cmap.inverse().compose(cmap)
    right_inverse = cmap.compose(cmap.inverse())
    true_map_gs = torch.eye(2*nqubits)
    true_map_ps = torch.zeros(2*nqubits)
    assert np.allclose(left_inverse.gs.cpu(), true_map_gs)
    assert np.allclose(left_inverse.ps.cpu(), true_map_ps)
    assert np.allclose(right_inverse.gs.cpu(), true_map_gs)
    assert np.allclose(right_inverse.ps.cpu(), true_map_ps)


def test_expectation():
    ### This convention is the +/-1 convention, rather than the 0 or -1 convention used in measurement. We should probably use this convention for both
    state = ghz_state(3, device=device)
    temp_exp = state.expect(paulis('ZII','IZI','IIZ', device=device))
    assert np.allclose(temp_exp.cpu(), [0, 0, 0])


def test_measure():
    exp = 0
    nmeasurements = 200
    for _ in range(nmeasurements):
        state = ghz_state(3, device=device)
        temp_exp, temp_meas = state.measure(paulis('ZII','IZI','IIZ', device=device))
        assert np.allclose(temp_exp.cpu(), np.array([1, 1, 1])) or np.allclose(temp_exp.cpu(), np.array([0, 0, 0]))
        assert temp_meas == -1.0
        exp += temp_exp
    exp = exp / nmeasurements
    assert torch.all(exp>0.4) and torch.all(exp<0.6)


'''
def test_overlap():
    state = random_clifford_state(3, device=device)
    state.ps = state.ps[0:3]
    assert state.expect(state) == 1.0

    nqubits = np.random.randint(1, 5)
    pauli_vals = torch.tensor(np.random.randint(4, size=(nqubits,)), device=device)
    a = clifford_rotation_map(pauli_vals)
    a_inverse = a.inverse()
    assert a.to_state().expect(a_inverse.to_state()) == 1.0

    pauli_op = 1
    for val in pauli_vals:
        pauli_op = np.kron(pauli_op, character_to_pauli(val))
    U_a = scipy.linalg.expm((1j*np.pi/4)*pauli_op)

    pauli_vals = torch.tensor(np.random.randint(4, size=(nqubits,)), device=device)
    b = clifford_rotation_map(pauli_vals)
    pauli_op = 1
    for val in pauli_vals:
        pauli_op = np.kron(pauli_op, character_to_pauli(val))
    U_b = scipy.linalg.expm((1j*np.pi/4)*pauli_op)

    state = np.zeros((2**nqubits, 1))
    state[0, 0] = 1
    output_a = np.matmul(U_a, state)
    output_b = np.conj(np.matmul(U_b, state)).T
    overlap = a.to_state().expect(b.to_state())
    true_overlap = np.abs(np.matmul(output_b, output_a))**2
    assert np.allclose(overlap, true_overlap)
'''


def test_rotate_by():
    nqubits = np.random.randint(1, 5)
    pauli_vals = torch.tensor(np.random.randint(4, size=(nqubits,)), device=device)
    rho = zero_state(nqubits, device=device)
    rho.rotate_by(pauli(pauli_vals))
    true_rho = clifford_rotation_map(pauli_vals).to_state()
    assert np.allclose(rho.stabilizers.gs.cpu(), true_rho.stabilizers.gs.cpu())


def test_pauli_transform():
    nqubits = np.random.randint(1, 5)
    pauli_vals = torch.tensor(np.random.randint(4, size=(nqubits,)), device=device)
    rho = zero_state(nqubits, device=device)
    rho_map = zero_state(nqubits, device=device).to_map()
    rho = rho.transform_by(clifford_rotation_map(pauli_vals))
    rho_map.transform_by(clifford_rotation_map(pauli_vals))

    rmgs, rmps = [], []
    for i in range(nqubits):
        rmgs.append(rho_map.gs[2*i+1, :])
        rmps.append(rho_map.ps[2*i+1])
    for i in range(nqubits):
        rmgs.append(rho_map.gs[2*i, :])
        rmps.append(rho_map.ps[2*i])

    for rmg, rg, rmp, rp in zip(rmgs, rho.gs.cpu(), rmps, rho.ps.cpu()):
        assert np.allclose(rmg.cpu(), rg)
        assert rmp.cpu() == rp


def test_entropy():
    nqubits = 4
    pauli_vals = torch.tensor(np.random.randint(4, size=(nqubits,)), device=device)
    a = clifford_rotation_map(pauli_vals)
    pauli_op = 1
    for val in pauli_vals:
        pauli_op = np.kron(pauli_op, character_to_pauli(val))
    U_a = scipy.linalg.expm((1j*np.pi/4)*pauli_op)

    state = np.zeros((2**nqubits, 1))
    state[0, 0] = 1
    output_a = np.matmul(U_a, state)

    qutip_ket = qutip.Qobj(output_a)
    qutip_ket.dims = [[2, 2, 4], [1, 1, 1]]

    assert np.allclose(a.to_state().entropy([0, 1]).cpu(), qutip.entropy_vn(qutip_ket.ptrace([0,1]), base=2))


def test_rank():
    nqubits = np.random.randint(1, 5)
    a = zero_state(nqubits, device=device)
    a.set_r(nqubits)
    maximally_mixed = maximally_mixed_state(nqubits, device=device)
    assert np.allclose(a.gs.cpu(), maximally_mixed.gs.cpu())
    assert np.allclose(a.ps.cpu(), maximally_mixed.ps.cpu())


def test_sample():
    ### Note that stabilizer sample is sometimes producing operators that produce the eigenvalue -1 rather than +1
    nqubits = np.random.randint(3, 5)
    pauli_vals = torch.tensor(np.random.randint(4, size=(nqubits,)), device=device)
    a = clifford_rotation_map(pauli_vals).to_state()
    pauli_op = 1
    for val in pauli_vals:
        pauli_op = np.kron(pauli_op, character_to_pauli(val))
    U_a = torch.tensor(scipy.linalg.expm((1j*np.pi/4)*pauli_op))
    state = torch.zeros((2**nqubits, 1), dtype=torch.complex64)
    state[0, 0] = 1
    output_a = np.matmul(U_a, state)
    sample = a.sample(nqubits-1)

    for output_pauli_string, phase in zip(sample.gs.cpu(), sample.ps.cpu()):
        output_operator = 1
        for i in range(nqubits):
            output_op = output_pauli_string[2*i:2*i+2]
            output_operator = np.kron(output_operator, one_hot_to_pauli(output_op))
        output_operator = torch.tensor(output_operator) * 1j**phase
        assert np.allclose(output_a, np.matmul(output_operator, output_a)) or np.allclose(output_a, -np.matmul(output_operator, output_a))

def test_batched_pauli_exp():
  """Test that batched expectation values are correct compared to single expectation values."""
  np.random.seed(0) # set seed for reproducibility 
  random_states = [random_clifford_state(3) for _ in range(100)] #TODO: random seed for stabilizer.random_clifford_state()?
  obs_types = [
    paulis(pauli([np.random.randint(0,4) for _ in range(3)]), 
    pauli([np.random.randint(0,4) for _ in range(3)])), # random pauli operator, as PolyList
    paulis('XXX','IZI','-ZZI'), # PauliList of higher length
    0.5*pauli('XXX')+0.2j*pauli('-ZZI') # Polynomials
  ]
  for obs in obs_types: # tests for each type of obs
    expected = np.stack([state.expect(obs) for state in random_states])  
    actual = vectorizable_expct(random_states, obs)
    np.testing.assert_allclose(actual, expected) 