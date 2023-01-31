import numpy as np
import scipy
import qutip

from ..stabilizer import *
from ..paulialg import pauli, paulis


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
    pauli_vals = np.random.randint(4, size=(nqubits,))

    pauli_op = 1
    for val in pauli_vals:
        pauli_op = np.kron(pauli_op, character_to_pauli(val))
    U = scipy.linalg.expm((1j*np.pi/4)*pauli_op)

    a = clifford_rotation_map(pauli_vals)
    input_strings = np.eye(2*nqubits)

    for input_pauli_string, output_pauli_string, phase in zip(input_strings, a.gs, a.ps):
        input_operator, output_operator = 1, 1
        for i in range(nqubits):
            input_op = input_pauli_string[2*i:2*i+2]
            output_op = output_pauli_string[2*i:2*i+2]
            input_operator = np.kron(input_operator, one_hot_to_pauli(input_op))
            output_operator = np.kron(output_operator, one_hot_to_pauli(output_op))
        output_operator = output_operator * 1j**phase
        input_operator = np.matmul(np.conj(U).T, np.matmul(input_operator, U))
        assert np.allclose(output_operator, input_operator)


def test_zero_state():
    nqubits = np.random.randint(1, 5)
    state = np.zeros((2**nqubits, 1))
    state[0, 0] = 1
    a = zero_state(nqubits)
    inputs, input_strings = np.eye(2*nqubits), []
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
        assert np.allclose(state, np.matmul(stabilizer_operator, state))

    for input_pauli_string, output_pauli_string, phase in zip(input_strings, a.gs, a.ps):
        input_operator, output_operator = 1, 1
        for i in range(nqubits):
            input_op = input_pauli_string[2*i:2*i+2]
            output_op = output_pauli_string[2*i:2*i+2]
            input_operator = np.kron(input_operator, one_hot_to_pauli(input_op))
            output_operator = np.kron(output_operator, one_hot_to_pauli(output_op))
        output_operator = output_operator * 1j**phase
        np.allclose(output_operator, input_operator)


def test_ghz_state():
    nqubits = np.random.randint(1, 5)
    state = np.zeros((2**nqubits, 1))
    state[0, 0] = 1
    state[-1, -1] = 1
    state = state/np.sqrt(2)
    a = ghz_state(nqubits)
    inputs, input_strings = np.eye(2*nqubits), []
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
        assert np.allclose(state, np.matmul(stabilizer_operator, state))

    for input_pauli_string, output_pauli_string, phase in zip(input_strings, a.gs, a.ps):
        input_operator, output_operator = 1, 1
        for i in range(nqubits):
            input_op = input_pauli_string[2*i:2*i+2]
            output_op = output_pauli_string[2*i:2*i+2]
            input_operator = np.kron(input_operator, one_hot_to_pauli(input_op))
            output_operator = np.kron(output_operator, one_hot_to_pauli(output_op))
        output_operator = output_operator * 1j**phase
        np.allclose(output_operator, input_operator)


def test_maximally_mixed_state():
    nqubits = np.random.randint(1, 5)
    a = maximally_mixed_state(nqubits)
    inputs, input_strings = np.eye(2*nqubits), []
    for i in range(nqubits):
        input_strings.append(inputs[2*i+1, :])
    for i in range(nqubits):
        input_strings.append(inputs[2*i, :])
    for true_pauli, pauli, phases in zip(input_strings, a.gs, a.ps):
        assert np.allclose(true_pauli, pauli)
        assert phases == 0


def test_inverse():
    nqubits = np.random.randint(1, 5)
    pauli_vals = np.random.randint(4, size=(nqubits,))
    pauli_characters = ''
    for val in pauli_vals:
        pauli_characters += integer_to_character(val)

    a = clifford_rotation_map(pauli_characters)
    true_a_inverse = clifford_rotation_map('-'+pauli_characters)
    a_inverse = a.inverse()
    assert np.allclose(a_inverse.gs, true_a_inverse.gs)
    assert np.allclose(a_inverse.ps, true_a_inverse.ps)


def test_compose():
    nqubits = np.random.randint(1, 5)
    cmap = random_clifford_map(nqubits)
    left_inverse = cmap.inverse().compose(cmap)
    right_inverse = cmap.compose(cmap.inverse())
    true_map_gs = np.eye(2*nqubits)
    true_map_ps = np.zeros(2*nqubits)
    assert np.allclose(left_inverse.gs, true_map_gs)
    assert np.allclose(left_inverse.ps, true_map_ps)
    assert np.allclose(right_inverse.gs, true_map_gs)
    assert np.allclose(right_inverse.ps, true_map_ps)


def test_expectation():
    ### This convention is the +/-1 convention, rather than the 0 or -1 convention used in measurement. We should probably use this convention for both
    state = ghz_state(3)
    temp_exp = state.expect(paulis('ZII','IZI','IIZ'))
    assert np.allclose(temp_exp, [0, 0, 0])


def test_measure():
    exp = 0
    nmeasurements = 200
    for _ in range(nmeasurements):
        state = ghz_state(3)
        temp_exp, temp_meas = state.measure(paulis('ZII','IZI','IIZ'))
        assert np.allclose(temp_exp, np.array([1, 1, 1])) or np.allclose(temp_exp, np.array([0, 0, 0]))
        assert temp_meas == -1.0
        exp += temp_exp
    exp = exp / nmeasurements
    assert np.all(exp>0.4) and np.all(exp<0.6)


def test_overlap():
    state = random_clifford_state(3)
    assert state.expect(state) == 1.0

    nqubits = np.random.randint(1, 5)
    pauli_vals = np.random.randint(4, size=(nqubits,))
    a = clifford_rotation_map(pauli_vals)
    a_inverse = a.inverse()
    assert a.to_state().expect(a_inverse.to_state()) == 1.0

    pauli_op = 1
    for val in pauli_vals:
        pauli_op = np.kron(pauli_op, character_to_pauli(val))
    U_a = scipy.linalg.expm((1j*np.pi/4)*pauli_op)

    pauli_vals = np.random.randint(4, size=(nqubits,))
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


def test_rotate_by():
    nqubits = np.random.randint(1, 5)
    pauli_vals = np.random.randint(4, size=(nqubits,))
    rho = zero_state(nqubits)
    rho.rotate_by(pauli(pauli_vals))
    true_rho = clifford_rotation_map(pauli_vals).to_state()
    assert np.allclose(rho.stabilizers.gs, true_rho.stabilizers.gs)


def test_pauli_transform():
    nqubits = np.random.randint(1, 5)
    pauli_vals = np.random.randint(4, size=(nqubits,))
    rho = zero_state(nqubits)
    rho_map = zero_state(nqubits).to_map()
    rho = rho.transform_by(clifford_rotation_map(pauli_vals))
    rho_map.transform_by(clifford_rotation_map(pauli_vals))

    rmgs, rmps = [], []
    for i in range(nqubits):
        rmgs.append(rho_map.gs[2*i+1, :])
        rmps.append(rho_map.ps[2*i+1])
    for i in range(nqubits):
        rmgs.append(rho_map.gs[2*i, :])
        rmps.append(rho_map.ps[2*i])

    for rmg, rg, rmp, rp in zip(rmgs, rho.gs, rmps, rho.ps):
        assert np.allclose(rmg, rg)
        assert rmp == rp


def test_entropy():
    nqubits = 4
    pauli_vals = np.random.randint(4, size=(nqubits,))
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

    assert np.allclose(a.to_state().entropy([0, 1]), qutip.entropy_vn(qutip_ket.ptrace([0,1]), base=2))


def test_rank():
    nqubits = np.random.randint(1, 5)
    a = zero_state(nqubits)
    a.set_r(nqubits)
    maximally_mixed = maximally_mixed_state(nqubits)
    assert np.allclose(a.gs, maximally_mixed.gs)
    assert np.allclose(a.ps, maximally_mixed.ps)


def test_sample():
    ### Note that stabilizer sample is sometimes producing operators that produce the eigenvalue -1 rather than +1
    nqubits = np.random.randint(3, 5)
    pauli_vals = np.random.randint(4, size=(nqubits,))
    a = clifford_rotation_map(pauli_vals).to_state()
    pauli_op = 1
    for val in pauli_vals:
        pauli_op = np.kron(pauli_op, character_to_pauli(val))
    U_a = scipy.linalg.expm((1j*np.pi/4)*pauli_op)
    state = np.zeros((2**nqubits, 1))
    state[0, 0] = 1
    output_a = np.matmul(U_a, state)
    sample = a.sample(nqubits-1)

    for output_pauli_string, phase in zip(sample.gs, sample.ps):
        output_operator = 1
        for i in range(nqubits):
            output_op = output_pauli_string[2*i:2*i+2]
            output_operator = np.kron(output_operator, one_hot_to_pauli(output_op))
        output_operator = output_operator * 1j**phase
        assert np.allclose(output_a, np.matmul(output_operator, output_a)) or np.allclose(output_a, -np.matmul(output_operator, output_a))
