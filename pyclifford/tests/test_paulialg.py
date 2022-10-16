import numpy as np

from ..paulialg import pauli, paulis

### Use 'to' functions in creation of Monomial, Polynomial, and PauliList to test those implicitly


ip, xp, yp, zp = np.array([[1., 0.], [0., 1.]], dtype=np.complex128), np.array([[0., 1.], [1., 0.]], dtype=np.complex128), np.array([[0., -1.0j], [1.0j, 0.]], dtype=np.complex128), np.array([[1., 0.], [0., -1.]], dtype=np.complex128)

def map_pauli_operator(op):
    if op == 'I':
        return ip
    elif op == 'X':
        return xp
    elif op == 'Y':
        return yp
    elif op == 'Z':
        return zp
    elif op == '-':
        return -1
    elif op == 'i':
        return 1j
    else:
        return 1

def build_pauli_string(pauli):
    pauli = pauli.__repr__()
    p_op = np.array(1.)
    for op in pauli:
        p_op = np.kron(p_op, map_pauli_operator(op))
    return p_op


def test_Pauli():

    ### Test __repr__
    rep = pauli('-XiXY').__repr__()
    assert rep == '-iXXY'

    ### Test N
    pauli_vals = np.random.randint(4, size=np.random.randint(1, 10))
    p = pauli(pauli_vals)
    assert pauli_vals.shape[0] == p.N

    ### Test addition, negation, scalar multiplication, and scalar division
    pauli_vals = np.random.randint(4, size=pauli_vals.shape)
    p2 = pauli(pauli_vals)
    assert p.__neg__().__repr__() == ' -' + p.__repr__()[2::]
    assert (2.5*p).__repr__() == (p+p+0.5*p).__repr__()
    assert (-0.5*p).__repr__() == (p-p-0.5*p).__repr__()
    assert paulis(2.5*p).__repr__() == p.__radd__(p).__radd__(0.5*p).__repr__()
    assert (p/2).__repr__() == (0.5*p).__repr__()

    ### Test Pauli multiplication
    p_op, p2_op, pmult_op = build_pauli_string(p), build_pauli_string(p2), build_pauli_string(p.__matmul__(p2))
    assert np.allclose(np.matmul(p_op, p2_op), pmult_op)

    ### Test trace
    assert (p.trace() == np.trace(p_op)) and (p2.trace() == np.trace(p2_op)) and (pauli('-iII').trace() == -4j)

    ### Test weight
    assert p2.weight() == np.sum((pauli_vals>0)*1)

    ### Test copy
    assert np.allclose(p2.g, p2.copy().g) and (p2.p == p2.copy().p)

    ### Test tokenize
    assert (pauli(p.tokenize()[0]).__repr__() == p.__repr__()) and (pauli(p2.tokenize()[0]).__repr__() == p2.__repr__()) and (pauli(p.__matmul__(p2).tokenize()[0]).__repr__() == p.__matmul__(p2).__repr__())


def test_PauliList():

    ### Test __repr__
    rep = paulis('-XiXY', 'YZI').__repr__()
    assert rep == '-iXXY\n +YZI'

    ### Test N and L
    pauli_vals = np.random.randint(4, size=(np.random.randint(1, 10), np.random.randint(1, 10)))
    p = paulis(pauli_vals)
    assert (pauli_vals.shape[0] == len(p)) and (pauli_vals.shape[1] == paulis(p).N)

    ### Test negation, scalar multiplication, and scalar division
    pauli_vals = np.random.randint(4, size=(np.random.randint(1, 10), pauli_vals.shape[1]))
    p2 = paulis(pauli_vals)
    neg_list, test_neg_list = [], []
    for pcomp in p:
        neg_list.append(' -' + pcomp.__repr__()[2::])
        test_neg_list.append((-pcomp).__repr__())
    assert test_neg_list == neg_list
    imag_list, test_imag_list = [], []
    for pcomp in p:
        imag_list.append('-i' + pcomp.__repr__()[2::])
        test_imag_list.append((pcomp/1j).__repr__())
    assert test_imag_list == imag_list

    ### Test trace
    p_op, p2_op = [], []
    for pcomp in p:
        p_op.append(np.trace(build_pauli_string(pcomp)).real)
    for pcomp2 in p2:
        p2_op.append(np.trace(build_pauli_string(pcomp2)).real)
    assert (np.allclose(p.trace(), p_op)) and (np.allclose(p2.trace(), p2_op)) and (pauli('-iII').as_polynomial().trace() == -4j)

    ### Test weight
    assert np.allclose(p2.weight(), np.sum((pauli_vals>0)*1, axis=1))

    ### Test copy
    assert np.allclose(p2.gs, p2.copy().gs) and np.allclose(p2.ps, p2.copy().ps)

    ### Test tokenize
    assert (paulis(p.tokenize()).__repr__() == p.__repr__()) and (paulis(p2.tokenize()).__repr__() == p2.__repr__())


def test_PauliPolynomial():

    ### repr method is failing for both Pauli to PauliPolynomial and PauliList to PauliPolynomial
    rep = pauli('XYIZZ') + pauli('YiYZII') + pauli('-iZXZXZ').as_polynomial()
    assert rep.__repr__() == '(0.00000-1.00000j) ZXZXZ +1 XYIZZ +(0.00000+1.00000j) YYZII'

    pauli_vals = np.random.randint(4, size=(np.random.randint(1, 10), np.random.randint(1, 10)))
    p = paulis(pauli_vals)
    assert (pauli_vals.shape[0] == len(p)) and (pauli_vals.shape[1] == paulis(p).as_polynomial().N)

    ### Test negation, scalar multiplication, and scalar division
    pauli_vals = np.random.randint(4, size=(np.random.randint(1, 10), pauli_vals.shape[1]))
    p2 = paulis(pauli_vals)
    assert np.allclose((p.as_polynomial()+p2.as_polynomial()).gs, np.unique(np.concatenate([p.gs, p2.gs]), axis=0))
    if (p.as_polynomial()-(p2.as_polynomial())).gs.shape == np.unique(np.concatenate([p.gs, (-p2).gs]), axis=0).shape:
        assert np.allclose((p.as_polynomial()-(p2.as_polynomial())).gs, np.unique(np.concatenate([p.gs, (-p2).gs]), axis=0))
    assert np.allclose((5*(p.as_polynomial())).gs, p.gs) and np.allclose((5*(p.as_polynomial())).ps, p.ps) and np.allclose((5*(p.as_polynomial())).cs, 5*(p.as_polynomial().cs))

    ### Test Pauli multiplication
    p_matrix, p2_matrix = 0, 0
    for op in p:
        p_matrix += build_pauli_string(op)
    for op in p2:
        p2_matrix += build_pauli_string(op)
    pmult = p.as_polynomial().__matmul__(p2.as_polynomial())
    pmult_matrix = 0
    for s in pmult.__repr__().split(' +'):
        num, op = s.split(' ')
        pmult_matrix += complex(num) * build_pauli_string(op)
    assert np.allclose(np.matmul(p_matrix, p2_matrix), pmult_matrix)

    ### Test trace
    p_op, p2_op = 0, 0
    for pcomp in p:
        p_op += build_pauli_string(pcomp)
    for pcomp2 in p2:
        p2_op += build_pauli_string(pcomp2)
    assert (np.allclose(p.as_polynomial().trace(), np.trace(p_op))) and (np.allclose(p2.as_polynomial().trace(), np.trace(p2_op))) and (pauli('II').as_polynomial().set_cs(np.array([1.0 + 0.0j])).trace() == 4)
