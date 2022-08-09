import numpy
import qutip as qt
from .utils import (
    ipow, pauli_tokenize, 
    clifford_rotate, pauli_transform,
    batch_dot, aggregate)

class Pauli(object):
    '''Represents a Pauli operator.

    Parameters:
    g: int (2*N) - a Pauli string in binary repr.
    p: int - phase indicator (i power).'''
    def __init__(self, g, p = None):
        self.g = g
        self.p = 0 if p is None else p

    def __repr__(self):
        # interprete phase factor
        if self.N > 0:
            if self.p == 0:
                txt = ' +'
            elif self.p == 1:
                txt = '+i'
            elif self.p == 2:
                txt = ' -'
            elif self.p == 3:
                txt = '-i'
        else:
            txt = 'null'
        # interprete Pauli string
        for i in range(self.N):
            x = self.g[2*i  ]
            z = self.g[2*i+1]
            if x == 0:
                if z == 0:
                    txt += 'I'
                elif z == 1:
                    txt += 'Z'
            elif x == 1:
                if z == 0:
                    txt += 'X'
                elif z == 1:
                    txt += 'Y'
        return txt

    @property
    def N(self): # number of qubits
        return self.g.shape[0]//2
    
    def __neg__(self):
        return type(self)(self.g, (self.p + 2) % 4)

    def __rmul__(self, c):
        if c == 1:
            return self
        elif c == 1j:
            return type(self)(self.g, (self.p + 1) % 4)
        elif c == -1:
            return type(self)(self.g, (self.p + 2) % 4)
        elif c == -1j:
            return type(self)(self.g, (self.p + 3) % 4)
        else: # upgrade to PauliMonomial
            return c * self.as_monomial()

    def __truediv__(self, other):
        return (1/other) * self

    def __add__(self, other):
        return self.as_polynomial() + other

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __matmul__(self, other):
        if isinstance(other, Pauli):
            p = (self.p + other.p + ipow(self.g, other.g)) % 4
            g = (self.g + other.g) % 2
            return Pauli(g, p)
        elif isinstance(other, (PauliMonomial, PauliPolynomial)):
            return self.as_polynomial() @ other.as_polynomial()
        else: 
            raise NotImplementedError('matmul is not implemented for between {} and {}'.format(type(self).__name__, type(other).__name__))

    def trace(self):
        if numpy.sum(self.g) == 0:
            return 2**self.N
        else:
            return 0

    def weight(self):
        return numpy.sum(numpy.sum(self.g.reshape(self.N, 2), -1) != 0)

    def copy(self):
        return Pauli(self.g.copy(), self.p)

    def as_monomial(self):
        '''cast a Pauli operator to a Pauli monomial assuming coefficient = 1'''
        return PauliMonomial(self.g, self.p)

    def as_polynomial(self):
        '''cast a Pauli operator to a Pauli polynomial'''
        return self.as_monomial().as_polynomial()

    def as_list(self):
        '''cast a Pauli operator to a Pauli list'''
        gs = numpy.expand_dims(self.g, 0)
        ps = numpy.array([self.p], dtype=numpy.int_)
        return PauliList(gs, ps)

    def rotate_by(self, generator, mask=None):
        result = self.as_list().rotate_by(generator, mask=mask)
        self.g = result.gs[0]
        self.p = result.ps[0]
        return self

    def transform_by(self, clifford_map, mask=None):
        result = self.as_list().transform_by(clifford_map, mask=mask)
        self.g = result.gs[0]
        self.p = result.ps[0]
        return self
    
    def to_qutip(self):
        paulis = [qt.qeye(2), qt.sigmax(), qt.sigmay(), qt.sigmaz()]
        tmp_list=[]
        for i in range(self.g.shape[0]//2):
            if (self.g[2*i]==1)&(self.g[2*i+1]==1):
                tmp_list.append(paulis[2])
            elif (self.g[2*i]==1)&(self.g[2*i+1]==0):
                tmp_list.append(paulis[1])
            elif (self.g[2*i]==0)&(self.g[2*i+1]==1):
                tmp_list.append(paulis[3])
            else:
                tmp_list.append(paulis[0])
        return (1j)**(self.p)*qt.tensor(tmp_list)

    def tokenize(self):
        gs = numpy.expand_dim(self.g, 0)
        ps = numpy.array([self.p])
        return pauli_tokenize(gs, ps)

class PauliList(object):
    '''Represents a list of Pauli operators.

    Parameters:
    gs: int (L, 2*N) - array of Pauli strings in binary repr.
    ps: int (L) - array of phase indicators (i powers).'''
    def __init__(self, gs, ps = None):
        self.gs = gs
        self.ps = numpy.zeros(self.L, dtype=numpy.int_) if ps is None else ps

    def __repr__(self):
        return '\n'.join([repr(pauli) for pauli in self])

    def __len__(self):
        return self.L

    @property
    def L(self):
        return self.gs.shape[0]

    @property
    def N(self):
        return self.gs.shape[1]//2

    def __getitem__(self, item):
        if isinstance(item, (int, numpy.integer)):
            return Pauli(self.gs[item], self.ps[item])
        return PauliList(self.gs[item], self.ps[item])

    def __neg__(self):
        return type(self)(self.gs, (self.ps + 2) % 4)

    def __truediv__(self, other):
        return (1/other) * self

    def __rmul__(self, c):
        if c == 1:
            return self
        elif c == 1j:
            return type(self)(self.gs, (self.ps + 1) % 4)
        elif c == -1:
            return type(self)(self.gs, (self.ps + 2) % 4)
        elif c == -1j:
            return type(self)(self.gs, (self.ps + 3) % 4)
        else: # upgrade to PauliPolynomial
            raise NotImplementedError('multiplication is not defined for {} when factor is not 1, -1, 1j, -1j.'.format(type(self).__name__))

    def trace(self):
        return numpy.where(numpy.sum(self.gs, -1) == 0, 2**self.N, 0)

    def weight(self):
        return numpy.sum(numpy.sum(self.gs.reshape(self.L, self.N, 2), -1) != 0, -1)

    def copy(self):
        return PauliList(self.gs.copy(), self.ps.copy())

    def as_polynomial(self):
        return PauliPolynomial(self.gs, self.ps)

    def rotate_by(self, generator, mask=None):
        # perform Clifford rotation by Pauli generator (in-place)
        if mask is None:
            clifford_rotate(generator.g, generator.p, self.gs, self.ps)
        else:
            mask2 = numpy.repeat(mask,  2)
            self.gs[:,mask2], self.ps = clifford_rotate(
                generator.g, generator.p, self.gs[:,mask2], self.ps)
        return self

    def transform_by(self, clifford_map, mask=None):
        # perform Clifford transformation by Clifford map (in-place)
        if mask is None:
            self.gs, self.ps = pauli_transform(self.gs, self.ps, 
                clifford_map.gs, clifford_map.ps)
        else:
            mask2 = numpy.repeat(mask, 2)
            self.gs[:,mask2], self.ps = pauli_transform(
                self.gs[:,mask2], self.ps, clifford_map.gs, clifford_map.ps)
        return self

    def tokenize(self):
        return pauli_tokenize(self.gs, self.ps)
    def to_qutip(self):
        lists = []
        paulis = [qt.qeye(2),qt.sigmax(),qt.sigmay(),qt.sigmaz()]
        for l in range(self.L):
            tmp_list=[]
            for i in range(self.N):
                if (self.gs[l,2*i]==1)&(self.gs[l,2*i+1]==1):
                    tmp_list.append(paulis[2])
                elif (self.gs[l,2*i]==1)&(self.gs[l,2*i+1]==0):
                    tmp_list.append(paulis[1])
                elif (self.gs[l,2*i]==0)&(self.gs[l,2*i+1]==1):
                    tmp_list.append(paulis[3])
                else:
                    tmp_list.append(paulis[0])
            lists.append((1j)**(self.ps[l])*qt.tensor(tmp_list))
        return lists
            

class PauliMonomial(Pauli):
    '''Represent a Pauli operator with a coefficient.

    Parameters:
    g: int (2*N) - a Pauli string in binary repr.
    p: int - phase indicator (i power).
    c: comlex - coefficient.'''
    def __init__(self, *args, **kwargs):
        super(PauliMonomial, self).__init__(*args, **kwargs)
        self.c = 1.+0.j # default coefficient

    def __repr__(self):
        # interprete coefficient
        c = self.c * 1j**self.p
        if c.imag == 0.:
            c = c.real
            if c.is_integer():
                txt = '{:d} '.format(int(c))
            else: 
                txt = '{:.5f} '.format(c)
        else:
            txt = '({:.5f}) '.format(c)
        # interprete Pauli string
        for i in range(self.N):
            x = self.g[2*i  ]
            z = self.g[2*i+1]
            if x == 0:
                if z == 0:
                    txt += 'I'
                elif z == 1:
                    txt += 'Z'
            elif x == 1:
                if z == 0:
                    txt += 'X'
                elif z == 1:
                    txt += 'Y'
        return txt

    def __neg__(self):
        return PauliMonomial(self.g, self.p).set_c(-self.c)

    def __rmul__(self, c):
        return PauliMonomial(self.g, self.p).set_c(c * self.c)

    def __truediv__(self, other):
        return (1/other) * self

    def __add__(self, other):
        return self.as_polynomial() + other

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __matmul__(self, other):
        if isinstance(other, (Pauli, PauliMonomial, PauliPolynomial)):
            return self.as_polynomial() @ other.as_polynomial()
        else:
            raise NotImplementedError('matmul is not implemented for between {} and {}'.format(type(self).__name__, type(other).__name__))

    def set_c(self, c):
        self.c = c
        return self

    def trace(self):
        return self.c * super(PauliMonomial, self).trace()

    def copy(self):
        return PauliMonomial(self.g.copy(), self.p).set_c(self.c)
        
    def as_polynomial(self):
        '''cast the Pauli monomial to a single-term Pauli polynomial'''
        gs = numpy.expand_dims(self.g, 0)
        ps = numpy.array([self.p], dtype=numpy.int_)
        cs = numpy.array([self.c], dtype=numpy.complex_)
        return PauliPolynomial(gs, ps).set_cs(cs)

    def inverse(self):
        return Pauli(self.g)/(self.c * 1j**self.p)
    def to_qutip(self):
        paulis = [qt.qeye(2), qt.sigmax(), qt.sigmay(), qt.sigmaz()]
        tmp_list=[]
        for i in range(self.g.shape[0]//2):
            if (self.g[2*i]==1)&(self.g[2*i+1]==1):
                tmp_list.append(paulis[2])
            elif (self.g[2*i]==1)&(self.g[2*i+1]==0):
                tmp_list.append(paulis[1])
            elif (self.g[2*i]==0)&(self.g[2*i+1]==1):
                tmp_list.append(paulis[3])
            else:
                tmp_list.append(paulis[0])
        return self.c*(1j)**(self.p)*qt.tensor(tmp_list)
        

class PauliPolynomial(PauliList):
    '''Represent a linear combination of Pauli operators.

    Parameters:
    gs: int (L, 2*N) - array of Pauli strings in binary repr.
    ps: int (L) - array of phase indicators (i powers).
    cs: comlex - coefficients.'''
    def __init__(self, *args, **kwargs):
        super(PauliPolynomial, self).__init__(*args, **kwargs)
        self.cs = numpy.ones(self.ps.shape, dtype=numpy.complex_) # default coefficient

    def __repr__(self):
        txt = ''
        for k, term in enumerate(self):
            txt_term = repr(term)
            if k != 0:
                if txt_term[0] == '-':
                    txt_term = ' ' + txt_term
                else:
                    txt_term = ' +' + txt_term
            txt  = txt + txt_term
        return txt

    def __getitem__(self, item):
        if isinstance(item, (int, numpy.integer)):
            return PauliMonomial(self.gs[item], self.ps[item]).set_c(self.cs[item])
        return PauliPolynomial(self.gs[item], self.ps[item]).set_cs(self.cs[item])

    def __neg__(self):
        return PauliPolynomial(self.gs, self.ps).set_cs(-self.cs)

    def __rmul__(self, c):
        return PauliPolynomial(self.gs, self.ps).set_cs(c * self.cs)

    def __truediv__(self, other):
        return (1/other) * self

    def __add__(self, other):
        if not isinstance(other, PauliPolynomial):
            if isinstance(other, (PauliMonomial, Pauli, PauliList)):
                other = other.as_polynomial()
            else: # otherwise assuming other is a number
                other = other * pauli_identity(self.N)
        gs = numpy.concatenate([self.gs, other.gs])
        ps = numpy.concatenate([self.ps, other.ps])
        cs = numpy.concatenate([self.cs, other.cs])
        return PauliPolynomial(gs, ps).set_cs(cs).reduce()

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __matmul__(self, other):
        if isinstance(other, (Pauli, PauliMonomial, PauliPolynomial)):
            other = other.as_polynomial()
        else:
            raise NotImplementedError('matmul is not implemented for between {} and {}'.format(type(self).__name__, type(other).__name__))
        gs, ps, cs = batch_dot(self.gs, self.ps, self.cs, other.gs, other.ps, other.cs)
        return PauliPolynomial(gs, ps).set_cs(cs)

    def set_cs(self, cs):
        '''set coefficients'''
        self.cs = cs
        return self

    def trace(self):
        return self.cs.dot(super(PauliPolynomial, self).trace())

    def copy(self):
        return PauliPolynomial(self.gs.copy(), self.ps.copy()).set_cs(self.cs.copy())

    def as_polynomial(self):
        return self

    def reduce(self, tol=1.e-10):
        '''Reduce the Pauli polynomial by 
            1. combine simiilar terms,
            2. move phase factors to coefficients,
            3. drop terms that are too small (coefficient < tol).'''
        gs, inds = numpy.unique(self.gs, return_inverse=True, axis=0)
        cs = aggregate(self.cs * 1j**self.ps, inds, gs.shape[0])
        mask = (numpy.abs(cs) > tol)
        return PauliPolynomial(gs[mask]).set_cs(cs[mask])
    def to_qutip(self):
        paulis = [qt.qeye(2),qt.sigmax(),qt.sigmay(),qt.sigmaz()]
        summation = 0
        for l in range(self.L):
            tmp_list=[]
            for i in range(self.N):
                if (self.gs[l,2*i]==1)&(self.gs[l,2*i+1]==1):
                    tmp_list.append(paulis[2])
                elif (self.gs[l,2*i]==1)&(self.gs[l,2*i+1]==0):
                    tmp_list.append(paulis[1])
                elif (self.gs[l,2*i]==0)&(self.gs[l,2*i+1]==1):
                    tmp_list.append(paulis[3])
                else:
                    tmp_list.append(paulis[0])
            summation += self.cs[l]*(1j)**(self.ps[l])*qt.tensor(tmp_list)
        return summation
        

# ---- constructors ----
def pauli(obj, N = None):
    if isinstance(obj, Pauli):
        return obj
    elif isinstance(obj, (tuple, list, numpy.ndarray)):
        N = len(obj)
        inds = enumerate(obj)
    elif isinstance(obj, dict):
        if N is None:
            raise ValueError('pauli(inds, N) must specify qubit number N when inds is dict.')
        inds = obj.items()
    elif isinstance(obj, str):
        return pauli(list(obj))
    else:
        raise TypeError('pauli(obj) recieves obj of type {}, which is not implemented.'.format(type(obj).__name__))
    g = numpy.zeros(2*N, dtype=numpy.int_)
    h = 0
    p = 0
    for i, mu in inds:
        assert i-h < N, 'qubit {} is out of bounds for system size {}.'.format(i, N)
        if mu == 0 or mu == 'I':
            continue
        elif mu == 1 or mu == 'X':
            g[2*(i-h)] = 1
        elif mu == 2 or mu == 'Y':
            g[2*(i-h)] = 1
            g[2*(i-h)+1] = 1
        elif mu == 3 or mu == 'Z':
            g[2*(i-h)+1] = 1
        elif mu == 4 or mu == '+':
            p = 0
            h += 1
        elif mu == 5 or mu == '-':
            p = 2
            h += 1
        elif mu == 'i':
            p += 1
            h += 1
        elif mu == 6:
            p = 1
            h += 1
        elif mu == 7:
            p = 3
            h += 1
        else:
            h += 1
    if h == 0:
        return Pauli(g, p)
    else:
        return Pauli(g[:-2*h], p)

import types
def paulis(*objs, N = None):
    # short cut if PauliList is passed in
    if len(objs) == 1 :
        if isinstance(objs[0], PauliList):
            return objs[0]
        if isinstance(objs[0], (tuple, list, set, numpy.ndarray, types.GeneratorType)):
            objs = objs[0]
    # otherwise construct data for Pauli operators
    objs = [pauli(obj, N = N) for obj in objs]
    gs = numpy.stack([obj.g for obj in objs])
    ps = numpy.array([obj.p for obj in objs])
    return PauliList(gs, ps)

def pauli_identity(N):
    '''Pauli polynomial of an idenity operator of N qubits.'''
    return PauliPolynomial(numpy.zeros((1,2*N), dtype=numpy.int_))

def pauli_zero(N):
    '''Pauli polynomial of zero operator of N qubit'''
    return 0 * pauli_identity(N)


