import numpy
from numba import njit

'''Conventions:
Binary representation of Pauli string. (arXiv:quant-ph/0406196)
    A N-qubit Pauli string is represented as a (2*N)-dimensional binary 
    array of the following form:
        g = [x0,z0;x1,z1;...;x(N-1),z(N-1)]  (flattened)
    which corresponds to
        sigma[g] = i^(x.z) prod_i X_i^xi Z_i^zi
Commutation relation of Pauli strings:
        sigma[g1] sigma[g2] = (-)^acq(g1,g2) sigma[g2] sigma[g1]
Product of Pauli strings:
        sigma[g1] sigma[g2] = i^ipow(g1,g2) sigma[(g1+g2)%2]
'''
# ---- Pauli foundation ----
@njit
def front(g):
    '''Find the first nontrivial qubit in a Pauli string.

    Parameters:
    g: int (2*N) -  a Pauli string in binary representation.

    Returns:
    i: int - position of its first nontrivial qubit.

    Note:
    If the Pauli string is identity, i = N-1 will be returned, although there 
    is no nontrivial qubit.'''
    (N2,) = g.shape
    N = N2//2
    for i in range(N):
        if g[2*i] != 0 or g[2*i+1] != 0:
            break
    return i

@njit
def condense(g):
    '''Condense the Pauli string by taking collecting it in its support, returns
    a shorter string and the support.

    Parameters:
    g: int (2*N) - a Pauli string in binary representation.

    Returns:
    g_cond: int (2*n) - the condensed Pauli string in binary representation.
    qubits: int (n) - indices of supporting qubits.'''
    (N2,) = g.shape
    N = N2//2
    mask = numpy.zeros(N, dtype=numpy.bool_)
    for i in range(N):
        if g[2*i] != 0 or g[2*i+1] != 0:
            mask[i] = True
    qubits = numpy.arange(N)[mask]
    return g[numpy.repeat(mask, 2)], qubits

@njit
def p0(g):
    '''Bare phase factor due to x.z for a Pauli string.

    Parameters:
    g: int (2*N) - a Pauli string in binary representation.

    Returns:
    p0: int - bare phase factor x.z for the string.'''
    (N2,) = g.shape
    N = N2//2
    p0 = 0
    for i in range(N):
        p0 += g[2*i] * g[2*i+1]
    return p0 % 4

@njit
def acq(g1, g2):
    '''Calculate Pauli operator anticmuunation indicator.

    Parameters:
    g1: int (2*N) - the first Pauli string in binary representation.
    g2: int (2*N) - the second Pauli string in binary representation.
    
    Returns:
    acq: int - acq = 0 if g1, g2 commute, acq = 1 if g1, g2 anticommute.'''
    assert g1.shape == g2.shape
    (N2,) = g1.shape
    N = N2//2
    acq = 0
    for i in range(N):
        acq += g1[2*i+1]*g2[2*i] - g1[2*i]*g2[2*i+1]
    return acq % 2

@njit
def ipow(g1, g2):
    '''Phase indicator for the product of two Pauli strings.

    Parameters:
    g1: int (2*N) - the first Pauli string in binary representation.
    g2: int (2*N) - the second Pauli string in binary representation.
    
    Returns:
    ipow: int - the phase indicator (power of i) when product 
        sigma[g1] with sigma[g2].'''
    assert g1.shape == g2.shape
    (N2,) = g1.shape
    N = N2//2
    ipow = 0
    for i in range(N):
        g1x = g1[2*i  ]
        g1z = g1[2*i+1]
        g2x = g2[2*i  ]
        g2z = g2[2*i+1]
        gx = g1x + g2x
        gz = g1z + g2z 
        ipow += g1z * g2x - g1x * g2z + 2*((gx//2) * gz + gx * (gz//2))
    return ipow % 4

@njit
def ps0(gs):
    '''Bare phase factor due to x.z for Pauli strings.

    Parameters:
    gs: int (L,2*N) - array of Pauli strings in binary representation.

    Returns:
    ps0: int (L) - bare phase factor x.z for all strings.'''
    (L, N2) = gs.shape
    N = N2//2
    ps0 = numpy.zeros(L, dtype=numpy.int_)
    for j in range(L):
        for i in range(N):
            ps0[j] += gs[j,2*i] * gs[j,2*i+1]
    return ps0 % 4

@njit
def acq_mat(gs):
    '''Construct anticommutation indicator matrix for a set of Pauli strings.

    Parameters:
    gs: int (L,2*N) - array of Pauli strings in binary representation.

    Returns:
    mat: int (L,L) - anticommutation indicator matrix.'''
    (L, N2) = gs.shape
    N = N2//2
    mat = numpy.zeros((L,L), dtype=numpy.int_)
    for j1 in range(L):
        for j2 in range(L):
            for i in range(N):
                mat[j1,j2] += gs[j1,2*i+1]*gs[j2,2*i] - gs[j1,2*i]*gs[j2,2*i+1]
    mat = mat % 2
    return mat

@njit
def batch_dot(gs1, ps1, cs1, gs2, ps2, cs2):
    '''batch dot product of two Pauli polynomials

    Parameters:
    gs1: int (L1,2*N) - Pauli strings in the first polynomial.
    ps1: int (L1) - phase indicators in the first polynomial.
    cs1: complex (L1) - coefficients in the first polynomial.
    gs2: int (L2,2*N) - Pauli strings in the second polynomial.
    ps2: int (L2) - phase indicators in the second polynomial.
    cs2: complex (L2) - coefficients in the second polynomial.

    Returns
    gs: int (L1*L2,2*N) - Pauli strings in the second polynomial.
    ps: int (L1*L2) - phase indicators in the second polynomial.
    cs: complex (L1*L2) - coefficients in the second polynomial.'''
    (L1, N2) = gs1.shape
    (L2, N2) = gs2.shape
    gs = numpy.empty((L1,L2,N2), dtype=numpy.int_)
    ps = numpy.empty((L1,L2), dtype=numpy.int_)
    cs = numpy.empty((L1,L2), dtype=numpy.complex128)
    for j1 in range(L1):
        for j2 in range(L2):
            ps[j1,j2] = (ps1[j1] + ps2[j2] + ipow(gs1[j1], gs2[j2]))%4
            gs[j1,j2] = (gs1[j1] + gs2[j2])%2
            cs[j1,j2] = cs1[j1] * cs2[j2]
    gs = numpy.reshape(gs, (L1*L2,-1))
    ps = numpy.reshape(ps, (L1*L2,))
    cs = numpy.reshape(cs, (L1*L2,))
    return gs, ps, cs

# ---- token related ----
@njit
def pauli_tokenize(gs, ps):
    '''Create a token of Pauli operators for learning tasks.

    Parameters:
    gs: int (L, 2*N) - Pauli strings in binary representation.
    ps: int (L) - phase indicators.

    Returns:
    ts: int (L, N+1) - tokens.
       0 = I, 1 = X, 2 = Y, 3 = Z, 4 = +, 5 = -1, 6 = +i, 7 = -i'''
    (L, N2) = gs.shape
    N = N2//2
    ts = numpy.zeros((L,N+1), dtype=numpy.int_)
    for j in range(L):
        for i in range(N):
            ts[j,i] = 3*gs[j,2*i+1] + (-1)**gs[j,2*i+1] * gs[j,2*i]
        x = ps[j]
        ts[j,N] = 4 + x * (11 - 9 * x + 2 * x**2) // 2
    return ts

# ---- combination, trasnformation, decomposition ----
@njit
def pauli_combine(C, gs_in, ps_in): 
    '''Combine Pauli operators by operator product.
        (left multiplication)

    Parameters:
    C: int (L_out, L_in) - one-hot encoding of selected operators.
    gs_in: int (L_in, 2*N) - input binary representation of Pauli strings.
    ps_in: int (L_in) - phase indicators of input operators.

    Returns:
    gs_out: int (L_out, 2*N) - output binary representation of Pauli strings.
    ps_out: int (L_out) - phase indicators of output operators.
    '''
    (L_out, L_in) = C.shape
    N2 = gs_in.shape[-1]
    gs_out = numpy.zeros((L_out, N2), dtype=numpy.int_) # identity
    ps_out = numpy.zeros((L_out,), dtype=numpy.int_)
    for j_out in range(L_out):
        for j_in in range(L_in):
            if C[j_out, j_in]:
                ps_out[j_out] = (ps_out[j_out] + ps_in[j_in] + ipow(gs_out[j_out], gs_in[j_in]))%4
                gs_out[j_out] = (gs_out[j_out] + gs_in[j_in])%2
    return gs_out, ps_out

@njit
def pauli_transform(gs_in, ps_in, gs_map, ps_map):
    '''Transform Pauli operators by Clifford map.
        (right multiplication)

    Parameters:
    gs_in: int (L, 2*N) - input binary representation of Pauli strings.
    ps_in: int (L) - phase indicators of input operators.
    gs_map: int (2*N, 2*N) - operator map in binary representation.
    ps_map: int (2*N) - phase indicators associated to target operators.

    Returns:
    gs_out: int (L, 2*N) - output binary representation of Pauli strings.
    ps_out: int (L) - phase indicators of output operators.'''
    gs_out, ps_out = pauli_combine(gs_in, gs_map, ps_map)
    ps_out = (ps_in + ps0(gs_in) + ps_out)%4
    return gs_out, ps_out

@njit
def pauli_decompose(gs_in, ps_in, gs_stb, ps_stb, r):
    '''Decompose Pauli operators into stabilizer and destabilizers.

    Parameters:
    gs_in: int (L, 2*N) - Pauli strings in binary representation.
    ps_in: int (L) - phase indicators of Pauli operators.
    gs_stb: int (2*N, 2*N) - stabilizer tableaux in binary representation.
    ps_stb: int (2*N) - phase indicators of (de)stabilizer.
    r: int - number of standby stabilizer pairs (0:r to be excluded).

    Returns:
    bs_out: int (L, N-r) - binary encoding of destabilizer decomposition.
    cs_out: int (L, N-r) - binary encoding of stabilizer decomposition.
    ps_out: int (L) - phase indicators of decomposed operators.'''
    (L, N2) = gs_in.shape
    N = N2//2
    bs_out = numpy.zeros((L, N-r), dtype=numpy.int_)
    cs_out = numpy.zeros((L, N-r), dtype=numpy.int_)
    ps_out = ps_in.copy()
    g_tmp = numpy.zeros(N2, dtype=numpy.int_)
    for k in range(L):
        g_tmp.fill(0)
        for j in range(r, N):
            if acq(gs_in[k], gs_stb[j]):
                bs_out[k,j-r] = 1
                ps_out[k] = ps_out[k] - ps_stb[j+N] - ipow(g_tmp, gs_stb[j+N])
                g_tmp = (g_tmp + gs_stb[j+N])%2
        for j in range(r, N):
            if acq(gs_in[k], gs_stb[j+N]):
                cs_out[k,j-r] = 1
                ps_out[k] = ps_out[k] - ps_stb[j] - ipow(g_tmp, gs_stb[j])
                g_tmp = (g_tmp + gs_stb[j])%2
    return bs_out, cs_out, ps_out%4

# ---- clifford rotation ----
@njit
def clifford_rotate(g, p, gs, ps):
    '''Apply Clifford rotation to Pauli operators.

    Parameters:
    g: int (2*N) -  Clifford rotation generator in binary representation.
    p: int - phase indicator (p = 0, 2 only).
    gs: int (L, 2*N) - input binary representation of Pauli strings.
    ps: int (L)  - phase indicators of input operators.
 
    Returns: gs, ps in-place modified.''' 
    (L, N2) = gs.shape
    for j in range(L):
        if acq(g, gs[j]):
            ps[j] = (ps[j] + p + 1 + ipow(gs[j], g))%4
            gs[j] = (gs[j] + g)%2
    return gs, ps

@njit
def clifford_rotate_signless(g, gs):
    '''Apply Clifford rotation to Pauli strings without signs.

    Parameters:
    g: int (2*N) -  Clifford rotation generator in binary representation.
    gs: int (L, 2*N) - array of Pauli strings in binary representation.

    Returns: gs in-place modified.''' 
    (L, N2) = gs.shape
    for j in range(L):
        if acq(g, gs[j]):
            gs[j] = (gs[j] + g)%2
    return gs

# ---- diagonalization ----
@njit
def pauli_is_onsite(g, i0=0):
    '''check if a Pauli string is localized on a qubit.

    Parameters:
    g: int (2*N) - Pauli string to check.
    i0: int  - target qubit.

    Returns: True/False'''
    (N2,) = g.shape
    N = N2//2
    out = True # assuming operator is onsite
    for i in range(N):
        if i == i0: # skip target site
            continue
        if g[2*i] != 0 or g[2*i+1] != 0:
            out = False
            break
    return out

@njit
def pauli_diagonalize1(g1, i0 = 0):
    '''Find a series of Clifford roations to diagonalize a single Pauli string
    to qubit i0 as Z.

    Parameters:
    g1: int (2*N) - Pauli string in binary representation.
    i0: int  - target qubit

    Returns:
    gs: int (L, 2*N) - binary representations of Clifford generators.'''
    (N2,) = g1.shape
    N = N2//2
    gs = [] # prepare to collect Clifford generators
    if not (pauli_is_onsite(g1, i0) and g1[2*i0] == 0): # if g1 is not on site and diagonal
        if g1[2*i0] == 0: # if g1 commute with Z0
            g = g1.copy()
            if g1[2*i0+1] == 0: # g1 is trivial at site 0
                i = front(g) # find the first non-trivial qubit as pivot
                # XYZ cyclic on the pivot qubit
                g[2*i] = (g[2*i] + g[2*i+1])%2
                g[2*i+1] = (g[2*i+1] + g[2*i])%2
                # now g anticommute with g1
            g[2*i0] = 1 # such that g also anticommute with Z0
            gs.append(g)
            g1 = (g1 + g)%2
        # now g1 anticommute with Z0                
        g = g1.copy()
        g[2*i0+1] = (g[2*i0+1] + 1)%2 # g = g1 (*) Z0
        gs.append(g)
        g1 = (g1 + g)%2
        # now g1 has been transformed to Z0
    return gs

@njit
def pauli_diagonalize2(g1, g2, i0 = 0):
    '''Find a series of Clifford roations to diagonalize a pair of anticommuting
    Pauli strings to qubit i0 as Z and X (or Y).

    Parameters:
    g1: int (2*N) - binary representation of stabilizer.
    g2: int (2*N) - binary representation of destabilizer.
    i0: int - target qubit
    Returns:
    gs: int (L, 2*N) - binary representations of Clifford generators.
    g1: int (2*N) - binary representation of transformed stabilizer.
    g2: int (2*N) - binary representation of transformed destabilizer.'''
    assert g1.shape == g2.shape
    (N2,) = g1.shape
    N = N2//2
    gs = [] # prepare to collect Clifford generators
    # bring g1 to Z0
    if not (pauli_is_onsite(g1, i0) and g1[2*i0] == 0): # if g1 is not on site and diagonal
        if g1[2*i0] == 0: # if g1 commute with Z0
            g = g1.copy()
            if g1[2*i0+1] == 0: # g1 is trivial at site 0
                i = front(g) # find the first non-trivial qubit as pivot
                # XYZ cyclic on the pivot qubit
                g[2*i] = (g[2*i] + g[2*i+1])%2
                g[2*i+1] = (g[2*i+1] + g[2*i])%2
                # now g anticommute with g1
            g[2*i0] = 1 # such that g also anticommute with Z0
            gs.append(g)
            g1 = (g1 + g)%2
            g2 = (g2 + acq(g, g2) * g)%2
        # now g1 anticommute with Z0                
        g = g1.copy()
        g[2*i0+1] = (g[2*i0+1] + 1)%2 # g = g1 (*) Z0
        gs.append(g)
        g1 = (g1 + g)%2
        g2 = (g2 + acq(g, g2) * g)%2
        # now g1 has been transformed to Z0
    # bring g2 to X0,Y0
    if not pauli_is_onsite(g2, i0): # if g2 is not on site
        g = g2.copy()
        g[2*i0] = 0
        g[2*i0+1] = 1
        gs.append(g)
        g2 = (g2 + g)%2
        # now g2 has been transformed to X0 or Y0
    return gs, g1, g2

# ---- random Clifford ---
@njit
def random_pair(N):
    '''Sample an anticommuting pair of random stabilizer and destabilizer.

    Parameters:
    N: int - number of qubits.

    Returns:
    g1: int (2*N) - binary representation of stabilizer.
    g2: int (2*N) - binary representation of destabilizer.
    '''
    g1 = numpy.random.randint(0,2,2*N)
    g2 = numpy.random.randint(0,2,2*N)
    while (g1 == 0).all(): # resample g1 if it is all zero
        g1 = numpy.random.randint(0,2,2*N)
    if acq(g1, g2) == 0: # if g1, g2 commute
        i = front(g1) # locate the first nontrivial g1 site
        # flip commutativity by chaning g2
        g2[2*i] = (g2[2*i] + g1[2*i+1])%2
        g2[2*i+1] = (g2[2*i+1] + g1[2*i] + g1[2*i+1])%2
    return g1, g2

@njit
def random_pauli(N):
    '''Sample a random Pauli map.

    Parameters:
    N: int - number of qubits.

    Returs:
    gs: int (2*N, 2*N) - random Pauli map matrix.'''
    gs = numpy.zeros((2*N,2*N), dtype=numpy.int_)
    for i in range(N):
        g1, g2 = random_pair(1)
        gs[2*i  ,2*i:2*i+2] = g1
        gs[2*i+1,2*i:2*i+2] = g2
    return gs

def random_clifford(N):
    '''Sample a random Clifford map: a binary matrix with elements specifying 
    how each single Pauli operator [X0,Z0,X1,Z1,...] should gets mapped to the 
    corresponding Pauli strings. 
        based on the algorithm in (https://arxiv.org/abs/2008.06011)

    Parameter:
    N: int - number of qubits.

    Returns:
    gs: int (2*N, 2*N) - random Clifford map matrix (phase not assigned).'''
    def random_clifford_(gs):
        '''Fill random anticommuting Pauli strings in an array.
            (as recursive constructor called by random_clifford) 
        Parameters:
        gs: int (2*n, 2*n) - array to fill in with Pauli strings.'''
        n = gs.shape[-1]//2
        g1, g2 = random_pair(n)
        if n == 1:
            gs[0] = g1
            gs[1] = g2
        else:
            gens, g1, g2 = pauli_diagonalize2(g1, g2)
            gs[0] = g1
            gs[1] = g2
            random_clifford_(gs[2:,2:])
            for g in reversed(gens):
                gs = clifford_rotate_signless(g, gs)
        return gs
    return random_clifford_(numpy.zeros((2*N,2*N), dtype=numpy.int_))

# ---- map/state conversion ----
@njit
def map_to_state(gs_in, ps_in):
    '''Convert Clifford map to stabilizer state.

    Parameters:
    gs_in: int (2*N, 2*N) - Pauli strings in map order.
    ps_in: int (2*N) - phase indicators in map order.

    Returns:
    gs_out: int (2*N, 2*N) - Pauli strings in tableau order.
    ps_out: int (2*N) - phase indicators in tableau order.'''
    (L, N2) = gs_in.shape
    N = N2//2
    gs_out = numpy.empty_like(gs_in)
    ps_out = numpy.empty_like(ps_in)
    for i in range(N):
        gs_out[N+i] = gs_in[2*i]
        gs_out[i] = gs_in[2*i+1]
        ps_out[N+i] = ps_in[2*i]
        ps_out[i] = ps_in[2*i+1]
    return gs_out, ps_out

@njit
def state_to_map(gs_in, ps_in):
    '''Convert stabilizer state to Clifford map.

    Parameters:
    gs_in: int (2*N, 2*N) - Pauli strings in tableau order.
    ps_in: int (2*N) - phase indicators in tableau order.

    Returns:
    gs_out: int (2*N, 2*N) - Pauli strings in map order.
    ps_out: int (2*N) - phase indicators in map order.'''
    (L, N2) = gs_in.shape
    N = N2//2
    gs_out = numpy.empty_like(gs_in)
    ps_out = numpy.empty_like(ps_in)
    for i in range(N):
        gs_out[2*i] = gs_in[N+i]
        gs_out[2*i+1] = gs_in[i]
        ps_out[2*i] = ps_in[N+i]
        ps_out[2*i+1] = ps_in[i]
    return gs_out, ps_out

# ---- stabilizer related ----
''' Formulation:
A stabilizer state encoding r logical qubits on N physical qubits 
is described by the density matrix of the following form:
    rho = sum_{i=1}^{N-r} 2^{-r} (1 + S_i)/2.
where 
* S_i are Pauli stabilizers (commuting with each other), 
  whose Pauli strings are stored in gs_stb[r:N] in binary representation,
  and phase indicators are stored in ps_stb[r:N].
* r is the log2 rank of the density matrix, i.e. the number of standby stabilizers.

Structure of Stabilizer Tableau:
* gs_stb[0:r]:     Pauli strings of standby stabilizers in binary representation
* gs_stb[r:N]:     Pauli strings of active stabilizers in binary representation
* gs_stb[N:N+r]:   Pauli strings of standby destabilizers in binary representation
* gs_stb[N+r:2*N]: Pauli strings of active destabilizers in binary representation
* ps_stb[0:r]:     phase indicators of standby stabilizers
* ps_stb[r:N]:     phase indicators of active stabilizers
* ps_stb[N:N+r]:   phase indicators of standby destabilizers
* ps_stb[N+r:2*N]: phase indicators of active destabilizers
Symplectic structure: each stabilizer gs_stb[i] only anticommutes with 
                      its destabilizer gs_stb[N+i].

Algorithm:
Stabilizer states can serve both as the prior density matrix 
and as the set of commuting observables.
- As state: described by density matrix 
        rho = sum_{i=1}^{N-r} 2^{-r} (1 + S_i)/2
    where S_i are the active stabilizers.
- As observable: described by projective measurement operator 
        pi(o) = sum_{k=1}^{L} 2^{-N+L} (1 + (-)^o_k O_k)/2
    where O_k are the commuting observables, and o_k is the measurement outcome.
---- measure ----
When measuring {O_k} on a stabilizer state of {S_i}, the algorithm is as follows:
for O_k in {O_k}:
    if O_k anticommutes with:
        case 1: any active stabilizer (the first of which being S_p)
        case 2: any standy stabilizer or destabilizer (the first of which being S_p)
        case 3: otherwise
    then:
        case 1: O_k is an error operator  -> update + readout
        case 2: O_k is a logical operator -> update + extend + readout
        case 3: O_k is a trivial operator -> readout
    if update:
        - O_k replaces S_p to be the active stabilizer,
        - S_p becomes its corresponding destabilizer,
        - for any other stabilizers and destabilizers that anticommute with O_k,
          update them to commute with O_k by multiplying O_k to them,
          - keep track of the phase accumulated in the process, [1]
        if extend:
            - rank r is reduced by 1,
            - include the new stabilizer S_p to active stabilizers,
            - include the new destabilizer S_q to active destabilizers.
        - sign of O_k is randomly sampled with half-to-half probability. [2]
          - record measurement outcome, [3]
          - update log2 probability. [4]
    else:
        - readout measurement outcome, [5]
        - log2 probability = 0. [6]
--- postselect ---
Same as measure, but:
- [2] measurement outcome not sampled butcopied from observation value.
- [3] omitted.
- [6] if stabilizer readout not match observation value, log2 probability = -inf.
--- project ---
Same as measure, but lines [1-6] are omitted.
'''
@njit
def stabilizer_measure(gs_stb, ps_stb, gs_obs, ps_obs, r):
    '''Measure a set of commuting Pauli observables on a stabilizer state.

    Given the prior state rho = sum_{i=1}^{N-r} 2^{-r} (1 + S_i)/2,
    specified by (gs_stb, ps_stb, r), and a set of commuting observables
    {O_k}, specified by (gs_obs, ps_obs), sample measurement outcomes 
    out := {o_k} from the conditional probability distribution:
        p(pi(o)|rho) := Tr(rho pi(o))
    where pi(o) = sum_{k=1}^{L} 2^{-N+L} (1 + (-)^o_k O_k)/2 is the 
    measurement operator. Upon measurement, the state is updated to:
        rho' = pi(o) rho pi(o) / p(pi(o)|rho).
    Returns rho', o, log2 p(pi(o)|rho).

    Parameters:
    gs_stb: int (2*N, 2*N) - Pauli strings in original stabilizer tableau.
    ps_stb: int (N) - phase indicators of (de)stabilizers.
    gs_obs: int (L, 2*N) - strings of Pauli operators to be measured.
    ps_obs: int (L) - phase indicators of Pauli operators to be measured.
    r: int - log2 rank of density matrix (num of standby stablizers).

    Returns:
    gs_stb: int (2*N, 2*N) - Pauli strings in updated stabilizer tableau.
    ps_stb: int (N) - phase indicators of (de)stabilizers.
    r: int - updated log2 rank of density matrix.
    out: int (L) - measurment outcomes (0 or 1 binaries).
    log2prob: real - log2 probability of this outcome.'''
    (L, Ng) = gs_obs.shape
    N = Ng//2
    assert 0<=r<=N
    out = numpy.empty(L, dtype=numpy.int_)
    ga = numpy.empty(2*N, dtype=numpy.int_) # workspace for stabilizer accumulation
    pa = 0 # workspace for phase accumulation
    log2prob = 0.
    for k in range(L): # for each observable gs_obs[k]
        update = False
        extend = False
        p = 0 # pointer to the first anticommuting operator
        ga[:] = 0
        pa = 0
        for j in range(2*N):
            if acq(gs_stb[j], gs_obs[k]): # find gs_stb[j] anticommute with gs_obs[k]
                if update: # if gs_stb[j] is not the first anticommuting operator
                    # update gs_stb[j] to commute with gs_obs[k]
                    if j < N: # if gs_stb[j] is a stablizer, phase matters
                        ps_stb[j] = (ps_stb[j] + ps_stb[p] + ipow(gs_stb[j], gs_stb[p]))%4
                    gs_stb[j] = (gs_stb[j] + gs_stb[p])%2
                else: # if gs_stb[j] is the first anticommuting operator
                    if j < N + r: # if gs_stb[j] is not an active destabilizer
                        p = j # move pointer to j
                        update = True
                        if not r <= j < N: # if gs_stb[j] is a standby operator
                            extend = True
                    else: # gs_stb[j] an active destabilizer, meaning gs_obs[k] already a combination of active stabilizers
                        # collect corresponding stabilizer component to ga
                        pa = (pa + ps_stb[j-N] + ipow(ga, gs_stb[j-N]))%4
                        ga = (ga + gs_stb[j-N])%2
        if update:
            # now gs_stb[p] and gs_obs[k] anticommute
            q = (p+N)%(2*N) # get q as dual of p 
            gs_stb[q] = gs_stb[p] # move gs_stb[p] to gs_stb[q]
            gs_stb[p] = gs_obs[k] # add gs_obs[k] to gs_stb[p]
            if extend:
                r -= 1 # rank will reduce under extension
                # bring new stabilizer from p to r
                if p == r:
                    pass
                elif q == r:
                    gs_stb[numpy.array([p,q])] = gs_stb[numpy.array([q,p])] # swap p,q
                else:
                    s = (r+N)%(2*N) # get s as dual of r
                    gs_stb[numpy.array([p,r])] = gs_stb[numpy.array([r,p])] # swap p,r
                    gs_stb[numpy.array([q,s])] = gs_stb[numpy.array([s,q])] # swap q,s
                p = r
            # as long as gs_obs[k] is not eigen, outcome will be half-to-half
            ps_stb[p] = 2 * numpy.random.randint(2)
            out[k] = ((ps_stb[p] - ps_obs[k])%4)//2 #0->0(+1 eigenvalue), 2->1(-1 eigenvalue)
            log2prob -= 1.
        else: # no update, gs_obs[k] is eigen, result is in pa
            assert((ga == gs_obs[k]).all())
            out[k] = ((pa - ps_obs[k])%4)//2
    return gs_stb, ps_stb, r, out, log2prob

@njit
def stabilizer_postselect(gs_stb, ps_stb, gs_obs, ps_obs, r):
    '''Postselect stabilizer state given a set of Pauli observations.

    Given the prior state rho = sum_{i=1}^{N-r} 2^{-r} (1 + S_i)/2,
    specified by (gs_stb, ps_stb, r), and a measurement operator
    sigma = sum_{k=1}^{L} 2^{-N+L} (1 + O_k)/2. Postselect the state by
        rho' = sigma rho sigma / p(sigma|rho),
    where p(sigma|rho) = Tr(rho sigma) is the success probability.
    Returns rho', log2 p(sigma|rho).

    Parameters:
    gs_stb: int (2*N, 2*N) - Pauli strings in original stabilizer tableau.
    ps_stb: int (N) - phase indicators of (de)stabilizers.
    gs_obs: int (L, 2*N) - strings of Pauli operators to be postselected.
    ps_obs: int (L) - phase indicators of Pauli operators to be postselected.
    r: int - log2 rank of density matrix (num of standby stablizers).

    Returns:
    gs_stb: int (2*N, 2*N) - Pauli strings in updated stabilizer tableau.
    ps_stb: int (N) - phase indicators of (de)stabilizers.
    r: int - updated log2 rank of density matrix.
    log2prob: real - log2 probability of successful postselection.'''
    (L, Ng) = gs_obs.shape
    N = Ng//2
    assert 0<=r<=N
    ga = numpy.empty(2*N, dtype=numpy.int_) # workspace for stabilizer accumulation
    pa = 0 # workspace for phase accumulation
    log2prob = 0.
    for k in range(L): # for each observable gs_obs[k]
        update = False
        extend = False
        p = 0 # pointer to the first anticommuting operator
        ga[:] = 0
        pa = 0
        for j in range(2*N):
            if acq(gs_stb[j], gs_obs[k]): # find gs_stb[j] anticommute with gs_obs[k]
                if update: # if gs_stb[j] is not the first anticommuting operator
                    # update gs_stb[j] to commute with gs_obs[k]
                    if j < N: # if gs_stb[j] is a stablizer, phase matters
                        ps_stb[j] = (ps_stb[j] + ps_stb[p] + ipow(gs_stb[j], gs_stb[p]))%4
                    gs_stb[j] = (gs_stb[j] + gs_stb[p])%2
                else: # if gs_stb[j] is the first anticommuting operator
                    if j < N + r: # if gs_stb[j] is not an active destabilizer
                        p = j # move pointer to j
                        update = True
                        if not r <= j < N: # if gs_stb[j] is a standby operator
                            extend = True
                    else: # gs_stb[j] an active destabilizer, meaning gs_obs[k] already a combination of active stabilizers
                        # collect corresponding stabilizer component to ga
                        pa = (pa + ps_stb[j-N] + ipow(ga, gs_stb[j-N]))%4
                        ga = (ga + gs_stb[j-N])%2
        if update:
            # now gs_stb[p] and gs_obs[k] anticommute
            q = (p+N)%(2*N) # get q as dual of p 
            gs_stb[q] = gs_stb[p] # move gs_stb[p] to gs_stb[q]
            gs_stb[p] = gs_obs[k] # add gs_obs[k] to gs_stb[p]
            if extend:
                r -= 1 # rank will reduce under extension
                # bring new stabilizer from p to r
                if p == r:
                    pass
                elif q == r:
                    gs_stb[numpy.array([p,q])] = gs_stb[numpy.array([q,p])] # swap p,q
                else:
                    s = (r+N)%(2*N) # get s as dual of r
                    gs_stb[numpy.array([p,r])] = gs_stb[numpy.array([r,p])] # swap p,r
                    gs_stb[numpy.array([q,s])] = gs_stb[numpy.array([s,q])] # swap q,s
                p = r
            # stabilizer sign set by observation value via postselection
            ps_stb[p] = ps_obs[k]
            log2prob -= 1.
        else: # no update, gs_obs[k] is eigen, result is in pa
            assert((ga == gs_obs[k]).all())
            if (pa - ps_obs[k])%4 != 0: # if result not match observation
                log2prob -= numpy.inf # log likelihood -inf
    return gs_stb, ps_stb, r, log2prob

@njit
def stabilizer_project(gs_stb, gs_obs, r):
    '''Project stabilizer tableau to a new stabilizer basis.

    Parameters:
    gs_stb: int (2*N, 2*N) - Pauli strings in original stabilizer tableau.
    gs_obs: int (L, 2*N) - Pauli strings of new stablizers to impose.
    r: int - log2 rank of density matrix (num of standby stablizers).

    Returns:
    gs_stb: int (2*N, 2*N) - Pauli strings in updated stabilizer tableau.
    r: int - updated log2 rank of density matrix.'''
    (L, Ng) = gs_obs.shape
    N = Ng//2
    assert 0<=r<=N
    for k in range(L): # loop over incoming projections gs_obs[k]
        update = False
        extend = False
        p = 0 # pointer to the first anticommuting operator
        for j in range(2*N):
            if acq(gs_stb[j], gs_obs[k]): # find gs_stb[j] anticommute with gs_obs[k]
                if update: # if gs_stb[j] is not the first anticommuting operator
                    gs_stb[j] = (gs_stb[j] + gs_stb[p])%2 # update gs_stb[j] to commute with gs_obs[k]
                else: # if gs_stb[j] is the first anticommuting operator
                    if j < N + r: # if gs_stb[j] is not an active destabilizer
                        p = j # move pointer to j
                        update = True
                        if not r <= j < N: # if gs_stb[j] is a standby operator
                            extend = True
                    # if gs_stb[j] is an active destablizer, gs_obs[k] alreay a combination of active stablizers, do nothing.
        if update:
            # now gs_stb[p] and gs_obs[k] anticommute
            q = (p+N)%(2*N) # get q as dual of p 
            gs_stb[q] = gs_stb[p] # move gs_stb[p] to gs_stb[q]
            gs_stb[p] = gs_obs[k] # add gs_obs[k] to gs_stb[p]
            if extend:
                r -= 1 # rank will reduce under extension
                # bring new stabilizer from p to r
                if p == r:
                    pass
                elif q == r:
                    gs_stb[numpy.array([p,q])] = gs_stb[numpy.array([q,p])] # swap p,q
                else:
                    s = (r+N)%(2*N) # get s as dual of r
                    gs_stb[numpy.array([p,r])] = gs_stb[numpy.array([r,p])] # swap p,r
                    gs_stb[numpy.array([q,s])] = gs_stb[numpy.array([s,q])] # swap q,s
    return gs_stb, r

@njit
def stabilizer_expect(gs_stb, ps_stb, gs_obs, ps_obs, r):
    '''Evaluate the expectation values of Pauli operators on a stabilizer state.

    Parameters:
    gs_stb: int (2*N, 2*N) - Pauli strings in original stabilizer tableau.
    ps_stb: int (N) - phase indicators of (de)stabilizers.
    gs_obs: int (L, 2*N) - strings of Pauli operators to be measured.
    ps_obs: int (L) - phase indicators of Pauli operators to be measured.
    r: int - log2 rank of density matrix (num of standby stablizers).

    Returns:
    xs: int (L) - expectation values of Pauli operators.'''
    (L, Ng) = gs_obs.shape
    N = Ng//2
    assert 0<=r<=N
    xs = numpy.empty(L, dtype=numpy.int_) # expectation values
    ga = numpy.empty(2*N, dtype=numpy.int_) # workspace for stabilizer accumulation
    pa = 0 # workspace for sign accumulation
    for k in range(L): # for each observable gs_obs[k] 
        ga[:] = 0
        pa = 0
        trivial = True # assuming gs_obs[k] is trivial in code subspace
        for j in range(2*N):
            if acq(gs_stb[j], gs_obs[k]): 
                if j < N + r: # if gs_stb[j] is active stablizer or standby.
                    xs[k] = 0 # gs_obs[k] is logical or error operator.
                    trivial = False # gs_obs[k] is not trivial
                    break
                else: # accumulate stablizer components
                    pa = (pa + ps_stb[j-N] + ipow(ga, gs_stb[j-N]))%4
                    ga = (ga + gs_stb[j-N])%2
        if trivial:
            xs[k] = (-1)**(((pa - ps_obs[k])%4)//2)
    return xs
    
@njit
def stabilizer_entropy(gs, mask):
    '''Entanglement entropy of the stabilizer state in a given region.

    Parameters:
    gs: int (L,2*N) - input stabilizers.
    mask: bool (N) - boolean vector specifying a subsystem.

    Returns:
    entropy: int - entanglement entropy in unit of bit (log2 based).

    Algorithm: 
        general case:
        entropy = # of subsystem qubits 
                - # of strictly inside stabilizers
                - # of hidden stabilizers (= nullity of gs across restricted to subsystem)

        pure state:
        entropy = 1/2 rank of (acq of gs across restricted to subsystem)
    '''
    (L, Ng) = gs.shape
    N = Ng//2
    mask2 = numpy.repeat(mask, 2)
    inside  = numpy.sum(gs[:,  mask2], -1) != 0
    outside = numpy.sum(gs[:, ~mask2], -1) != 0
    across = numpy.logical_and(inside, outside)
    gs_across_sub = gs[across][:, mask2]
    if L == N: # state is pure
        entropy = z2rank(acq_mat(gs_across_sub))//2
    else:
        strict = numpy.sum(inside) - numpy.sum(across)
        hidden = z2rank(gs_across_sub) - z2rank(acq_mat(gs_across_sub))
        entropy = numpy.sum(mask) - strict - hidden
    return entropy

# ---- Z2 linear algebra ----
@njit
def z2rank(mat):
    '''Calculate Z2 rank of a binary matrix.

    Parameters:
    mat: int matrix - input binary matrix.
        caller must ensure that mat contains only 0 and 1.
        mat is destroyed upon output! 

    Returns:
    r: int - rank of the matrix under Z2 algebra.'''
    nr, nc = mat.shape # get num of rows and cols
    r = 0 # current row index
    for i in range(nc): # run through cols
        if r == nr: # row exhausted first
            return r # row rank is full, early return
        if mat[r, i] == 0: # need to find pivot
            found = False # set a flag
            for k in range(r + 1, nr):
                if mat[k, i]: # mat[k, i] nonzero
                    found = True # pivot found in k
                    break
            if found: # if pivot found in k
                # swap rows r, k
                for j in range(i, nc):
                    tmp = mat[k,j]
                    mat[k,j] = mat[r, j]
                    mat[r,j] = tmp
            else: # if pivot not found
                continue # done with this col
        # pivot has moved to mat[r, i], perform GE
        for j in range(r + 1, nr):
            if mat[j, i]: # mat[j, i] nonzero
                mat[j, i:] = (mat[j, i:] + mat[r, i:])%2
        r = r + 1 # rank inc
    # col exhausted, last nonvanishing row indexed by r
    return r

@njit
def z2inv(mat):
    '''Calculate Z2 inversion of a binary matrix.'''
    assert mat.shape[0] == mat.shape[1] # assuming matrix is square
    n = mat.shape[0] # get matrix dimension
    a = numpy.zeros((n,2*n), dtype=mat.dtype) # prepare a workspace
    a[:,:n] = mat # copy matrix to the left part
    # create a diagonal matrix on the right part
    for i in range(n):
        a[i, i+n] = 1
    # forward pass
    for i in range(n): # run through cols
        if a[i, i] == 0: # need to find pivot
            found = False # set a flag
            for k in range(i + 1, n):
                if a[k, i]: # a[k, i] nonzero
                    found  = True # pivot found at k
                    break
            if found: # if pivot found at k
                # swap rows i, k
                for j in range(i, 2*n):
                    tmp = a[k, j]
                    a[k, j] = a[i, j]
                    a[i, j] = tmp
            else: # if pivot not found, matrix not invertable
                raise ValueError('binary matrix not invertable.')
        # pivot has moved to a[i, i], perform GE
        for j in range(i + 1, n):
            if a[j, i]: # a[j, i] nonzero
                a[j, i:] = (a[j, i:] + a[i, i:])%2
    # backward pass
    for i in range(n-1,0,-1):
        for j in range(i):
            if a[j, i]: # a[j, i] nonzero
                a[j, i:] = (a[j, i:] + a[i, i:])%2
    return a[:,n:]

# ---- auxilary functions ----
def mask(qubits, N):
    '''Create a mask vector for a subsystem of qubits.

    Parameters:
    qubits: int (n) - a subsystem specified by qubit indices.
    N: int - total system size.

    Returns:
    mask: bool (N) -  a boolean vector with True at specified qubits.

    Note: complement region is ~mask.'''
    assert max(qubits) < N, 'qubit {} is out of bounds for system size {}.'.format(max(qubits), N)
    mask = numpy.zeros(N, dtype=numpy.bool_)
    mask[numpy.array(qubits)] = True
    return mask

def binary_repr(ints, width = None):
    '''Convert an array of integers to their binary representations.
    
    Parameters:
    ints: int array - array of integers.
    width: width of the binary representation (default: determined by the bit length of the maximum int).
    
    Returns:
    new array where each integter is unpacked to binary subarray.
    '''
    width = numpy.ceil(numpy.log2(numpy.max(ints)+1)).astype(int) if width is None else width
    dt0 = ints.dtype
    dt1 = numpy.dtype((dt0, [('bytes','u1',dt0.itemsize)]))
    bins = numpy.unpackbits(ints.view(dtype=dt1)['bytes'], axis=-1, bitorder='little')
    return numpy.flip(bins, axis=-1)[...,-width:]

@njit
def aggregate(data_in, inds, l):
    '''Aggregate data (1d array) by unique inversion indices.

    Parameter:
    data_in: any (L) - input data array.
    inds: int (L) - indices that each element should be mapped to.
    l : int - number of unique elements in data_in.

    Returns:
    data_out: any (l) - output data array.'''
    data_out = numpy.zeros(l, dtype=data_in.dtype)
    for i in range(data_in.shape[0]):
        data_out[inds[i]] += data_in[i]
    return data_out

# ---- generalized stabilizer utilities ----
@njit
def calculate_chi(chi_old, phi, fusion_map, fusion_p, L_new):
    L_old, L_add = fusion_map.shape
    chi_new = numpy.zeros((L_new,L_new), dtype=numpy.complex128)
    for i1 in range(L_old):
        for j1 in range(L_add):
            k1 = fusion_map[i1,j1]
            for i2 in range(L_old):
                for j2 in range(L_add):
                    k2 = fusion_map[i2,j2]
                    chi_new[k1,k2] += chi_old[i1,i2] * phi[j1,j2] * 1j**(fusion_p[i1,j1] + fusion_p[i2,j2])
    return chi_new