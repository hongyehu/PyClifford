import torch
import numpy as np

from ..utils import front, condense, acq, ipow, ps0, acq_mat, batch_dot, pauli_tokenize, clifford_rotate, pauli_is_onsite, pauli_diagonalize1, pauli_diagonalize2, random_pair, random_pauli, \
        impose_leading_noncommutivity, build_pauli_map, map_to_state, state_to_map, pauli_combine, pauli_transform, mask, binary_repr, aggregate, z2rank, z2inv, stabilizer_project, \
        stabilizer_measure, stabilizer_expect, acq_grid, ipow_product, stabilizer_projection_trace


device = 'cpu'
#device = 'cuda:0'


def np_front(g):
    (N2,) = g.shape
    N = N2//2
    for i in range(N):
        if g[2*i] != 0 or g[2*i+1] != 0:
            break
    return i

def np_condense(g):
    (N2,) = g.shape
    N = N2//2
    mask = np.zeros(N, dtype=np.bool_)
    for i in range(N):
        if g[2*i] != 0 or g[2*i+1] != 0:
            mask[i] = True
    qubits = np.arange(N)[mask]
    return g[np.repeat(mask, 2)], qubits

def np_acq(g1, g2):
    assert g1.shape == g2.shape
    (N2,) = g1.shape
    N = N2//2
    acq = 0
    for i in range(N):
        acq += g1[2*i+1]*g2[2*i] - g1[2*i]*g2[2*i+1]
    return acq % 2

def np_ipow(g1, g2):
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

def np_p0(g):
    (N2,) = g.shape
    N = N2//2
    p0 = 0
    for i in range(N):
        p0 += g[2*i] * g[2*i+1]
    return p0 % 4

def np_ps0(gs):
    (L, N2) = gs.shape
    N = N2//2
    ps0 = np.zeros(L, dtype=np.int_)
    for j in range(L):
        for i in range(N):
            ps0[j] += gs[j,2*i] * gs[j,2*i+1]
    return ps0 % 4

def np_acq_mat(gs):
    (L, N2) = gs.shape
    N = N2//2
    mat = np.zeros((L,L), dtype=np.int_)
    for j1 in range(L):
        for j2 in range(L):
            for i in range(N):
                mat[j1,j2] += gs[j1,2*i+1]*gs[j2,2*i] - gs[j1,2*i]*gs[j2,2*i+1]
    mat = mat % 2
    return mat

def np_batch_dot(gs1, ps1, cs1, gs2, ps2, cs2):
    (L1, N2) = gs1.shape
    (L2, N2) = gs2.shape
    gs = np.empty((L1,L2,N2), dtype=np.int_)
    ps = np.empty((L1,L2), dtype=np.int_)
    cs = np.empty((L1,L2), dtype=np.complex_)
    for j1 in range(L1):
        for j2 in range(L2):
            ps[j1,j2] = (ps1[j1] + ps2[j2] + np_ipow(gs1[j1], gs2[j2]))%4
            gs[j1,j2] = (gs1[j1] + gs2[j2])%2
            cs[j1,j2] = cs1[j1] * cs2[j2]
    gs = np.reshape(gs, (L1*L2,-1))
    ps = np.reshape(ps, (L1*L2,))
    cs = np.reshape(cs, (L1*L2,))
    return gs, ps, cs

def np_pauli_tokenize(gs, ps):
    (L, N2) = gs.shape
    N = N2//2
    ts = np.zeros((L,N+1), dtype=np.int_)
    for j in range(L):
        for i in range(N):
            ts[j,i] = 3*gs[j,2*i+1] + (-1)**gs[j,2*i+1] * gs[j,2*i]
        x = ps[j]
        ts[j,N] = 4 + x * (11 - 9 * x + 2 * x**2) // 2
    return ts

def np_pauli_combine(C, gs_in, ps_in):
    (L_out, L_in) = C.shape
    N2 = gs_in.shape[-1]
    gs_out = np.zeros((L_out, N2), dtype=np.int_) # identity
    ps_out = np.zeros((L_out,), dtype=np.int_)
    for j_out in range(L_out):
        for j_in in range(L_in):
            if C[j_out, j_in]:
                ps_out[j_out] = (ps_out[j_out] + ps_in[j_in] + np_ipow(gs_out[j_out], gs_in[j_in]))%4
                gs_out[j_out] = (gs_out[j_out] + gs_in[j_in])%2
    return gs_out, ps_out

def np_clifford_rotate(g, p, gs, ps):
    (L, N2) = gs.shape
    for j in range(L):
        if np_acq(g, gs[j]):
            ps[j] = (ps[j] + p + 1 + np_ipow(gs[j], g))%4
            gs[j] = (gs[j] + g)%2
    return gs, ps

def np_pauli_is_onsite(g, i0=0):
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

def np_pauli_diagonalize1(g1, i0 = 0):
    (N2,) = g1.shape
    N = N2//2
    gs = [] # prepare to collect Clifford generators
    if not (np_pauli_is_onsite(g1, i0) and g1[2*i0] == 0): # if g1 is not on site and diagonal
        if g1[2*i0] == 0: # if g1 commute with Z0
            g = g1.copy()
            if g1[2*i0+1] == 0: # g1 is trivial at site 0
                i = np_front(g) # find the first non-trivial qubit as pivot
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

def np_pauli_diagonalize2(g1, g2, i0=0):
    assert g1.shape == g2.shape
    (N2,) = g1.shape
    N = N2//2
    gs = [] # prepare to collect Clifford generators
    # bring g1 to Z0
    if not (np_pauli_is_onsite(g1, i0) and g1[2*i0] == 0): # if g1 is not on site and diagonal
        if g1[2*i0] == 0: # if g1 commute with Z0
            g = g1.copy()
            if g1[2*i0+1] == 0: # g1 is trivial at site 0
                i = np_front(g) # find the first non-trivial qubit as pivot
                # XYZ cyclic on the pivot qubit
                g[2*i] = (g[2*i] + g[2*i+1])%2
                g[2*i+1] = (g[2*i+1] + g[2*i])%2
                # now g anticommute with g1
            g[2*i0] = 1 # such that g also anticommute with Z0
            gs.append(g)
            g1 = (g1 + g)%2
            g2 = (g2 + np_acq(g, g2) * g)%2
        # now g1 anticommute with Z0                
        g = g1.copy()
        g[2*i0+1] = (g[2*i0+1] + 1)%2 # g = g1 (*) Z0
        gs.append(g)
        g1 = (g1 + g)%2
        g2 = (g2 + np_acq(g, g2) * g)%2
        # now g1 has been transformed to Z0
    # bring g2 to X0,Y0
    if not np_pauli_is_onsite(g2, i0): # if g2 is not on site
        g = g2.copy()
        g[2*i0] = 0
        g[2*i0+1] = 1
        gs.append(g)
        g2 = (g2 + g)%2
        # now g2 has been transformed to X0 or Y0
    return gs, g1, g2

def np_random_pair(N, gs=None):
    if gs == None:
        g1 = np.random.randint(0,2,2*N)
        g2 = np.random.randint(0,2,2*N)
        while (g1 == 0).all(): # resample g1 if it is all zero
            g1 = np.random.randint(0,2,2*N)
    else:
        g1, g2 = gs
        if np_acq(g1, g2) == 0: # if g1, g2 commute
            i = np_front(g1) # locate the first nontrivial g1 site
            # flip commutativity by chaning g2
            g2[2*i] = (g2[2*i] + g1[2*i+1])%2
            g2[2*i+1] = (g2[2*i+1] + g1[2*i] + g1[2*i+1])%2
    return g1, g2

def np_random_pauli(N, gtuple=None):
    if gtuple == None:
        gtuple = [np_random_pair(1) for _ in range(N)]
    else:
        gtuple = [(gtuple[0][i], gtuple[1][i]) for i in range(N)]
    gs = np.zeros((2*N,2*N), dtype=np.int_)
    for i in range(N):
        g1, g2 = gtuple[i]
        gs[2*i  ,2*i:2*i+2] = g1
        gs[2*i+1,2*i:2*i+2] = g2
    return gs

def np_map_to_state(gs_in, ps_in):
    (L, N2) = gs_in.shape
    N = N2//2
    gs_out = np.empty_like(gs_in)
    ps_out = np.empty_like(ps_in)
    for i in range(N):
        gs_out[N+i] = gs_in[2*i]
        gs_out[i] = gs_in[2*i+1]
        ps_out[N+i] = ps_in[2*i]
        ps_out[i] = ps_in[2*i+1]
    return gs_out, ps_out

def np_state_to_map(gs_in, ps_in):
    (L, N2) = gs_in.shape
    N = N2//2
    gs_out = np.empty_like(gs_in)
    ps_out = np.empty_like(ps_in)
    for i in range(N):
        gs_out[2*i] = gs_in[N+i]
        gs_out[2*i+1] = gs_in[i]
        ps_out[2*i] = ps_in[N+i]
        ps_out[2*i+1] = ps_in[i]
    return gs_out, ps_out

def np_pauli_combine(C, gs_in, ps_in):
    (L_out, L_in) = C.shape
    N2 = gs_in.shape[-1]
    gs_out = np.zeros((L_out, N2), dtype=np.int_) # identity
    ps_out = np.zeros((L_out,), dtype=np.int_)
    for j_out in range(L_out):
        for j_in in range(L_in):
            if C[j_out, j_in]:
                ps_out[j_out] = (ps_out[j_out] + ps_in[j_in] + np_ipow(gs_out[j_out], gs_in[j_in]))%4
                gs_out[j_out] = (gs_out[j_out] + gs_in[j_in])%2
    return gs_out, ps_out

def np_pauli_transform(gs_in, ps_in, gs_map, ps_map):
    gs_out, ps_out = np_pauli_combine(gs_in, gs_map, ps_map)
    ps_out = (ps_in + np_ps0(gs_in) + ps_out)%4
    return gs_out, ps_out

def np_z2rank(mat):
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

def np_z2inv(mat):
    assert mat.shape[0] == mat.shape[1] # assuming matrix is square
    n = mat.shape[0] # get matrix dimension
    a = np.zeros((n,2*n), dtype=mat.dtype) # prepare a workspace
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

def np_mask(qubits, N):
    assert max(qubits) < N, 'qubit {} is out of bounds for system size {}.'.format(max(qubits), N)
    mask = np.zeros(N, dtype=np.bool_)
    mask[np.array(qubits)] = True
    return mask

def np_binary_repr(ints, width = None):
    width = np.ceil(np.log2(np.max(ints)+1)).astype(int) if width is None else width
    dt0 = ints.dtype
    dt1 = np.dtype((dt0, [('bytes','u1',dt0.itemsize)]))
    bins = np.unpackbits(ints.view(dtype=dt1)['bytes'], axis=-1, bitorder='little')
    return np.flip(bins, axis=-1)[...,-width:]

def np_aggregate(data_in, inds, l):
    data_out = np.zeros(l, dtype=data_in.dtype)
    for i in range(data_in.shape[0]):
        data_out[inds[i]] += data_in[i]
    return data_out

def np_stabilizer_project(gs_stb, gs_obs, r):
    (L, Ng) = gs_obs.shape
    N = Ng//2
    assert 0<=r<=N
    for k in range(L): # loop over incoming projections gs_obs[k]
        update = False
        extend = False
        p = 0 # pointer
        for j in range(2*N):
            if np_acq(gs_stb[j], gs_obs[k]): # find gs_stb[j] anticommute with gs_obs[k]
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
                    gs_stb[np.array([p,q])] = gs_stb[np.array([q,p])] # swap p,q
                else:
                    s = (r+N)%(2*N) # get s as dual of r
                    gs_stb[np.array([p,r])] = gs_stb[np.array([r,p])] # swap p,r
                    gs_stb[np.array([q,s])] = gs_stb[np.array([s,q])] # swap q,s
    return gs_stb, r

def np_stabilizer_measure(gs_stb, ps_stb, gs_obs, ps_obs, r, samples):
    (L, Ng) = gs_obs.shape
    N = Ng//2
    assert 0<=r<=N
    out = np.empty(L, dtype=np.int_)
    ga = np.empty(2*N, dtype=np.int_) # workspace for stabilizer accumulation
    pa = 0 # workspace for phase accumulation
    log2prob = 0.
    for k in range(L): # for each observable gs_obs[k]
        update = False
        extend = False
        p = 0 # pointer
        ga[:] = 0
        pa = 0
        for j in range(2*N):
            if np_acq(gs_stb[j], gs_obs[k]): # find gs_stb[j] anticommute with gs_obs[k]
                if update: # if gs_stb[j] is not the first anticommuting operator
                    # update gs_stb[j] to commute with gs_obs[k]
                    if j < N: # if gs_stb[j] is a stablizer, phase matters
                        ps_stb[j] = (ps_stb[j] + ps_stb[p] + np_ipow(gs_stb[j], gs_stb[p]))%4
                    gs_stb[j] = (gs_stb[j] + gs_stb[p])%2
                else: # if gs_stb[j] is the first anticommuting operator
                    if j < N + r: # if gs_stb[j] is not an active destabilizer
                        p = j # move pointer to j
                        update = True
                        if not r <= j < N: # if gs_stb[j] is a standby operator
                            extend = True
                    else: # gs_stb[j] anticommute with destabilizer, meaning gs_obs[k] already a combination of active stabilizers
                        # collect corresponding stabilizer component to ga
                        pa = (pa + ps_stb[j-N] + np_ipow(ga, gs_stb[j-N]))%4
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
                    gs_stb[np.array([p,q])] = gs_stb[np.array([q,p])] # swap p,q
                else:
                    s = (r+N)%(2*N) # get s as dual of r
                    gs_stb[np.array([p,r])] = gs_stb[np.array([r,p])] # swap p,r
                    gs_stb[np.array([q,s])] = gs_stb[np.array([s,q])] # swap q,s
                p = r
            # as long as gs_obs[k] is not eigen, outcome will be half-to-half
            ps_stb[p] = 2 * samples[k] #np.random.randint(2)
            out[k] = ((ps_stb[p] - ps_obs[k])%4)//2 #0->0(+1 eigenvalue), 2->1(-1 eigenvalue)
            log2prob -= 1.
        else: # no update, gs_obs[k] is eigen, result is in pa
            out[k] = ((pa - ps_obs[k])%4)//2
    return gs_stb, ps_stb, r, out, log2prob

def np_stabilizer_expect(gs_stb, ps_stb, gs_obs, ps_obs, r):
    (L, Ng) = gs_obs.shape
    N = Ng//2
    assert 0<=r<=N
    xs = np.empty(L, dtype=np.int_) # expectation values
    ga = np.empty(2*N, dtype=np.int_) # workspace for stabilizer accumulation
    pa = 0 # workspace for sign accumulation
    for k in range(L): # for each observable gs_obs[k] 
        ga[:] = 0
        pa = 0
        trivial = True # assuming gs_obs[k] is trivial in code subspace
        for j in range(2*N):
            if np_acq(gs_stb[j], gs_obs[k]): 
                if j < N + r: # if gs_stb[j] is active stablizer or standby.
                    xs[k] = 0 # gs_obs[k] is logical or error operator.
                    trivial = False # gs_obs[k] is not trivial
                    break
                else: # accumulate stablizer components
                    pa = (pa + ps_stb[j-N] + np_ipow(ga, gs_stb[j-N]))%4
                    ga = (ga + gs_stb[j-N])%2
        if trivial:
            xs[k] = (-1)**(((pa - ps_obs[k])%4)//2)
    return xs


def np_stabilizer_projection_trace(gs_stb, ps_stb, gs_obs, ps_obs, r):
    (L, Ng) = gs_obs.shape
    N = Ng//2
    assert 0<=r<=N
    ga = np.empty(2*N, dtype=np.int_)
    pa = 0
    trace = 1
    for k in range(L):
        update = False
        extend = False
        p = 0 # pointer
        ga[:] = 0
        pa = 0
        for j in range(2*N):
            if np_acq(gs_stb[j], gs_obs[k]):
                if update:
                    if j < N:
                        ps_stb[j] = (ps_stb[j] + ps_stb[p] + np_ipow(gs_stb[j], gs_stb[p]))%4
                    gs_stb[j] = (gs_stb[j] + gs_stb[p])%2
                else:
                    if j < N + r:
                        p = j
                        update = True
                        if not r <= j < N:
                            extend = True
                    else:
                        pa = (pa + ps_stb[j-N] + np_ipow(ga, gs_stb[j-N]))%4
                        ga = (ga + gs_stb[j-N])%2
        if update:
            q = (p+N)%(2*N)
            gs_stb[q] = gs_stb[p]
            gs_stb[p] = gs_obs[k]
            if extend:
                r -= 1
                if p == r:
                    pass
                elif q == r:
                    gs_stb[np.array([p,q])] = gs_stb[np.array([q,p])]
                else:
                    s = (r+N)%(2*N)
                    gs_stb[np.array([p,r])] = gs_stb[np.array([r,p])]
                    gs_stb[np.array([q,s])] = gs_stb[np.array([s,q])]
                p = r
            ps_stb[p] = ps_obs[k]
            trace = trace/2.
        else:
            if not pa == ps_obs[k]:
                trace = 0.
    return gs_stb, ps_stb, r, trace


def test_front():
    nqubits = np.random.randint(1, 10)
    g = np.random.randint(2, size=2*nqubits)
    assert np.allclose(front(torch.tensor(g, device=device)).cpu(), np_front(g))


def test_condense():
    nqubits = np.random.randint(1, 10)
    g = np.random.randint(2, size=2*nqubits)
    state, indices = condense(torch.tensor(g, device=device))
    true_state, true_indices = np_condense(g)
    assert np.allclose(state.cpu(), true_state)
    assert np.allclose(indices.cpu(), true_indices)


def test_acq():
    nqubits = np.random.randint(1, 10)
    ga = np.random.randint(2, size=2*nqubits)
    gb = np.random.randint(2, size=2*nqubits)
    assert np.allclose(acq(torch.tensor(ga, device=device), torch.tensor(gb, device=device)).cpu(), np_acq(ga, gb))
    La = np.random.randint(1, 10)
    ga = np.random.randint(2, size=(La, 2*nqubits))
    gb = np.random.randint(2, size=(La, 2*nqubits))
    true_acq_grid = np.zeros((La,))
    for i in range(La):
        true_acq_grid[i] = np_acq(ga[i], gb[i])
    assert np.allclose(acq(torch.tensor(ga, device=device), torch.tensor(gb, device=device)).cpu(), true_acq_grid)


def test_acq_grid():
    nqubits = np.random.randint(1, 10)
    ga = np.array(np.random.randint(2, size=2*nqubits), dtype=float)
    gb = np.array(np.random.randint(2, size=2*nqubits), dtype=float)
    assert np.allclose(acq(torch.tensor(ga, device=device), torch.tensor(gb, device=device)).cpu(), np_acq(ga, gb))
    La = np.random.randint(1, 10)
    Lb = np.random.randint(1, 10)
    ga = np.array(np.random.randint(2, size=(La, 2*nqubits)), dtype=float)
    gb = np.array(np.random.randint(2, size=(Lb, 2*nqubits)), dtype=float)
    true_acq_grid = np.zeros((La,Lb))
    for i in range(La):
        for j in range(Lb):
            true_acq_grid[i, j] = np_acq(ga[i], gb[j])
    assert np.allclose(acq_grid(torch.tensor(ga, device=device), torch.tensor(gb, device=device)).cpu(), true_acq_grid)


def test_ipow():
    nqubits = np.random.randint(1, 10)
    ga = np.random.randint(2, size=2*nqubits)
    gb = np.random.randint(2, size=2*nqubits)
    assert np.allclose(ipow(torch.tensor(ga, device=device), torch.tensor(gb, device=device)).cpu(), np_ipow(ga, gb))
    La = np.random.randint(1, 10)
    Lb = np.random.randint(1, 10)
    ga = np.random.randint(2, size=(La, 2*nqubits))
    gb = np.random.randint(2, size=(La, 2*nqubits))
    true_ipow, count = np.zeros((La,)), 0
    for i in range(La):
        true_ipow[count] = np_ipow(ga[i, :], gb[i, :])
        count += 1
    assert np.allclose(ipow(torch.tensor(ga, device=device), torch.tensor(gb, device=device)).cpu(), true_ipow)


def test_ipow_product():
    nqubits = np.random.randint(1, 10)
    ga = np.random.randint(2, size=2*nqubits)
    gb = np.random.randint(2, size=2*nqubits)
    assert np.allclose(ipow(torch.tensor(ga, device=device), torch.tensor(gb, device=device)).cpu(), np_ipow(ga, gb))
    La = np.random.randint(1, 10)
    Lb = np.random.randint(1, 10)
    ga = np.random.randint(2, size=(La, 2*nqubits))
    gb = np.random.randint(2, size=(Lb, 2*nqubits))
    true_ipow, count = np.zeros((La*Lb,)), 0
    for i in range(La):
        for j in range(Lb):
            true_ipow[count] = np_ipow(ga[i, :], gb[j, :])
            count += 1
    assert np.allclose(ipow_product(torch.tensor(ga, device=device), torch.tensor(gb, device=device)).cpu(), true_ipow)


def test_ps0():
    nqubits = np.random.randint(1, 10)
    g = np.random.randint(2, size=2*nqubits)
    assert np.allclose(ps0(torch.tensor(g, device=device)).cpu(), np_p0(g))
    L = np.random.randint(1, 10)
    g = np.random.randint(2, size=(L, 2*nqubits))
    assert np.allclose(ps0(torch.tensor(g, device=device)).cpu(), np_ps0(g))


def test_acq_mat():
    nqubits = np.random.randint(1, 10)
    L = np.random.randint(1, 10)
    g = np.array(np.random.randint(2, size=(L, 2*nqubits)), dtype=float)
    assert np.allclose(acq_mat(torch.tensor(g, device=device)).cpu(), np_acq_mat(g))


def test_batch_dot():
    nqubits = np.random.randint(1, 10)
    L1, L2 = np.random.randint(1, 10), np.random.randint(1, 10)
    g1, g2 = np.random.randint(2, size=(L1, 2*nqubits)), np.random.randint(2, size=(L2, 2*nqubits))
    p1, p2 = np.random.randint(2, size=L1), np.random.randint(2, size=L2)
    c1, c2 = p1.copy(), p2.copy()
    gs, ps, cs = batch_dot(torch.tensor(g1, device=device), torch.tensor(p1, device=device), torch.tensor(c1, device=device), torch.tensor(g2, device=device), torch.tensor(p2, device=device), torch.tensor(c2, device=device))
    true_gs, true_ps, true_cs = np_batch_dot(g1, p1, c1, g2, p2, c2)
    assert np.allclose(gs.cpu(), true_gs)
    assert np.allclose(ps.cpu(), true_ps)
    assert np.allclose(cs.cpu(), true_cs)


def test_pauli_tokenize():
    nqubits = np.random.randint(1, 10)
    L1 = np.random.randint(1, 10)
    g1 = np.random.randint(4, size=(L1, 2*nqubits))
    p1 = np.random.randint(4, size=L1)
    assert np.allclose(pauli_tokenize(torch.tensor(g1, device=device), torch.tensor(p1, device=device)).cpu(), np_pauli_tokenize(g1, p1))


def test_pauli_combine():
    nqubits = np.random.randint(1, 10)
    L_out, L_in = np.random.randint(1, 10), np.random.randint(1, 10)
    C = np.random.randint(2, size=(L_out, L_in))
    gs_in = np.random.randint(2, size=(L_in, 2*nqubits))
    ps_in = np.random.randint(2, size=(L_in,))
    gs_out, ps_out = pauli_combine(torch.tensor(C, device=device), torch.tensor(gs_in, device=device), torch.tensor(ps_in, device=device))
    true_gs_out, true_ps_out = np_pauli_combine(C, gs_in, ps_in)
    assert np.allclose(gs_out.cpu(), true_gs_out)
    assert np.allclose(ps_out.cpu(), true_ps_out)


def test_pauli_transform():
    nqubits = np.random.randint(1, 10)
    L = np.random.randint(1, 10)
    gs_in = np.array(np.random.randint(2, size=(L, 2*nqubits)), dtype=float)
    ps_in = np.array(np.random.randint(2, size=(L,)), dtype=float)
    gs_map = np.array(np.random.randint(2, size=(2*nqubits, 2*nqubits)), dtype=float)
    ps_map = np.array(np.random.randint(2, size=(2*nqubits,)), dtype=float)
    gs_out, ps_out = pauli_transform(torch.tensor(gs_in, device=device), torch.tensor(ps_in, device=device), torch.tensor(gs_map, device=device), torch.tensor(ps_map, device=device))
    true_gs_out, true_ps_out = np_pauli_transform(gs_in, ps_in, gs_map, ps_map)
    assert np.allclose(gs_out.cpu(), true_gs_out)
    assert np.allclose(ps_out.cpu(), true_ps_out)


def test_clifford_rotate():
    nqubits = np.random.randint(1, 10)
    L = np.random.randint(1, 10)
    g, gs = np.random.randint(2, size=(2*nqubits,)), np.random.randint(2, size=(L, 2*nqubits))
    p, ps = np.random.randint(2, size=1), np.random.randint(2, size=L)
    output_gs, output_ps = clifford_rotate(torch.tensor(g, device=device), torch.tensor(p, device=device), torch.tensor(gs, device=device), torch.tensor(ps, device=device))
    true_output_gs, true_output_ps = np_clifford_rotate(g, p, gs, ps)
    assert np.allclose(output_gs.cpu(), true_output_gs)
    assert np.allclose(output_ps.cpu(), true_output_ps)


def test_pauli_is_onsight():
    nqubits = np.random.randint(1, 10)
    i0 = np.random.randint(0, nqubits)
    g = np.random.randint(2, size=(2*nqubits,))
    assert np.allclose(pauli_is_onsite(torch.tensor(g, device=device), i0=i0).cpu(), np_pauli_is_onsite(g, i0=i0))


def test_pauli_diagonalize1():
    nqubits = np.random.randint(1, 10)
    i0 = np.random.randint(0, nqubits)
    g = np.random.randint(2, size=(2*nqubits,))
    for diagonalized, true_diagonalized in zip(pauli_diagonalize1(torch.tensor(g, device=device)), np_pauli_diagonalize1(g)):
        assert np.allclose(diagonalized.cpu(), true_diagonalized)


def test_pauli_diagonalize2():
    nqubits = np.random.randint(1, 10)
    i0 = np.random.randint(0, nqubits)
    g1 = np.random.randint(2, size=(2*nqubits,))
    g2 = np.random.randint(2, size=(2*nqubits,))
    gs_out, g1_out, g2_out = pauli_diagonalize2(torch.tensor(g1, device=device), torch.tensor(g2, device=device))
    true_gs_out, true_g1_out, true_g2_out = np_pauli_diagonalize2(g1, g2)
    for gs, true_gs in zip(gs_out, true_gs_out):
        assert np.allclose(gs.cpu(), true_gs)
    assert np.allclose(g1_out.cpu(), true_g1_out)
    assert np.allclose(g2_out.cpu(), true_g2_out)


def test_random_pair():
    nqubits = np.random.randint(1, 10)
    g1, g2 = np.random.randint(0,2,2*nqubits), np.random.randint(0,2,2*nqubits)
    while (g1 == 0).all():
        g1 = np.random.randint(0,2,2*nqubits)
    g1_out, g2_out = impose_leading_noncommutivity(torch.tensor(g1, device=device), torch.tensor(g2, device=device))
    g1_true_out, g2_true_out = np_random_pair(nqubits, gs=(g1, g2))
    assert np.allclose(g1_out.cpu(), g1_true_out)
    assert np.allclose(g2_out.cpu(), g2_true_out)
    L = np.random.randint(1, 10)
    g1, g2 = np.random.randint(0, 2, (L, 2*nqubits)), np.random.randint(0, 2, (L, 2*nqubits))
    g1_out, g2_out = impose_leading_noncommutivity(torch.tensor(g1, device=device), torch.tensor(g2, device=device))
    g1_true_out, g2_true_out = np.zeros((L, 2*nqubits)), np.zeros((L, 2*nqubits))
    for i in range(L):
        g1_true_out[i], g2_true_out[i] = np_random_pair(nqubits, gs=(g1[i], g2[i]))
    assert np.allclose(g1_out.cpu(), g1_true_out)
    assert np.allclose(g2_out.cpu(), g2_true_out)
    g1, g2 = random_pair(nqubits, L=L)
    if g1.dim() > 1:
        assert g1.shape[0] == g2.shape[0] == L
    assert g1.shape[-1] == g2.shape[-1] == 2*nqubits
    assert torch.all(g1>=0) and torch.all(g1<=1)
    assert torch.all(g2>=0) and torch.all(g2<=1)


def test_random_pauli():
    nqubits = np.random.randint(2, 10)
    g1, g2 = random_pair(1, nqubits, device=device)
    gs = build_pauli_map(nqubits, g1, g2)
    true_gs = np_random_pauli(nqubits, gtuple=(g1.cpu().numpy(), g2.cpu().numpy()))
    assert np.allclose(gs.cpu(), true_gs)


def test_map_to_state():
    nqubits = np.random.randint(1, 10)
    gs = np.random.randint(2, size=(2*nqubits, 2*nqubits))
    ps = np.random.randint(2, size=2*nqubits)
    gs_out, ps_out = map_to_state(torch.tensor(gs, device=device), torch.tensor(ps, device=device))
    true_gs_out, true_ps_out = np_map_to_state(gs, ps)
    assert np.allclose(gs_out.cpu(), true_gs_out)
    assert np.allclose(ps_out.cpu(), true_ps_out)


def test_state_to_map():
    nqubits = np.random.randint(1, 10)
    gs = np.random.randint(2, size=(2*nqubits, 2*nqubits))
    ps = np.random.randint(2, size=2*nqubits)
    gs_out, ps_out = state_to_map(torch.tensor(gs, device=device), torch.tensor(ps, device=device))
    true_gs_out, true_ps_out = np_state_to_map(gs, ps)
    assert np.allclose(gs_out.cpu(), true_gs_out)
    assert np.allclose(ps_out.cpu(), true_ps_out)


def test_stabilizer_project():
    nqubits, L = np.random.randint(2, 10), np.random.randint(2, 10)
    r = np.random.randint(1, nqubits)
    gs_stb = np.random.randint(2, size=(2*nqubits, 2*nqubits))
    gs_obs = np.random.randint(2, size=(L, 2*nqubits))
    output_gs_stb, output_r = stabilizer_project(torch.tensor(gs_stb, device=device), torch.tensor(gs_obs, device=device), r)
    true_output_gs_stb, true_output_r = np_stabilizer_project(gs_stb, gs_obs, r)
    assert np.allclose(output_gs_stb.cpu(), true_output_gs_stb)
    assert np.allclose(output_r, true_output_r)


def test_stabilizer_measure():
    nqubits, L = np.random.randint(2, 10), np.random.randint(2, 10)
    r = 1
    gs_stb, ps_stb = np.random.randint(2, size=(2*nqubits, 2*nqubits)), np.random.randint(4, size=(nqubits,))
    gs_obs, ps_obs = np.random.randint(2, size=(L, 2*nqubits)), np.random.randint(4, size=(L,))
    gs_stb_copy, ps_stb_copy, gs_obs_copy, ps_obs_copy, r_copy = gs_stb.copy(), ps_stb.copy(), gs_obs.copy(), ps_obs.copy(), r
    samples = torch.randint(0, 2, (L,))
    out_gs_stb, out_ps_stb, out_r, out_out, out_log2prob = stabilizer_measure(torch.tensor(gs_stb, device=device), torch.tensor(ps_stb, device=device), torch.tensor(gs_obs, device=device), torch.tensor(ps_obs, device=device), r, samples=samples)
    true_out_gs_stb, true_out_ps_stb, true_out_r, true_out_out, true_out_log2prob = np_stabilizer_measure(gs_stb_copy, ps_stb_copy, gs_obs_copy, ps_obs_copy, r_copy, samples.numpy())
    assert np.allclose(out_gs_stb.cpu(), true_out_gs_stb)
    assert np.allclose(out_ps_stb.cpu(), true_out_ps_stb)
    assert np.allclose(out_r, true_out_r)
    assert np.allclose(out_out.cpu(), true_out_out)
    assert np.allclose(out_log2prob, true_out_log2prob)


def test_stabilizer_expect():
    nqubits, L = np.random.randint(2, 10), np.random.randint(2, 10)
    r = np.random.randint(1, nqubits)
    gs_stb, ps_stb = np.random.randint(2, size=(2*nqubits, 2*nqubits)), np.random.randint(4, size=(nqubits,))
    gs_obs, ps_obs = np.random.randint(2, size=(L, 2*nqubits)), np.random.randint(4, size=(L,))
    xs = stabilizer_expect(torch.tensor(gs_stb, device=device), torch.tensor(ps_stb, device=device), torch.tensor(gs_obs, device=device), torch.tensor(ps_obs, device=device), r)
    true_xs = np_stabilizer_expect(gs_stb, ps_stb, gs_obs, ps_obs, r)
    assert np.allclose(xs.cpu(), true_xs)


def test_mask():
    N = np.random.randint(5, 10)
    upper_qubit = np.random.randint(3, N-1)
    nvals = np.random.randint(1, upper_qubit-1)
    qubits = np.random.choice(upper_qubit, nvals, replace=False)
    m = mask(qubits, N)
    true_m = np_mask(qubits, N)
    assert np.allclose(m, true_m)


def test_binary_repr():
    N = np.random.randint(3, 10)
    nvals = np.random.randint(1, N-1)
    integers = np.random.choice(N, nvals, replace=False)
    br = binary_repr(torch.tensor(integers, device=device))
    true_br = np_binary_repr(integers)
    assert np.allclose(br.cpu(), true_br)


def test_aggregate():
    nindices = np.random.randint(1, 10)
    data_in = np.random.randint(2, size=nindices)
    l = np.random.randint(nindices, 20)
    indices = np.random.randint(l, size=nindices)
    a = aggregate(torch.tensor(data_in, device=device), torch.tensor(indices, device=device), l)
    true_a = np_aggregate(data_in, indices, l)
    assert np.allclose(a.cpu(), true_a)


def test_z2rank():
    ### Approximate for now as we're using matrix rank rather than modulo 2 matrix rank due to PyTorch's lack of functionality
    L1, L2 = np.random.randint(1, 100), np.random.randint(1, 100)
    mat = np.random.randint(2, size=(L1, L2))
    rank = z2rank(torch.tensor(mat))
    true_rank = np_z2rank(mat)
    assert true_rank-1 < rank < true_rank+1


def test_stabilizer_projection_trace():
    nqubits, L = np.random.randint(2, 10), np.random.randint(2, 10)
    r = 1
    gs_stb, ps_stb = np.random.randint(2, size=(2*nqubits, 2*nqubits)), np.random.randint(4, size=(nqubits,))
    gs_obs, ps_obs = np.random.randint(2, size=(L, 2*nqubits)), np.random.randint(4, size=(L,))
    gs_stb_copy, ps_stb_copy, gs_obs_copy, ps_obs_copy, r_copy = gs_stb.copy(), ps_stb.copy(), gs_obs.copy(), ps_obs.copy(), r
    out_gs_stb, out_ps_stb, out_r, out_trace = stabilizer_projection_trace(torch.tensor(gs_stb, device=device), torch.tensor(ps_stb, device=device), torch.tensor(gs_obs, device=device), torch.tensor(ps_obs, device=device), r)
    true_out_gs_stb, true_out_ps_stb, true_out_r, true_out_trace = np_stabilizer_projection_trace(gs_stb_copy, ps_stb_copy, gs_obs_copy, ps_obs_copy, r_copy)
    assert np.allclose(out_gs_stb.cpu(), true_out_gs_stb)
    assert np.allclose(out_ps_stb.cpu(), true_out_ps_stb)
    assert np.allclose(out_r, true_out_r)
    assert np.allclose(out_trace, true_out_trace)
