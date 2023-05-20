import torch
import numpy as np
from numba import njit


def front(g):
    '''Find the first nontrivial qubit in a Pauli string.
    Parameters:
    g: int (2*N) -  a Pauli string in binary repr.
    Returns:
    i: int - position of its first nontrivial qubit.
    Note:
    If the Pauli string is identity, i = N-1 will be returned, although there
    is no nontrivial qubit.'''
    return torch.div(torch.argmax(g, dim=-1), 2, rounding_mode='floor')


def condense(g):
    '''Condense the Pauli string by taking collecting it in its support, returns
    a shorter string and the support.
    Parameters:
    g: int (2*N) - a Pauli string in binary repr.
    Returns:
    g_cond: int (2*n) - the condensed Pauli string in binary repr.
    qubits: int (n) - indices of supporting qubits.'''
    mask = (g[::2] + g[1::2])
    mask = mask.ge(0.5)
    return torch.masked_select(g, torch.repeat_interleave(mask, 2)), mask.nonzero().flatten()


def acq(g1, g2):
    '''Calculate Pauli operator anticmuunation indicator.
    Parameters:
    g1: int (2*N) or (L1, 2*N) - the first Pauli string in binary repr.
    g2: int (2*N) or (L2, 2*N) - the second Pauli string in binary repr.
    
    Returns:
    acq: int or (L1, L2) array of ints - acq = 0 if g1, g2 commute, acq = 1 if g1, g2 anticommute.'''
    gx1, gx2 = g1[...,::2], g2[...,::2]
    gz1, gz2 = g1[...,1::2], g2[...,1::2]
    return torch.sum(gz1*gx2 - gx1*gz2, dim=-1) % 2


def acq_grid(g1, g2):
    gx1, gx2 = g1[...,::2], g2[...,::2]
    gz1, gz2 = g1[...,1::2], g2[...,1::2]
    return (torch.matmul(gz1, gx2.T) - torch.matmul(gx1, gz2.T)) % 2


@torch.jit.script
def ipow(g1,g2):
    N2 = g1.shape[-1]
    g1x, g1z = g1[...,::2], g1[...,1::2]
    g2x, g2z = g2[...,::2], g2[...,1::2]
    gx = g1x + g2x
    gz = g1z + g2z
    return torch.sum(g1z * g2x - g1x * g2z + 2*(torch.div(gx, 2, rounding_mode='floor') * gz + gx * torch.div(gz, 2, rounding_mode='floor')), dim=-1) % 4


def ipow_product(g1,g2):
    ### we can get rid of this logic if we're willing to go to (1, 2*N) arrays for the case where L1 = 1
    if g1.dim() == 1:
        L1, L2 = 1, 1
    else:
        L1, L2 = g1.shape[0], g2.shape[0]
    g1x, g1z = g1[...,::2].repeat(1, L2).view(L1*L2, -1), g1[...,1::2].repeat(1, L2).view(L1*L2, -1)
    g2x, g2z = g2[...,::2].repeat(L1, 1).view(L1*L2, -1), g2[...,1::2].repeat(L1, 1).view(L1*L2, -1)
    gx = g1x + g2x
    gz = g1z + g2z
    return torch.sum(g1z * g2x - g1x * g2z + 2*(torch.div(gx, 2, rounding_mode='floor') * gz + gx * torch.div(gz, 2, rounding_mode='floor')), axis=-1) % 4


@torch.jit.script
def ps0(gs):
    '''Bare phase factor due to x.z for Pauli strings.
    Parameters:
    gs: int (L,2*N) - array of Pauli strings in binary repr.
    Returns:
    ps0: int (L) - bare phase factor x.z for all strings.'''
    return torch.sum(gs[...,::2] * gs[...,1::2], dim=-1) % 4


def acq_mat(gs):
    '''Construct anticommutation indicator matrix for a set of Pauli strings.
    Parameters:
    gs: int (L,2*N) - array of Pauli strings in binary repr.
    Returns:
    mat: int (L,L) - anticommutation indicator matrix.'''
    gx, gz = gs[...,::2], gs[...,1::2]
    return (torch.matmul(gz, gx.T) - torch.matmul(gx, gz.T)) % 2


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
    gs = ((gs1.unsqueeze(1) + gs2.unsqueeze(0)) % 2).view(-1, gs1.shape[1])
    ps = ((ps1.unsqueeze(1) + ps2.unsqueeze(0)).view(-1,) + ipow_product(gs1, gs2)) % 4
    cs = (cs1.unsqueeze(1)*cs2.unsqueeze(0)).view(-1,)
    return gs, ps, cs


def pauli_tokenize(gs, ps):
    '''Create a token of Pauli operators for learning tasks.
    Parameters:
    gs: int (L, 2*N) - Pauli strings in binary repr.
    ps: int (L) - phase indicators.
    Returns:
    ts: int (L, N+1) - tokens.
       0 = I, 1 = X, 2 = Y, 3 = Z, 4 = +, 5 = -1, 6 = +i, 7 = -i'''
    gx, gz = gs[...,::2], gs[...,1::2]
    ts = 3*gz + (-1)**gz * gx
    x = 4 + torch.div(ps*(11 - 9 * ps + 2 * ps**2),  2, rounding_mode='floor')
    return torch.cat((ts, x.view(-1, 1)), 1)


def pauli_combine(C, gs_in, ps_in):
    '''Combine Pauli operators by operator product.
        (left multiplication)
    Parameters:
    C: int (L_out, L_in) - one-hot encoding of selected operators.
    gs_in: int (L_in, 2*N) - input binary repr of Pauli strings.
    ps_in: int (L_in) - phase indicators of input operators.
    Returns:
    gs_out: int (L_out, 2*N) - output binary repr of Pauli strings.
    ps_out: int (L_out) - phase indicators of output operators.

    Note: gs_out= gs_out.index_add_(0, C_rows, gs_in[C_columns, :]) may be used in future fully vectorized version
    '''
    (L_out, _), device = C.shape, C.device
    gs_out = torch.zeros((L_out, gs_in.shape[-1]), dtype=torch.float32, device=device) # identity
    ps_out = torch.zeros((L_out,), dtype=torch.float32, device=device)
    C_rows, C_columns = torch.nonzero(C, as_tuple=True)
    for i in range(len(C_rows)):
        row_ind, column_ind = C_rows[i], C_columns[i]
        ps_out[row_ind] = (ps_out[row_ind] + ps_in[column_ind] + ipow(gs_out[row_ind], gs_in[column_ind]))%4
        gs_out[row_ind] = (gs_out[row_ind] + gs_in[column_ind])%2
    return gs_out, ps_out


def pauli_transform(gs_in, ps_in, gs_map, ps_map):
    '''Transform Pauli operators by Clifford map.
        (right multiplication)
    Parameters:
    gs_in: int (L, 2*N) - input binary repr of Pauli strings.
    ps_in: int (L) - phase indicators of input operators.
    gs_map: int (2*N, 2*N) - operator map in binary representation.
    ps_map: int (2*N) - phase indicators associated to target operators.
    Returns:
    gs_out: int (L, 2*N) - output binary repr of Pauli strings.
    ps_out: int (L) - phase indicators of output operators.'''
    gs_out, ps_out = pauli_combine(gs_in, gs_map, ps_map)
    ps_out = (ps_in + ps0(gs_in) + ps_out)%4
    return gs_out, ps_out


def clifford_rotate(g, p, gs, ps):
    '''Apply Clifford rotation to Pauli operators.
    Parameters:
    g: int (2*N) -  Clifford rotation generator in binary repr.
    p: int - phase indicator (p = 0, 2 only).
    gs: int (L, 2*N) - input binary repr of Pauli strings.
    ps: int (L)  - phase indicators of input operators.

    Returns: gs, ps NOT in-place modified.'''
    mask = acq(g, gs)
    ps = (ps + (p + 1 + ipow(gs, g.unsqueeze(0))) * mask) % 4
    gs =  (gs + g * mask.view(-1, 1)) % 2
    return gs, ps


def clifford_rotate_signless(g, gs):
    '''Apply Clifford rotation to Pauli operators.
    Parameters:
    g: int (2*N) -  Clifford rotation generator in binary repr.
    gs: int (L, 2*N) - input binary repr of Pauli strings.

    Returns: gs, ps in-place modified.'''
    return (gs + g * acq(g, gs).view(-1, 1)) % 2


def pauli_is_onsite(g, i0=0):
    '''check if a Pauli string is localized on a qubit.
    Parameters:
    g: int (2*N) - Pauli string to check.
    i0: int  - target qubit.
    Returns: True/False'''
    return ~(torch.count_nonzero(g[0:2*i0]) + torch.count_nonzero(g[2*i0+2::])).ge(0.5)


def pauli_diagonalize1(g1, i0=0):
    '''Find a series of Clifford roations to diagonalize a single Pauli string
    to qubit i0 as Z.
    Parameters:
    g1: int (2*N) - Pauli string in binary repr.
    i0: int  - target qubit
    Returns:
    gs: int (L, 2*N) - binary representations of Clifford generators.'''
    gs = [] # prepare to collect Clifford generators
    if not (pauli_is_onsite(g1, i0) and g1[2*i0] == 0): # if g1 is not on site and diagonal
        if g1[2*i0] == 0: # if g1 commute with Z0
            g = g1.clone()
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
        g = g1.clone()
        g[2*i0+1] = (g[2*i0+1] + 1)%2 # g = g1 (*) Z0
        gs.append(g)
        g1 = (g1 + g)%2
        # now g1 has been transformed to Z0
    return gs


def pauli_diagonalize2(g1, g2, i0=0):
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
    gs = [] # prepare to collect Clifford generators
    # bring g1 to Z0
    if not (pauli_is_onsite(g1, i0) and g1[2*i0] == 0): # if g1 is not on site and diagonal
        if g1[2*i0] == 0: # if g1 commute with Z0
            g = g1.clone()
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
        g = g1.clone()
        g[2*i0+1] = (g[2*i0+1] + 1)%2 # g = g1 (*) Z0
        gs.append(g)
        g1 = (g1 + g)%2
        g2 = (g2 + acq(g, g2) * g)%2
        # now g1 has been transformed to Z0
    # bring g2 to X0,Y0
    if not pauli_is_onsite(g2, i0): # if g2 is not on site
        g = g2.clone()
        g[2*i0] = 0
        g[2*i0+1] = 1
        gs.append(g)
        g2 = (g2 + g)%2
        # now g2 has been transformed to X0 or Y0
    return gs, g1, g2


def random_pair(N, L=1, device='cpu'):
    '''Sample an anticommuting pair of random stabilizer and destabilizer.
    Parameters:
    N: int - number of qubits.
    Returns:
    g1: int (2*N) or (L, 2*N) - binary representation of stabilizer.
    g2: int (2*N) or (L, 2*N) - binary representation of destabilizer.
    '''
    g1, g2 = torch.randint(0, 2, (L, 2*N), device=device), torch.randint(0, 2, (L, 2*N), device=device)
    while (g1 == 0).all(): # resample g1 if it is all zero
        g1 = torch.randint(0, 2, (L, 2*N), device=device)
    g1, g2 = impose_leading_noncommutivity(g1, g2)
    return g1.squeeze(0).to(torch.float32), g2.squeeze(0).to(torch.float32)


def impose_leading_noncommutivity(g1, g2):
    mask = (torch.logical_not(acq(g1, g2))*1) # if g1, g2 commute
    i = front(g1) # locate the first nontrivial g1 site
    if g2.dim() > 1:
        mask, i = mask.view(-1, 1), i.view(-1, 1)
    # if the first elements of g1 and g2 commute, flip commutativity by changing g2
    g2 = g2.scatter(-1, 2*i, (torch.gather(g2, -1, 2*i) + torch.gather(g1, -1, 2*i+1)*mask) % 2)
    g2 = g2.scatter(-1, 2*i+1, (torch.gather(g2, -1, 2*i+1) + (torch.gather(g1, -1, 2*i) + torch.gather(g1, -1, 2*i+1))*mask) % 2)
    return g1, g2


def random_pauli(N, device='cpu'):
    '''Sample a random Pauli map.
    Parameters:
    N: int - number of qubits.
    Returs:
    gs: int (2*N, 2*N) - random Pauli map matrix.'''
    g1, g2 = random_pair(1, N, device=device)
    return build_pauli_map(N, g1, g2)


def build_pauli_map(N, g1, g2):
    device=g1.device
    gs = torch.zeros((2*N,2*N), dtype=torch.float32, device=device)
    i = torch.arange(N, device=device).view(-1, 1)
    i = torch.stack((2*i, 2*i), dim=1).view(-1, 1)
    g1 = torch.stack((g1, g2), dim=1).view(-1, 2)
    gs = gs.scatter(-1, i, g1[:, 0:1])
    gs = gs.scatter(-1, i+1, g1[:, 1::])
    return gs


def random_clifford(N, device='cpu'):
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
        g1, g2 = random_pair(n, device=gs.device)
        if n == 1:
            gs[0] = g1
            gs[1] = g2
        else:
            gens, g1, g2 = pauli_diagonalize2(g1, g2)
            gs[0] = g1
            gs[1] = g2
            random_clifford_(gs[2:,2:])
            for g in reversed(gens):
                g = clifford_rotate_signless(g, gs)
        return gs
    return random_clifford_(torch.zeros((2*N,2*N), device=device, dtype=torch.float32))


def map_to_state(gs_in, ps_in):
    #Convert Clifford map to stabilizer state.
    #Parameters:
    #gs_in: int (2*N, 2*N) - Pauli strings in map order.
    #ps_in: int (2*N) - phase indicators in map order.
    #Returns:
    #gs_out: int (2*N, 2*N) - Pauli strings in tableau order.
    #ps_out: int (2*N) - phase indicators in tableau order.
    N = gs_in.shape[-1] // 2
    gs_out = torch.zeros_like(gs_in)
    ps_out = torch.zeros_like(ps_in)
    gs_out[N::] = gs_in[::2]
    gs_out[0:N] = gs_in[1::2]
    ps_out[N::] = ps_in[::2]
    ps_out[0:N] = ps_in[1::2]
    return gs_out, ps_out


def state_to_map(gs_in, ps_in):
    '''Convert stabilizer state to Clifford map.
    Parameters:
    gs_in: int (2*N, 2*N) - Pauli strings in tableau order.
    ps_in: int (2*N) - phase indicators in tableau order.
    Returns:
    gs_out: int (2*N, 2*N) - Pauli strings in map order.
    ps_out: int (2*N) - phase indicators in map order.'''
    N = gs_in.shape[-1] // 2
    gs_out = torch.zeros_like(gs_in)
    ps_out = torch.zeros_like(ps_in)
    gs_out[::2] = gs_in[N::]
    gs_out[1::2] = gs_in[0:N]
    ps_out[::2] = ps_in[N::]
    ps_out[1::2] = ps_in[0:N]
    return gs_out, ps_out


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
    N, indices = Ng//2, torch.arange(Ng, device=gs_stb.device)
    assert 0<=r<=N
    for k in range(L): # loop over incoming projections gs_obs[k]
        acqs = acq(gs_stb, gs_obs[k]).to(torch.bool)
        p = torch.logical_and(acqs, indices<N+r).nonzero()
        if p.shape[0] > 0:
            p = p[0].item()
            acqs[0:p+1] = False
            gs_stb[acqs] = (gs_stb[acqs] + gs_stb[p]) % 2
            q = (p+N)%(2*N) # get q as dual of p
            gs_stb[q] = gs_stb[p] # move gs_stb[p] to gs_stb[q]
            gs_stb[p] = gs_obs[k] # add gs_obs[k] to gs_stb[p]
            if not (r <= p < N):
                r -= 1 # rank will reduce under extension
                if p == r: # bring new stabilizer from p to r
                    pass
                elif q == r:
                    gs_stb[[p,q]] = gs_stb[[q,p]] # swap p,q
                else:
                    s = (r+N)%(2*N) # get s as dual of r
                    gs_stb[[p,r]] = gs_stb[[r,p]] # swap p,r
                    gs_stb[[q,s]] = gs_stb[[s,q]] # swap q,s
    return gs_stb, r


@torch.jit.script
def stabilizer_measure(gs_stb, ps_stb, gs_obs, ps_obs, r, samples=torch.tensor(0)):
    '''Measure Pauli operators on a stabilizer state.
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
    (L, Ng), device = gs_obs.shape, gs_stb.device
    N, log2prob, out, indices, k = Ng//2, 0, torch.zeros(L, dtype=torch.float32, device=device), torch.arange(Ng, device=device), 0
    ps_stb = ps_stb[0:N]
    if torch.allclose(samples, torch.tensor(0)):
        samples = torch.randint(0, 2, (L,), device=device)
    assert 0<=r<=N
    for k in range(L): # for each observable gs_obs[k]
        update = torch.zeros(Ng, dtype=torch.bool, device=device)
        acqs = acq(gs_stb, gs_obs[k]).to(torch.bool)
        p_temp = torch.logical_and(acqs, indices<N+r).nonzero()
        if p_temp.shape[0] > 0:
            p = torch.tensor(p_temp[0].item(), dtype=torch.long)
            update[p+1::] = True
            p2 = torch.logical_and(acqs, update).nonzero().flatten()
            p1 = torch.logical_and(torch.logical_and(acqs, indices<N), update).nonzero().flatten()
            ps_stb = ps_stb.scatter(-1, p1, (ps_stb[p1] + ps_stb[p] + ipow(gs_stb[p1], gs_stb[p]))%4)
            gs_stb[p2] = (gs_stb[p2] + gs_stb[p])%2
        else:
            p = p_temp.to(torch.long)
        temp_acqs = torch.logical_and(torch.logical_and(acqs,  ~update), ~(indices<N+r))
        temp_acqs = torch.roll(temp_acqs, shifts=(-N), dims=(0))
        ga = torch.cumsum(temp_acqs.unsqueeze(-1)*torch.cat((ps_stb, ps_stb)).unsqueeze(0), dim=-1) % 2
        pa = torch.sum(torch.cat((ps_stb, ps_stb))*temp_acqs + ipow(ga, ga), dim=0) % 4
        if torch.any(update):
            q = (p+N)%(2*N)
            gs_stb[q] = gs_stb[p]
            gs_stb[p] = gs_obs[k]
            if ((p < N + r) and not (r <= p < N)):
                r -= 1
                if p == r:
                    pass
                elif q == r:
                    gs_stb[[p,q]] = gs_stb[[q,p]] # swap p,q
                else:
                    s = (r+N)%(2*N) # get s as dual of r
                    gs_stb[[p,r]] = gs_stb[[r,p]] # swap p,r
                    gs_stb[[q,s]] = gs_stb[[s,q]] # swap q,s
                p = r
            ps_stb[p] = 2 * samples[k]
            out[k] = torch.div((ps_stb[p] - ps_obs[k])%4, 2, rounding_mode='trunc') #0->0(+1 eigenvalue), 2->1(-1 eigenvalue)
            log2prob -= 1
        else:
            out[k] = torch.div((pa - ps_obs[k])%4, 2, rounding_mode='trunc')
    return gs_stb, ps_stb, r, out, log2prob


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
    device = gs_obs.device
    assert 0<=r<=N
    xs = torch.zeros(L, dtype=torch.float32, device=device) # expectation values
    ga = torch.zeros(2*N, dtype=torch.float32, device=device) # workspace for stabilizer accumulation
    for k in range(L): # for each observable gs_obs[k]
        ga[:] = 0
        pa, trivial = 0, True
        for j in range(Ng):
            if acq(gs_stb[j], gs_obs[k]):
                if j < N + r: # if gs_stb[j] is active stablizer or standby.
                    xs[k] = 0 # gs_obs[k] is logical or error operator.
                    trivial = False # gs_obs[k] is not trivial
                    break
                else: # accumulate stablizer components
                    pa = (pa + ps_stb[j-N] + ipow(ga, gs_stb[j-N]))%4
                    ga = (ga + gs_stb[j-N])%2
        if trivial:
            xs[k] = (-1)**torch.div(((pa - ps_obs[k])%4), 2, rounding_mode='floor')
    return xs


def z2rank(mat):
    '''Calculate Z2 rank of a binary matrix.
    Parameters:
    mat: int matrix - input binary matrix.
        caller must ensure that mat contains only 0 and 1.
        mat is destroyed upon output!
    Returns:
    r: int - rank of the matrix under Z2 algebra.'''
    return torch.linalg.matrix_rank(mat.to(torch.float32))


@torch.jit.script
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
    mask2 = mask.repeat_interleave(2)
    inside  = torch.sum(gs[:,  mask2], -1) != 0
    outside = torch.sum(gs[:, ~mask2], -1) != 0
    across = torch.logical_and(inside, outside)
    gs_across_sub = gs[across][:, mask2]
    if L == N: # state is pure
        entropy = torch.div(z2rank(acq_mat(gs_across_sub)), 2, rounding_mode='floor')
    else:
        strict = torch.sum(inside) - torch.sum(across)
        hidden = z2rank(gs_across_sub) - z2rank(acq_mat(gs_across_sub))
        entropy = torch.sum(mask) - strict - hidden
    return entropy


def mask(qubits, N, device='cpu'):
    '''Create a mask vector for a subsystem of qubits.
    Parameters:
    qubits: int (n) - a subsystem specified by qubit indices.
    N: int - total system size.
    Returns:
    mask: bool (N) -  a boolean vector with True at specified qubits.
    Note: complement region is ~mask.'''
    mask = torch.zeros(N, dtype=torch.bool, device=device)
    mask[torch.tensor(qubits)] = True
    return mask


def binary_repr(ints, width=None):
    '''Convert an array of integers to their binary representations.

    Parameters:
    ints: int array - array of integers.
    width: width of the binary representation (default: determined by the bit length of the maximum int).

    Returns:
    new array where each integter is unpacked to binary subarray.
    '''
    device = ints.device
    ints = ints.cpu().numpy()
    width = np.ceil(np.log2(np.max(ints)+1)).astype(int) if width is None else width
    dt0 = ints.dtype
    dt1 = np.dtype((dt0, [('bytes','u1', dt0.itemsize)]))
    bins = np.unpackbits(ints.view(dtype=dt1)['bytes'], axis=-1, bitorder='little')
    return torch.tensor(np.flip(bins, axis=-1)[...,-width:].copy(), device=device)


def aggregate(data_in, inds, l):
    '''Aggregate data (1d array) by unique inversion indices.
    Parameter:
    data_in: any (L) - input data array.
    inds: int (L) - indices that each element should be mapped to.
    l : int - number of unique elements in data_in.
    Returns:
    data_out: any (l) - output data array.'''
    return torch.zeros(l, dtype=data_in.dtype, device=data_in.device).index_add_(-1, inds, data_in)


@njit
def z2inv(mat):
    '''
    Calculate Z2 inversion of a binary matrix.
    Left as Numpy version for now.
    '''
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


def stabilizer_projection_trace(gs_stb, ps_stb, gs_obs, ps_obs, r):
    '''Measure Pauli operators on a stabilizer state.
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
    trace: float - Tr(P * rho1 * P*)'''
    (L, Ng), device = gs_obs.shape, gs_stb.device
    N, trace, out, indices = Ng//2, 1, torch.zeros(L, dtype=torch.float32, device=device), torch.arange(Ng, device=device)
    ps_stb = ps_stb[0:N]
    assert 0<=r<=N
    for k in range(L): # for each observable gs_obs[k]
        update = torch.zeros(Ng, dtype=torch.bool, device=device)
        acqs = acq(gs_stb, gs_obs[k]).to(torch.bool)
        p = torch.logical_and(acqs, indices<N+r).nonzero()
        if p.shape[0] > 0:
            p = p[0].item()
            update[p+1::] = True
            p2 = torch.logical_and(acqs, update).nonzero().flatten()
            p1 = torch.logical_and(torch.logical_and(acqs, indices<N), update).nonzero().flatten()
            ps_stb = ps_stb.scatter(-1, p1, (ps_stb[p1] + ps_stb[p] + ipow(gs_stb[p1], gs_stb[p]))%4)
            gs_stb[p2] = (gs_stb[p2] + gs_stb[p])%2
        temp_acqs = torch.logical_and(torch.logical_and(acqs,  ~update), ~(indices<N+r))
        temp_acqs = torch.roll(temp_acqs, shifts=(-N), dims=(0))
        ga = torch.cumsum(temp_acqs.unsqueeze(-1)*torch.cat((ps_stb, ps_stb)).unsqueeze(0), dim=-1) % 2
        pa = torch.sum(torch.cat((ps_stb, ps_stb))*temp_acqs + ipow(ga, ga), dim=0) % 4
        if torch.any(update):
            q = (p+N)%(2*N)
            gs_stb[q] = gs_stb[p]
            gs_stb[p] = gs_obs[k]
            if ((p < N + r) and not (r <= p < N)):
                r -= 1
                if p == r:
                    pass
                elif q == r:
                    gs_stb[[p,q]] = gs_stb[[q,p]] # swap p,q
                else:
                    s = (r+N)%(2*N) # get s as dual of r
                    gs_stb[[p,r]] = gs_stb[[r,p]] # swap p,r
                    gs_stb[[q,s]] = gs_stb[[s,q]] # swap q,s
                p = r
            # the projection will change phase of stabilizer
            ps_stb[p] = ps_obs[k]
            trace = trace/2.
        else: # no update, gs_obs[k] is eigen, result is in pa
            if not pa == ps_obs[k]:
                trace = 0.
    return gs_stb, ps_stb, r, trace
