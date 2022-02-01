import qutip as qt
import numpy as np

def pauli2pauli(g,p):
    '''
    g: int (2*N) - an array of pauli string
    p: int - phase indicator
    '''
    tmp = [qt.qeye(2), qt.sigmax(), qt.sigmay(), qt.sigmaz()]
    N = np.shape(g)[0]//2
    tmp_list=[]
    for i in range(N):
        if (g[2*i]==1)&(g[2*i+1]==1):
            tmp_list.append(tmp[2])
        elif (g[2*i]==1)&(g[2*i+1]==0):
            tmp_list.append(tmp[1])
        elif (g[2*i]==0)&(g[2*i+1]==1):
            tmp_list.append(tmp[3])
        else:
            tmp_list.append(tmp[0])
    return (1j)**(p)*qt.tensor(tmp_list)
def state2state(state):
    N = state.ps.shape[0]//2
    r = state.r
    ID = qt.tensor([qt.qeye(2) for i in range(N)])
    rho = ID
    for i in range(r,N):
        rho = rho*(ID+pauli2pauli(state.gs[i],state.ps[i]))/2
    return rho
def state2projector(state):
    proj_list = []
    N = state.ps.shape[0]//2
    r = state.r
    ID = qt.tensor([qt.qeye(2) for i in range(N)])
    rho = ID
    for i in range(r,N):
        tmp = (ID+pauli2pauli(state.gs[i],state.ps[i]))/2
        proj_list.append(tmp)
    return proj_list
def state2paulis(state):
    pauli_list = []
    N = state.ps.shape[0]//2
    r = state.r
    for i in range(r,N):
        tmp = pauli2pauli(state.gs[i],state.ps[i])
        pauli_list.append(tmp)
    return pauli_list
def embed(reduce_rho,N,position):
    '''
    reduce_rho: is the reduced density matrix
    N: is the total qubits number
    position: is the kept qubits
    '''
    identity_qubits = {i for i in range(N)}.difference(set(position))
    current_idx_order = list(position)+list(identity_qubits)
    transpose_order = np.argsort(np.array(current_idx_order)).tolist()
    return qt.tensor([reduce_rho]+[qt.qeye(2)/2 for i in range(int(N-len(position)))]).permute(transpose_order)
#### I/O 
def save_StabilizerState(filename, list_of_state):
    '''
    list_of_state: list of Stabilizer State
    '''
    num = len(list_of_state)
    doubleN = list_of_state[0].gs.shape[0]
    gs_array = np.zeros((num,doubleN,doubleN))
    ps_array = np.zeros((num,doubleN))
    r_array = np.zeros(num)
    for i in range(num):
        gs_array[i,:,:] = list_of_state[i].gs
        ps_array[i,:] = list_of_state[i].ps
        r_array[i] = list_of_state[i].r
    with h5py.File(filename,"w") as F:
        F.create_dataset("Number",data = np.array([num]))
        F.create_dataset("gs_array",data = gs_array)
        F.create_dataset("ps_array",data = ps_array)
        F.create_dataset("r_array",data = r_array)
def load_StabilizerState(filename):
    with h5py.File(filename,"r") as F:
        num = F["Number"][0]
        gs_array = F["gs_array"][:]
        ps_array = F["ps_array"][:]
        r_array = F["r_array"][:]
    list_state = []
    for i in range(num):
        tmp = stabilizer.StabilizerState(gs_array[i,:,:],ps_array[i,:])
        tmp.set_r(int(r_array[i]))
        list_state.append(tmp)
    return list_state

#### other utilities
def sample_from_mean(mean):
    '''
    return: +/- 1 with mean value given
    '''
    p = (mean+1)/2.
    return 2*np.random.binomial(1,p)-1
def generate_locMixer(px,py,pz):
    p = px+py+pz
    I = np.array([[1, 0],[0, 1]]);
    X = np.array([[0, 1],[1, 0]]);    s1 = X;
    Z = np.array([[1, 0],[0, -1]]);   s3 = Z;
    Y = np.array([[0, -1j],[1j, 0]]); s2 = Y;

    USA=np.zeros((2,2,4,4));

    E00 = np.zeros((4,4));
    E10 = np.zeros((4,4));
    E20 = np.zeros((4,4));
    E30 = np.zeros((4,4));
    E00[0,0] = 1;
    E10[1,0] = 1;
    E20[2,0] = 1;
    E30[3,0] = 1;


    USA = USA + np.sqrt(1.0-p)*ncon((I,E00),([-1,-2],[-3,-4]))
    USA = USA + np.sqrt(px)*ncon((s1,E10),([-1,-2],[-3,-4]))
    USA = USA + np.sqrt(py)*ncon((s2,E20),([-1,-2],[-3,-4]))
    USA = USA + np.sqrt(pz)*ncon((s3,E30),([-1,-2],[-3,-4]))

    E0=np.zeros((4));
    E0[0] = 1;
    locMixer = ncon( ( USA,E0, np.conj(USA), E0 ),([-1,-2,1,3],[3],[-4,-3,1,2],[2]));
    return locMixer
### on going ###
def depolarize_channel(N, rho0,px,py,pz):
    rho_tensor = rho0.full().reshape([2 for i in range(int(2*N))])
    locMixer = generate_locMixer(px,py,pz)
    index = []
    index.append([i for i in range(1,int(2*N+1))])
    for i in range(N):
        index.append([i+1,-(i+1),-(i+1+N),i+1+N])
    eps_rho = ncon([rho_tensor]+[locMixer for i in range(N)],index)
    eps_rho=eps_rho.reshape(2**N,2**N)
    rho = qt.Qobj(eps_rho,dims=[[2 for i in range(N)],[2 for i in range(N)]])
    return rho

def DensityMatrix_Measure(rho_ori,ops,N):
    rho = rho_ori.copy()
    gs = ops.gs.copy()
    ps = ops.ps.copy()
    ID = qt.tensor([qt.qeye(2) for i in range(N)])
    pauli_list = state2paulis(ops)
    for i in range(len(pauli_list)):
        tmp_mean = np.trace(rho*pauli_list[i])
        readout = sample_from_mean(tmp_mean.real)#+/- 1
        readout_p = 1-readout
        ps[i] = (ps[i]+readout_p)%4
        proj = (ID+readout*pauli_list[i])/2
        rho = (proj*rho*proj).unit()
    return stabilizer.StabilizerState(gs,ps)










