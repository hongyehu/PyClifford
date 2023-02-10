import numpy
import numpy as np
from .utils import mask, condense, pauli_diagonalize1,stabilizer_measure
from .paulialg import Pauli, pauli, PauliMonomial, pauli_zero
from .stabilizer import (StabilizerState, CliffordMap,
    zero_state, identity_map, clifford_rotation_map, random_clifford_map)

class CliffordGate(object):
    '''Represents a Clifford gate.

    Parameters:
    *qubits: int - the qubits that this gate acts on.

    Data:
    generator: Pauli - if the clifford gate is a rotation generated by a single 
        Pauli generator (which is generally not the case), then this records 
        its generator. It is more efficient to implement Clifford rotation than 
        generic Clifford transform.
    forward_map / backward_map: CliffordMap - a generic Clifford gate will be 
        described by the Clifford map, which is a table specifying how each 
        single Pauli operator gets mapped to. (forward and backward maps must 
        be inverse to each other).

    Note: if either the geneator or Clifford maps are specified, the gate will 
        represent the specific unitary transformation; otherwise, the gate 
        is treated as a random Clifford gate that resamples at every call.'''
    def __init__(self, *qubits):
        self.qubits = qubits # the qubits this gate acts on
        self.n = len(self.qubits) # number of qubits it acts on
        self.generator = None
        self.forward_map = None
        self.backward_map = None
        
    def __repr__(self):
        return '[{}]'.format(','.join(str(qubit) for qubit in self.qubits))
    
    def set_generator(self, gen):
        if not isinstance(gen, Pauli):
            raise TypeError("Rotation generator must be a Pauli string")
        self.generator = gen
    def set_forward_map(self,forward_map):
        if not isinstance(forward_map, CliffordMap):
            raise TypeError("Forward map must be a instance of CliffordMap")
        self.forward_map = forward_map
    def set_backward_map(self,backward_map):
        if not isinstance(backward_map, CliffordMap):
            raise TypeError("Backward map must be a instance of CliffordMap")
        self.backward_map = backward_map
        

    def copy(self):
        gate = CliffordGate(*self.qubits)
        if self.generator is not None:
            gate.generator = self.generator.copy()
        if self.forward_map is not None:
            gate.forward_map = self.forward_map.copy()
        if self.backward_map is not None:
            gate.backward_map = self.backward_map.copy()
        return gate
    
    def independent_from(self, other_gate):
        return len(set(self.qubits) & set(other_gate.qubits))==0

    def forward(self, obj):
        if self.generator is not None: # if generator is given, use generator
            if self.n == obj.N: # global gate
                obj.rotate_by(self.generator)
            else: # local gate
                obj.rotate_by(self.generator, mask(self.qubits, obj.N))
        else: # if generator not given, check maps
            if self.forward_map is None:
                if self.backward_map is None: 
                    # if both maps not given, treated as random gate
                    clifford_map = random_clifford_map(self.n)
                else:
                    self.forward_map = self.backward_map.inverse()
                    clifford_map = self.forward_map
            else:
                clifford_map = self.forward_map
            if self.n == obj.N: # global gate
                obj.transform_by(clifford_map)
            else: # local gate
                obj.transform_by(clifford_map, mask(self.qubits, obj.N))
        return obj

    def backward(self, obj):
        if self.generator is not None: # if generator is given, use generator
            if self.n == obj.N: # global gate
                obj.rotate_by(-self.generator)
            else: # local gate
                obj.rotate_by(-self.generator, mask(self.qubits, obj.N))
        else: # if generator not given, check maps
            if self.backward_map is None:
                if self.forward_map is None: 
                    # if both maps not given, treated as random gate
                    clifford_map = random_clifford_map(self.n)
                else:
                    self.backward_map = self.forward_map.inverse()
                    clifford_map = self.backward_map
            else:
                clifford_map = self.backward_map
            if False and self.n == obj.N: # global gate
                obj.transform_by(clifford_map)
            else: # local gate
                obj.transform_by(clifford_map, mask(self.qubits, obj.N))
        return obj

    def compile(self):
        '''construct forward and backward Clifford maps for this gate'''
        if self.generator is not None:
            self.forward_map = clifford_rotation_map(self.generator)
            self.backward_map = clifford_rotation_map(-self.generator)
        else:
            if self.forward_map is None:
                if self.backward_map is None:
                    raise Exception('random Clifford gate can not be compiled.')
                else:
                    self.forward_map = self.backward_map.inverse()
            else:
                if self.backward_map is None:
                    self.backward_map = self.forward_map.inverse()
        return self


class CliffordLayer(object):
    '''Representes a layer of Clifford gates.

    Parameters:
    *gate: CliffordGate - the gates that this layer contains'''
    def __init__(self, *gates):
        self.gates = list(gates) # the gates this layer have
        self.prev_layer = None   # the previous layer
        self.next_layer = None   # the next layer
        self.forward_map = None
        self.backward_map = None
        
    def __repr__(self):
        return '|{}|'.format(''.join(repr(gate) for gate in self.gates))

    def copy(self):
        layer = CliffordLayer(*[gate.copy() for gate in self.gates])
        if self.forward_map is not None:
            layer.forward_map = self.forward_map.copy()
        if self.backward_map is not None:
            layer.backward_map = self.backward_map.copy()
        return layer
    
    def independent_from(self, other_gate):
        return all(gate.independent_from(other_gate) for gate in self.gates)
    
    def take(self, gate):
        if self.prev_layer is None: # if I have no previous layer
            self.gates.append(gate) # I will take the gate
        else: # if I have a previous layer, check it
            if self.prev_layer.independent_from(gate): # if independent (not overlapping)
                self.prev_layer.take(gate) # previous layer take the gate
            else: # if not independent
                self.gates.append(gate) # I will have to keep the gate

    def forward(self, obj):
        if self.forward_map is None:
            for gate in self.gates:
                gate.forward(obj)
        else:
            obj.transform_by(self.forward_map)
        return obj

    def backward(self, obj):
        if self.backward_map is None:
            for gate in self.gates:
                gate.backward(obj)
        else:
            obj.transform_by(self.backward_map)
        return obj

    def compile(self, N):
        '''construct forward and backward Clifford maps for this layer'''
        self.forward_map = identity_map(N)
        self.backward_map = identity_map(N)
        for gate in self.gates:
            gate.compile()
            self.forward_map.embed(gate.forward_map, mask(gate.qubits, N))
            self.backward_map.embed(gate.backward_map, mask(gate.qubits, N))
        return self

class MeasureLayer(object):
    '''
    Computational basis measurement
    '''
    def __init__(self, *qubits,N):
        self.qubits = qubits # qubits that are measured
        self.result = None
        self.log2prob = None
        self.N = N
        self.gs, self.ps = self.obs_gs_ps()
        self.prev_layer = None   # the previous layer
        self.next_layer = None   # the next layer
    def __repr__(self):
        return '|Mz[{}]|'.format(','.join(str(qubit) for qubit in self.qubits))
    def obs_gs_ps(self):
        ps = np.zeros(len(self.qubits)).astype(int)
        gs = np.zeros((len(self.qubits),2*self.N)).astype(int)
        for i in range(len(self.qubits)):
            gs[i,2*self.qubits[i]+1]=1
        return gs, ps
    def forward(self,obj):
        '''
        Measurement will in-place change the stabilizer state. 
        '''
        if not isinstance(obj,StabilizerState):
            raise NotImplementedError("the object is not a stabilizer state")
        tmp_gs_stb, tmp_ps_stb, tmp_r, tmp_out, tmp_log2prob = \
        stabilizer_measure(obj.gs,obj.ps,self.gs,self.ps,obj.r)
        self.result = (-1)**tmp_out
        self.log2prob = tmp_log2prob
        return obj
    def backward(self,obj,measure_result = None):
        '''
        backward is post-selection
        Input:
        - measure_result: list of integers, such as [1,-1,1]
        '''
        if measure_result is not None:
            if len(measure_result)!=len(self.qubits):
                raise ValueError("classical bit does not match number of qubits.")
            else:
                for ii in range(1,len(measure_result)+1):
                    tmp = list(np.zeros(self.N).astype(int))
                    tmp[self.qubits[-ii]]=3
                    tmp_res = int((1-measure_result[-ii])/2)
                    prob = obj.postselect(pauli(tmp), tmp_res)
                    if prob == 0.0:
                        raise ValueError("Post-selection result is not possible, they are orthogonal states.")
                return obj
        else:
            if self.result is None:
                raise ValueError("post-selection result is not given.")
            else:
                for ii in range(1,len(self.result)+1):
                    tmp = list(np.zeros(self.N).astype(int))
                    tmp[self.qubits[-ii]]=3
                    tmp_res = int((1-self.result[-ii])/2)
                    prob = obj.postselect(pauli(tmp), tmp_res)
                    if prob == 0.0:
                        raise ValueError("Post-selection result is not possible, they are orthogonal states.")
                return obj

class CliffordCircuit(object):
    '''Represents a circuit of Clifford gates.

    Examples:
    # create a circuit
    circ = CliffordCircuit()
    # add a gate between qubits 0 and 1
    circ.gate(0,1)
    # or take in a specific gate
    g = pauli('-XX')
    circ.take(clifford_rotation_gate(g))'''
    def __init__(self,N):
        self.N = N # number of qubits in the system
        self.first_layer = CliffordLayer()
        self.last_layer = self.first_layer
        self.forward_map = None
        self.backward_map = None
        
    def __repr__(self):
        layout = '\n'.join(repr(layer) for layer in self.layers_backward())
        return 'CliffordCircuit(\n{})'.format(layout).replace('\n','\n  ')

    def __getattr__(self, item):
        if item == 'N': # if self.N not defined
            # infer from gates (assuming last qubit is covered)
            N = 0
            for layer in self.layers_forward():
                for gate in layer.gates:
                    N = max(N, max(gate.qubits)+1)
            return N
        else:
            return super().__getattribute__(item)

    def copy(self):
        circ = CliffordCircuit(self.N)
        for i, layer in enumerate(self.layers_forward()):
            new_layer = layer.copy()
            if i == 0:
                circ.first_layer = new_layer
                circ.last_layer = new_layer
            else:
                circ.last_layer.next_layer = new_layer
                new_layer.prev_layer = circ.last_layer
                circ.last_layer = new_layer
        if self.forward_map is not None:
            circ.forward_map = self.forward_map.copy()
        if self.backward_map is not None:
            circ.backward_map = self.backward_map.copy()
        return circ

    def layers_backward(self):
        # yield from last to first layers
        layer = self.last_layer
        while layer is not None:
            yield layer
            layer = layer.prev_layer
    
    def layers_forward(self):
        # yield from first to last layers
        layer = self.first_layer
        while layer is not None:
            yield layer
            layer = layer.next_layer

    def take(self, gate):
        if max(gate.qubits)>=self.N:
            raise ValueError("The gate acting on unregistered qubits!")
        if self.last_layer.independent_from(gate): # if last layer commute with the new gate
            self.last_layer.take(gate) # the last layer takes the gate
        else: # otherwise create a new layer to handle this
            new_layer = CliffordLayer(gate) # a new layer with the new gate
            # link to the layer structure
            self.last_layer.next_layer = new_layer
            new_layer.prev_layer = self.last_layer
            self.last_layer = new_layer # new layer becomes the last
        return self
        
    def gate(self, *qubits):
        if max(qubits)>=self.N:
            raise ValueError("The gate acting on unregistered qubits!")
        return self.take(CliffordGate(*qubits)) # create a new gate

    def compose(self, other):
        '''Compose the circuit with another circuit.
            U = U_other U_self

        Parameters:
        other: CliffordCircuit - another circuit to be combined.

        Note: composition will not update the compiled information. Need 
            compilation after circuit composition.'''
        if not other.N == self.N:
            raise ValueError("Qubit number does not match!")
        for layer in other.layers_forward():
            for gate in layer.gates:
                self.take(gate)
        return self

    def forward(self, obj):
        if self.forward_map is None:
            for layer in self.layers_forward():
                layer.forward(obj)
        else:
            obj.transform_by(self.forward_map)
        return obj

    def backward(self, obj):
        if self.backward_map is None:
            for layer in self.layers_backward():
                layer.backward(obj)
        else:
            obj.transform_by(self.backward_map)
        return obj

    def compile(self, N=None):
        '''Construct forward and backward Clifford maps for this circuit
        
        Note: The compilation creates a single Clifford map representing the
            entire circuit, which allows it to run faster for deep circuits.'''
        N = self.N if N is None else N
        self.forward_map = identity_map(N)
        self.backward_map = identity_map(N)
        for layer in self.layers_forward():
            layer.compile(N)
            self.forward_map = self.forward_map.compose(layer.forward_map)
            self.backward_map = self.backward_map.compose(layer.backward_map)
        return self 

    def povm(self, nsample):
        '''Assuming computational basis measurement follows the circuit, this
        will back evolve the computational basis state to generate prior POVM.
        This returns a generator.'''
        for _ in range(nsample):
            zero = zero_state(self.N)
            yield self.backward(zero)

class Circuit(object):
    '''
    Reprsents a Clifford circuit with measurements
    '''
    def __init__(self,N):
        self.N = N # number of qubits in the system
        self.first_layer = CliffordLayer()
        self.last_layer = self.first_layer
        self.measure_result = []
        self.log2prob = 0.0
        self.unitary = True # if True, no measurement layer is inside
        self.forward_map = None
        self.backward_map = None
        self.num_of_measures = 0
    def __repr__(self):
        layout = '\n'.join(repr(layer) for layer in self.layers_backward())
        c =  'CliffordCircuit(\n{})'.format(layout).replace('\n','\n  ')+\
        '\n Unitary:{}'.format(self.unitary)
        return c

    def layers_backward(self):
        # yield from last to first layers
        layer = self.last_layer
        while layer is not None:
            yield layer
            layer = layer.prev_layer
    
    def layers_forward(self):
        # yield from first to last layers
        layer = self.first_layer
        while layer is not None:
            yield layer
            layer = layer.next_layer
            
    def take(self, gate):
        '''
        it can take Clifford gate or measurement layers
        '''
        if (max(gate.qubits)>=self.N):
            raise ValueError("The gate acting on unregistered qubits!")
        if isinstance(gate,CliffordGate):
            if not isinstance(self.last_layer,MeasureLayer):
                if self.last_layer.independent_from(gate): # if last layer commute with the new gate
                    self.last_layer.take(gate) # the last layer takes the gate
                else: # otherwise create a new layer to handle this
                    new_layer = CliffordLayer(gate) # a new layer with the new gate
                    # link to the layer structure
                    self.last_layer.next_layer = new_layer
                    new_layer.prev_layer = self.last_layer
                    self.last_layer = new_layer # new layer becomes the last
            else: # otherwise create a new layer to handle this
                new_layer = CliffordLayer(gate) # a new layer with the new gate
                # link to the layer structure
                self.last_layer.next_layer = new_layer
                new_layer.prev_layer = self.last_layer
                self.last_layer = new_layer # new layer becomes the last
            return self
        elif isinstance(gate,MeasureLayer):
            self.unitary = False
            self.num_of_measures += len(gate.qubits)
            self.last_layer.next_layer = gate
            gate.prev_layer = self.last_layer
            self.last_layer = gate
            return self
        else:
            raise NotImplementedError("Unknow input.")
    def gate(self, *qubits):
        if max(qubits)>=self.N:
            raise ValueError("The gate acting on unregistered qubits!")
        return self.take(CliffordGate(*qubits)) # create a new gate
    def measure(self,*qubits):
        if max(qubits)>=self.N:
            raise ValueError("The gate acting on unregistered qubits!")
        return self.take(MeasureLayer(*qubits,N = self.N)) # create measurement layer
    def forward(self,obj):
        if self.unitary:
            if self.forward_map is None:
                for layer in self.layers_forward():
                    layer.forward(obj)
            else:
                obj.transform_by(self.forward_map)
            return obj
        else:
            for layer in self.layers_forward():
                if not isinstance(layer,MeasureLayer):
                    layer.forward(obj)
                else:
                    layer.forward(obj)
                    self.measure_result += layer.result.tolist()
                    self.log2prob += layer.log2prob
            return obj
    def backward(self,obj,measure_result = None):
        if self.unitary:
            if self.backward_map is None:
                for layer in self.layers_backward():
                    layer.backward(obj)
            else:
                obj.transform_by(self.backward_map)
            return obj
        else:
            if measure_result is not None:
                if len(measure_result)!=self.num_of_measures:
                    raise ValueError("classical bit does not match number of measurements.")
                else:
                    pointer = 0
                    for layer in self.layers_backward():
                        if isinstance(layer,MeasureLayer):
                            new_pointer = pointer - len(layer.qubits)
                            if pointer == 0:
                                layer.backward(obj,measure_result = measure_result[new_pointer:])
                            else:
                                layer.backward(obj,measure_result = measure_result[new_pointer:pointer])
                            pointer = new_pointer
                        else:
                            layer.backward(obj)
                    return obj
            else:
                if len(self.measure_result)>0:
                    pointer = 0
                    for layer in self.layers_backward():
                        if isinstance(layer,MeasureLayer):
                            new_pointer = pointer - len(layer.qubits)
                            if pointer == 0:
                                layer.backward(obj,measure_result = self.measure_result[new_pointer:])
                            else:
                                layer.backward(obj,measure_result = self.measure_result[new_pointer:pointer])
                            pointer = new_pointer
                        else:
                            layer.backward(obj)
                    return obj
                else:
                    raise ValueError("self measurement result is empty, and measurement result is not given!")
        
    def compile(self):
        '''Construct forward and backward Clifford maps for this circuit
        
        Note: The compilation creates a single Clifford map representing the
            entire circuit, which allows it to run faster for deep circuits.'''
        N = self.N
        if self.unitary:
            self.forward_map = identity_map(N)
            self.backward_map = identity_map(N)
            for layer in self.layers_forward():
                layer.compile(N)
                self.forward_map = self.forward_map.compose(layer.forward_map)
                self.backward_map = self.backward_map.compose(layer.backward_map)
            return self 
        else: # has measurement, compile unitary layers
            for layer in self.layers_forward():
                if not isinstance(layer, MeasureLayer):
                    layer.compile(N)
            return self

# ---- gate constructors ----
def clifford_rotation_gate(generator, qubits=None):
    '''Construct a Clifford rotation gate generted by a generator.

    Parameters:
    generator: Pauli - Pauli operator that generates the rotation.
        U = exp( i pi/4 g) = (1 + i g)/sqrt(2)'''
    generator = pauli(generator)
    g_cond, qubits_cond = condense(generator.g) # extract generator support
    if qubits is None:
        qubits = qubits_cond
    else:
        qubits = qubits[qubits_cond]
    gate = CliffordGate(*qubits)
    gate.generator = Pauli(g_cond, generator.p) # condensed generator
    return gate

# ---- circuit constructors ----
def identity_circuit(N = None):
    '''Construct a identity Clifford circuit containing no gate.

    Parameters:
    N: int - number of qubits.'''
    circ = CliffordCircuit(N)
    if N is not None:
        circ.N = N  # fix number of qubits explicitly
    return circ

def brickwall_rcc(N, depth):
    '''Construct random Clifford circuit with brick wall circuit structure.

    Parameters:
    N: int - number of qubits.
    depth: int - circuit depth.'''
    assert(N % 2 == 0) # N should be even
    circ = identity_circuit(N)
    for l in range(depth):
        for i in range(l % 2, N, 2):
            circ.gate(i, (i+1) % N)
    return circ

def onsite_rcc(N):
    '''Construct random Clifford circuit of a layer of single-site gates.
        (useful for implementing random Pauli measurements)

    Parameters:
    N: int - number of qubits.'''
    circ = identity_circuit(N)
    for i in range(N):
        circ.gate(i)
    return circ

def global_rcc(N):
    '''Construct random Clifford circuit of a global Clifford gate.
        (useful for implementing random Clifford measurements)

    Parameters:
    N: int - number of qubits.'''
    circ = identity_circuit(N)
    circ.gate(*range(N))
    return circ

# ---- diagonalization ----
def diagonalize(obj, i0 = 0, causal=False):
    '''Diagonalize a Pauli operator or a stabilizer state (density matrix).

    Parameters:
    obj: Pauli - the operator to be diagonalized, or
         StabilizerState - the state to be diagonalized.
    i0: int - index of the qubit to diagonalize to.
    causal: bool - whether to preserve the causal structure by restricting 
                   the action of Clifford transformation to the qubits at i0 and afterwards.

    Returns:
    circ: CliffordCircuit - circuit that diagonalizes obj.'''
    circ = identity_circuit(obj.N)
    if isinstance(obj, (Pauli, PauliMonomial)):
        if causal:
            for g in pauli_diagonalize1(obj.g[2*i0:]):
                circ.take(clifford_rotation_gate(Pauli(g), numpy.arange(i0,obj.N)))
        else:
            for g in pauli_diagonalize1(obj.g, i0):
                circ.take(clifford_rotation_gate(Pauli(g)))
    elif isinstance(obj, StabilizerState):
        gate = CliffordGate(*numpy.arange(obj.N))
        gate.backward_map = obj.to_map() # set backward map to encoding map
        # then the forward map automatically decodes (= diagonalize) the state
        circ.take(gate)
    else:
        raise NotImplementedError('diagonalization is not implemented for {}.'.format(type(obj).__name__))
    return circ


def SBRG(hmdl, N, max_rate=2., tol=1.e-8):
    '''Approximately diagonalize a Hamiltonian by SBRG.

    Parameters:
    hmdl: PauliPolynomial - model Hamiltonian.

    Returns:
    heff: PauliPolynomial - MBL effective Hamiltonian.
    circ: CliffordCircuit - Clifford circuit to diagonalize the Hamiltonian.'''
    htmp = hmdl.copy() # copy of model Hamiltonian to workspace
    N = htmp.N # system size
    circ = identity_circuit(N) # create circuit
    heff = pauli_zero(N) # create effective Hamiltonian
    # SBRG iteration
    for i0 in range(N): # pivot through every qubit
        if len(htmp) == 0:
            break
        leading = numpy.argmax(numpy.abs(htmp.cs)) # get leading term index
        # find circuit to diagonalize leading term to i0
        circ_i0 = diagonalize(htmp[leading], i0, causal=True) 
        circ.compose(circ_i0) # append it to total circuit
        circ_i0.forward(htmp) # apply it to Hamiltonian
        mask_commute = htmp.gs[:,2*i0] == 0 # mask diagonal terms
        len_anti = sum(~mask_commute) # count number of offdiagonal terms
        if len_anti != 0: # if offdiagonal terms exist
            # split diagonal from offdiagonal terms
            diag = htmp[mask_commute]
            offdiag = htmp[~mask_commute]
            # eleminate offdiagonal terms by perturbation theory
            len_max = int(round(max_rate * len_anti)) # max perturbation terms to keep
            prod = (offdiag @ offdiag).reduce(tol)[:len_max]
            htmp = diag + 0.5 * (htmp[leading].inverse() @ prod)
        # mask terms that has become trivial on the remaining qubits
        mask_trivial = numpy.all(htmp.gs[:,(2*i0+2):] == 0, -1)
        heff += htmp[mask_trivial] # collect trivial terms to effective Hamiltonian
        htmp = htmp[~mask_trivial] # retain with non-trivial terms
    return heff, circ




