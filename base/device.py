import numpy

class QuantumDevice(object):
    def __init__(self, 
                 state,    # base state
                 circuit   # measurement circuit
                ):
        self.state = state
        self.circuit = circuit
        assert self.state.N >= self.circuit.N
        self.N = self.state.N
        
    def __repr__(self):
        return 'QuantumDevice(\n{},\n{})'.format(self.state, self.circuit).replace('\n','\n  ')

    def measure(self, nsample):
        for povm in self.circuit.povm(nsample):
            snapshot = self.state.copy()
            snapshot.measure(povm)
            yield snapshot
