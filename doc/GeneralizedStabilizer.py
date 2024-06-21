class GraphState:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.stabilizers = [self._initial_stabilizer(i) for i in range(num_qubits)]
        self.destabilizers = [self._initial_destabilizer(i) for i in range(num_qubits)]
        self.active_stabilizers = {i: True for i in range(num_qubits)}
        self.active_destabilizers = {i: True for i in range(num_qubits)}

    def _initial_stabilizer(self, i):
        """Create initial stabilizer for qubit i: Z_i (all Zs except for an X at position i)."""
        return ['I'] * i + ['Z'] + ['I'] * (self.num_qubits - i - 1)

    def _initial_destabilizer(self, i):
        """Create initial destabilizer for qubit i: X_i (all Is except for an X at position i)."""
        return ['I'] * i + ['X'] + ['I'] * (self.num_qubits - i - 1)

    def apply_gate(self, gate, *args):
        if gate == 'H':
            qubit_index = args[0]
            self._apply_hadamard(qubit_index)
        elif gate == 'S':
            qubit_index = args[0]
            self._apply_phase(qubit_index)
        elif gate == 'CZ':
            qubit_index1, qubit_index2 = args
            self._apply_controlled_z(qubit_index1, qubit_index2)
        elif gate == 'CNOT':
            control, target = args
            self._apply_cnot(control, target)
        else:
            raise ValueError(f"Unsupported gate: {gate}")

    def _apply_hadamard(self, qubit_index):
        for stabilizer in self.stabilizers:
            if stabilizer[qubit_index] == 'X':
                stabilizer[qubit_index] = 'Z'
            elif stabilizer[qubit_index] == 'Z':
                stabilizer[qubit_index] = 'X'
        for destabilizer in self.destabilizers:
            if destabilizer[qubit_index] == 'X':
                destabilizer[qubit_index] = 'Z'
            elif destabilizer[qubit_index] == 'Z':
                destabilizer[qubit_index] = 'X'

    def _apply_phase(self, qubit_index):
        for stabilizer in self.stabilizers:
            if stabilizer[qubit_index] == 'X':
                stabilizer[qubit_index] = 'Y'
            elif stabilizer[qubit_index] == 'Y':
                stabilizer[qubit_index] = 'X'
        for destabilizer in self.destabilizers:
            if destabilizer[qubit_index] == 'X':
                destabilizer[qubit_index] = 'Y'
            elif destabilizer[qubit_index] == 'Y':
                destabilizer[qubit_index] = 'X'

    def _apply_controlled_z(self, qubit_index1, qubit_index2):
        for stabilizer in self.stabilizers:
            if stabilizer[qubit_index1] == 'X' and stabilizer[qubit_index2] == 'X':
                stabilizer[qubit_index2] = 'Z'
            elif stabilizer[qubit_index1] == 'Z' and stabilizer[qubit_index2] == 'Z':
                stabilizer[qubit_index1] = 'X'
        for destabilizer in self.destabilizers:
            if destabilizer[qubit_index1] == 'X' and destabilizer[qubit_index2] == 'X':
                destabilizer[qubit_index2] = 'Z'
            elif destabilizer[qubit_index1] == 'Z' and destabilizer[qubit_index2] == 'Z':
                destabilizer[qubit_index1] = 'X'

    def _apply_cnot(self, control, target):
        for stabilizer in self.stabilizers:
            if stabilizer[control] == 'X':
                if stabilizer[target] == 'I':
                    stabilizer[target] = 'X'
                elif stabilizer[target] == 'X':
                    stabilizer[target] = 'I'
                elif stabilizer[target] == 'Z':
                    stabilizer[target] = 'Y'
                elif stabilizer[target] == 'Y':
                    stabilizer[target] = 'Z'
            elif stabilizer[control] == 'Z':
                if stabilizer[target] == 'I':
                    stabilizer[target] = 'Z'
                elif stabilizer[target] == 'Z':
                    stabilizer[target] = 'I'
                elif stabilizer[target] == 'X':
                    stabilizer[target] = 'Y'
                elif stabilizer[target] == 'Y':
                    stabilizer[target] = 'X'
        for destabilizer in self.destabilizers:
            if destabilizer[control] == 'X':
                if destabilizer[target] == 'I':
                    destabilizer[target] = 'X'
                elif destabilizer[target] == 'X':
                    destabilizer[target] = 'I'
                elif destabilizer[target] == 'Z':
                    destabilizer[target] = 'Y'
                elif destabilizer[target] == 'Y':
                    destabilizer[target] = 'Z'
            elif destabilizer[control] == 'Z':
                if destabilizer[target] == 'I':
                    destabilizer[target] = 'Z'
                elif destabilizer[target] == 'Z':
                    destabilizer[target] = 'I'
                elif destabilizer[target] == 'X':
                    destabilizer[target] = 'Y'
                elif destabilizer[target] == 'Y':
                    destabilizer[target] = 'X'

    def measure(self, qubit_index):
        stabilizer = self.stabilizers[qubit_index]
        return '0' if stabilizer[qubit_index] == 'Z' else '1'

    def __str__(self):
        stabs = '\n'.join([' '.join(stabilizer) for stabilizer in self.stabilizers])
        destabs = '\n'.join([' '.join(destabilizer) for destabilizer in self.destabilizers])
        return f'Stabilizers:\n{stabs}\n\nDestabilizers:\n{destabs}'

# Usage
gs_state = GraphState(3)
print("Initial stabilizers and destabilizers:")
print(gs_state)

# Apply Clifford gates
gs_state.apply_gate('H', 0)
gs_state.apply_gate('S', 1)
gs_state.apply_gate('CZ', 1, 2)
print("\nState after applying Clifford gates:")
print(gs_state)

# Apply non-Clifford gates
gs_state.apply_gate('H', 0)
gs_state.apply_gate('CNOT', 1, 2)
print("\nState after applying non-Clifford gates:")
print(gs_state)
