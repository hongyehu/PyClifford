# Define the Pauli matrices
Pauli_I = np.array([[1, 0], [0, 1]])
Pauli_X = np.array([[0, 1], [1, 0]])
Pauli_Y = np.array([[0, -1j], [1j, 0]])
Pauli_Z = np.array([[1, 0], [0, -1]])

# Define the single-qubit Clifford gates
Clifford_gates = {
    'I': Pauli_I,
    'X': Pauli_X,
    'Y': Pauli_Y,
    'Z': Pauli_Z,
    'H': 1/np.sqrt(2) * (Pauli_X + Pauli_Z),
    'S': Pauli_Z,
    'T': np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])
}


class GeneralizedStabilizerState:
    """
    Class representing a generalized stabilizer state in a quantum system.

    Attributes:
        num_qubits (int): Number of qubits in the state.
        stabilizers (list): List of stabilizer generators (strings of Pauli operators).
        phase (list): List of phase indicators (0 or 1).

    Methods:
        __init__(self, num_qubits):
            Initializes a GeneralizedStabilizerState object with given number of qubits.

        apply_clifford_map(self, clifford_map):
            Applies a Clifford map to the stabilizer state.

        apply_gate(self, gate, *args):
            Applies a single-qubit or two-qubit gate to the stabilizer state.

        measure(self, qubit_index):
            Measures a qubit and updates the stabilizers and phases accordingly.

        __str__(self):
            Returns a string representation of the GeneralizedStabilizerState object.
    """

    def __init__(self, num_qubits):
        """
        Initialize a GeneralizedStabilizerState object.

        Args:
            num_qubits (int): Number of qubits in the state.
        """
        self.num_qubits = num_qubits
        self.stabilizers = ['I' * num_qubits]  # Initialize with the identity stabilizer
        self.phase = [0] * num_qubits  # Initialize with all phases set to 0

    def apply_clifford_map(self, clifford_map):
        """
        Apply a Clifford map to the stabilizer state.

        Args:
            clifford_map (dict): Dictionary specifying the Clifford operations to apply.

        Returns:
            GeneralizedStabilizerState: Updated state after applying the Clifford map.
        """
        new_stabilizers = []
        new_phase = self.phase[:]  # Copy current phase

        for g, p in zip(self.stabilizers, self.phase):
            new_g, _ = self.apply_single_clifford(g, p, clifford_map)
            new_stabilizers.append(new_g)

        return GeneralizedStabilizerState(self.num_qubits)._from_stabilizers(new_stabilizers, new_phase)

    def apply_single_clifford(self, g, p, clifford_map):
        """
        Apply a single Clifford operation to a stabilizer generator.

        Args:
            g (str): Stabilizer generator (string of Pauli operators).
            p (int): Phase indicator (0 or 1).
            clifford_map (dict): Dictionary specifying the Clifford operations to apply.

        Returns:
            str: Updated stabilizer generator after applying the Clifford operation.
            int: Updated phase after applying the Clifford operation.
        """
        new_g = []
        for char in g:
            if char == 'I':
                new_g.append('I')
            elif len(char) >= 2:
                qubit_index = int(char[1])
                if char[0] in clifford_map:
                    new_g.append(clifford_map[char[0]][qubit_index])
                else:
                    new_g.append('I')  # If the character doesn't match known patterns, default to 'I'
            else:
                new_g.append('I')  # If the character doesn't match known patterns, default to 'I'

        return ''.join(new_g), p

    def apply_gate(self, gate, *args):
        """
        Apply a gate (single-qubit or two-qubit) to the stabilizer state.

        Args:
            gate (str): Gate to apply ('H', 'X', 'Y', 'Z', 'S', 'T', 'CNOT').
            *args: Qubit index/indices for single-qubit/two-qubit gates.

        Raises:
            ValueError: If an unsupported gate is provided.
        """
        if gate in Clifford_gates:
            gate_matrix = Clifford_gates[gate]

            if len(args) == 1:  # Single-qubit gate
                qubit_index = args[0]
                for i in range(len(self.stabilizers)):
                    if self.stabilizers[i][qubit_index] == 'X':
                        self.stabilizers[i] = self.stabilizers[i][:qubit_index] + gate + self.stabilizers[i][qubit_index + 1:]
                    elif self.stabilizers[i][qubit_index] == 'Y':
                        self.stabilizers[i] = self.stabilizers[i][:qubit_index] + gate + self.stabilizers[i][qubit_index + 1:]
                        self.phase[i] = (self.phase[i] + 1) % 2
                    elif self.stabilizers[i][qubit_index] == 'Z':
                        self.phase[i] = (self.phase[i] + 1) % 2

            elif len(args) == 2 and gate == 'CNOT':  # Two-qubit CNOT gate
                control_qubit = args[0]
                target_qubit = args[1]
                for i in range(len(self.stabilizers)):
                    if self.stabilizers[i][control_qubit] == 'X':
                        self.stabilizers[i] = self.stabilizers[i][:control_qubit] + 'X' + self.stabilizers[i][control_qubit + 1:]
                        self.stabilizers[i] = self.stabilizers[i][:target_qubit] + 'X' + self.stabilizers[i][target_qubit + 1:]
                    elif self.stabilizers[i][control_qubit] == 'Y':
                        self.stabilizers[i] = self.stabilizers[i][:control_qubit] + 'Y' + self.stabilizers[i][control_qubit + 1:]
                        self.stabilizers[i] = self.stabilizers[i][:target_qubit] + 'Y' + self.stabilizers[i][target_qubit + 1:]
                        self.phase[i] = (self.phase[i] + 1) % 2
                    elif self.stabilizers[i][control_qubit] == 'Z':
                        self.stabilizers[i] = self.stabilizers[i][:target_qubit] + 'Z' + self.stabilizers[i][target_qubit + 1:]

            else:
                raise ValueError(f"Unsupported gate: {gate}")

        else:
            raise ValueError(f"Unsupported gate: {gate}")

    def measure(self, qubit_index):
        """
        Measure a qubit and update the stabilizers and phase accordingly.

        Args:
            qubit_index (int): Index of the qubit to measure.
        """
        for i in range(len(self.stabilizers)):
            if self.stabilizers[i][qubit_index] == 'X' or self.stabilizers[i][qubit_index] == 'Y':
                self.phase[i] = (self.phase[i] + 1) % 2
            self.stabilizers[i] = self.stabilizers[i][:qubit_index] + 'I' + self.stabilizers[i][qubit_index + 1:]

    def _from_stabilizers(self, stabilizers, phase):
        """
        Internal method to create a new GeneralizedStabilizerState from stabilizers and phase.

        Args:
            stabilizers (list): List of stabilizer generators.
            phase (list): List of phase indicators.

        Returns:
            GeneralizedStabilizerState: New instance of GeneralizedStabilizerState.
        """
        state = GeneralizedStabilizerState(self.num_qubits)
        state.stabilizers = stabilizers
        state.phase = phase
        return state

    def __str__(self):
        """
        Return a string representation of the GeneralizedStabilizerState object.

        Returns:
            str: String representation of the state.
        """
        output = []
        for g in self.stabilizers:
            output.append(f" +{g}")
        return "\n".join(output)


if __name__ == "__main__":
    # Example usage
    num_qubits = 3

    # Create an initial state
    gs_state = GeneralizedStabilizerState(num_qubits)
    gs_state.stabilizers = ['XIZ', 'ZXX', 'YIZ']
    gs_state.phase = [0, 1, 0]

    print("Initial state:")
    print(gs_state)

     # Apply a Clifford map 
    clifford_map = {
        'X0': '+IIXI', 'Y0': '+IIYI', 'Z0': '+IIZI',
        'X1': '+IXII', 'Y1': '+IYII', 'Z1': '+IIZI',
        'X2': '+IXII', 'Y2': '+IYII', 'Z2': '+IIZI'
    }

    gs_state.apply_clifford_map(clifford_map)
    print("\nState after applying Clifford map:")
    print(gs_state)

    # Apply non-Clifford gates
    gs_state.apply_gate('H', 0)
    gs_state.apply_gate('CNOT', 1, 2)
    print("\nState after applying non-Clifford gates:")
    print(gs_state)

    # Measure qubit 0
    gs_state.measure(0)
    print("\nState after measuring qubit 0:")
    print(gs_state)
