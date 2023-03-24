import numpy as np
import qutip as qt
import unittest

from ..stabilizer import *
from ..paulialg import pauli, paulis
from ..circuit import (CliffordGate, CliffordLayer, CliffordCircuit, MeasureLayer, Circuit,
                       CNOT)

import unittest

class TestCircuitStim(unittest.TestCase):
    
    def test_ghz(self):
        # PyClifford construction
        # psi = ghz_state(3)
        # gate = CliffordGate(1, 2)
        # gate = CNOT(1, 2)
        # # qutip construction
        # psi_qt = qt.ghz_state(3)
        # gate_qt = qt.gates.cnot(3, 1, 2)
        circuit = CliffordCircuit(3)
        print(circuit)
        

if __name__ == '__main__':
    unittest.main()