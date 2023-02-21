from qiskit_nature.drivers import UnitsType, Molecule
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper, ParityMapper, BravyiKitaevMapper
from qiskit_nature.transformers.second_quantization.electronic import FreezeCoreTransformer
from qiskit.algorithms import NumPyMinimumEigensolver
from qiskit_nature.algorithms import GroundStateEigensolver
from pyscf import scf, gto, fci
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.drivers.second_quantization import (
    ElectronicStructureDriverType,
    ElectronicStructureMoleculeDriver,
)
from qiskit_nature.mappers.second_quantization import JordanWignerMapper, ParityMapper, BravyiKitaevMapper
import numpy
import sys
sys.path.insert(0, '../')
from pyclifford.paulialg import pauli

def binary_sum(x):
    # Summing in pyclifford seems to take quadratic time. More efficient to do a recursive sum over halves of the list.
    if len(x) < 30:
        return sum(x)
    else:
        return binary_sum(x[::2]) + binary_sum(x[1::2])
    
def qchem_hamiltonian(geometry, use_pyscf=False, multiplicity=1, freeze=True, mapper=ParityMapper()):
    molecule = Molecule(geometry=geometry, charge=0, multiplicity=multiplicity)

    driver = ElectronicStructureMoleculeDriver(
        molecule, basis="sto3g", driver_type=ElectronicStructureDriverType.PYSCF
    )

    es_problem = ElectronicStructureProblem(driver, transformers=[FreezeCoreTransformer()] if freeze else [])
    
    qubit_converter = QubitConverter(mapper, two_qubit_reduction=True, z2symmetry_reduction='auto')
    second_q_op = es_problem.second_q_ops()
    qubit_op = qubit_converter.convert(second_q_op['ElectronicEnergy'], num_particles=es_problem.num_particles).to_pauli_op().oplist
    shift = 0 # shift includes nuclear repulsion, core freezing, and a constant term (removing the identity). 
    nuclear_repuls = es_problem.grouped_property_transformed.get_property("ElectronicEnergy").nuclear_repulsion_energy
    shift += nuclear_repuls
    if freeze:
        core_shift = es_problem.grouped_property_transformed.get_property("ElectronicEnergy")._shift['FreezeCoreTransformer']
        shift += core_shift
    
    print('Done calculating qubit_op.')
    is_identity = lambda x: str(x.primitive).count('I') == len(str(x.primitive))
    const_shift = sum([x.coeff for x in qubit_op if is_identity(x)])
    shift += const_shift
    pyc_hamiltonian = binary_sum([x.coeff * pauli(str(x.primitive)) for x in qubit_op if not is_identity(x)])
    print('Done calculating pyclifford Hamiltonian.')
    if pyc_hamiltonian.N >= 14 or use_pyscf:
        print("Using pyscf ground state estimate")
        mol_s = '; '.join([x + ' ' + ' '.join(map(str, y)) for x,y in geometry])
        mol = gto.M(atom = mol_s, basis = 'sto3g', spin=multiplicity-1)
        rhf = scf.RHF(mol)
        E0 = rhf.kernel()
        return pyc_hamiltonian, numpy.real(E0),shift
    else:
        print("Using exact ground state")
        numpy_solver = NumPyMinimumEigensolver()
        calc = GroundStateEigensolver(qubit_converter, numpy_solver)
        res = calc.solve(es_problem)
        return pyc_hamiltonian, numpy.real(min(res.total_energies) ),shift