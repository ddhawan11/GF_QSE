from hamiltonian import *
from decomposer  import *
from circuits    import *
from mapper      import *

H = Hamiltonian('hamiltonian_ethylene-10.npy')

print(H.print_details())

# ----------------------------------------------------------------------

decomposer_dict = {'tolerance'       : 1e-6,
                   'echo_pulses'     : 0,
                   'split_givens'    : False,
                   'post_selection'  : True,
                   'n_layer_circuit' : 1}
D = Decomposer(H,decomposer_dict)
circuit_instruction,measure_instruction = D.generate_unitaries()

print("circuit instruction\n"+circuit_instruction)
print("measure instruction\n"+measure_instruction)

# ----------------------------------------------------------------------

operator_dict = {'qubit_mapping'       : 'jordan_wigner',
                 'two_qubit_reduction' : False}
M = Mapper(operator_dict,H,D)
print("eigenvalues of H")
print(M.diagonalize_H(H))

# ----------------------------------------------------------------------

from qiskit      import *
from qiskit.aqua import QuantumInstance,aqua_globals

seed = 0
aqua_globals.random_seed = seed
backend  = Aer.get_backend('qasm_simulator')
instance = QuantumInstance(backend=backend,shots=8000,seed_transpiler=seed,seed_simulator=seed)

# JW: [10-10] -> P: [11-00] -> 2QR: [1-0]

circuit_dict = {'initial_circuit'         : ([[1,0,1,0]],None),
                'num_steps'               : 2,
                'time_step'               : 10.0,
                'instance'                : instance,
                'dump_data'               : ('results.pkl','read'),
                'post_selection_function' : 'spin'}
post_process = {'ntry' : 10000, 'threshold' : 5e-2, 'seed' : 0}
C = QFD_THC_circuits(H,D,M,circuit_dict)
C.run(post_process)

