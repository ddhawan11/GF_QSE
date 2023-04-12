from hamiltonian import *
from decomposer  import *
from circuits    import *
from mapper      import *

H = Hamiltonian('hamiltonian_acene-1.npy')

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

# ----------------------------------------------------------------------

from qiskit      import *
from qiskit.aqua import QuantumInstance

backend  = Aer.get_backend('qasm_simulator')
instance = QuantumInstance(backend=backend,shots=4)

# JW: [111000-111000] -> P: [101111-010000] -> 2QR: [10111-01000]
# JW: [110100-111000] -> P: [100111-010000] -> 2QR: [10011-01000]
# JW: [111000-110100] -> P: [101111-011000] -> 2QR: [10111-01100]

circuit_dict = {'initial_circuit' : ([[1,1,1,0,0,0,1,1,1,0,0,0]],None),
                'num_steps'       : 1,
                'time_step'       : 0.1,
                'instance'        : instance}

C = QFD_THC_circuits(H,D,M,circuit_dict)
C.run()

