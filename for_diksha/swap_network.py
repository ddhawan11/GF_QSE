from qiskit import *
import itertools,functools
import numpy as np

from qiskit.quantum_info                      import Pauli
from qiskit.aqua.operators                    import WeightedPauliOperator
from qiskit.aqua.components.variational_forms import VariationalForm
from qiskit.aqua                              import QuantumInstance,aqua_globals
from qiskit.providers.aer.noise               import NoiseModel
from qiskit.extensions import SGate

nqubits = 5

qr = QuantumRegister(nqubits,'q')
cr = ClassicalRegister(nqubits,'c')
circuit = QuantumCircuit(qr,cr)

circuit.cx(0,4)
circuit.measure(0,0)

print(circuit.draw())

provider     = IBMQ.load_account()
provider     = IBMQ.get_provider(hub='ibm-q-internal',group='deployed',project='default')
backend      = provider.get_backend('ibmq_athens')

for idx in itertools.permutations([0,1,2,3,4]):
 from qiskit.compiler import transpile
 transpiled_circuit = transpile(circuit,backend,optimization_level=3,
                                initial_layout=list(idx))
 print("with order ",idx)
 print(transpiled_circuit.draw())
 print("OPERATIONS ",transpiled_circuit.count_ops())
 print("DEPTH      ",transpiled_circuit.depth())
 
 exit()


