from qiskit      import *
from qiskit.aqua import QuantumInstance
from qiskit.providers.aer.backends.aerbackend import AerBackend
from qiskit.providers.ibmq.ibmqbackend import IBMQBackend
from qiskit.providers.aer import noise
from qiskit.providers.aer.noise import NoiseModel
from qiskit.ignis.mitigation.measurement import (complete_meas_cal,
                                                 CompleteMeasFitter)

from subroutines import *

nqubits = 3
Sx = single_qubit_pauli('x',0,nqubits) + single_qubit_pauli('x',1,nqubits) + single_qubit_pauli('x',2,nqubits)
Sy = single_qubit_pauli('y',0,nqubits) + single_qubit_pauli('y',1,nqubits) + single_qubit_pauli('y',2,nqubits)
Sz = single_qubit_pauli('z',0,nqubits) + single_qubit_pauli('z',1,nqubits) + single_qubit_pauli('z',2,nqubits)
Sx = single_qubit_pauli('x',0,nqubits) + single_qubit_pauli('x',2,nqubits)
Sy = single_qubit_pauli('y',0,nqubits) + single_qubit_pauli('y',2,nqubits)
Sz = single_qubit_pauli('z',0,nqubits) + single_qubit_pauli('z',2,nqubits)
H  = 0.25*(Sx*Sx + Sy*Sy + Sz*Sz)

print(H.print_details())

qr      = QuantumRegister(nqubits,'q')
cr      = ClassicalRegister(nqubits,'c')
circ    = QuantumCircuit(qr,cr)
circ.ry(np.pi,1)
for i in range(2):
    circ.cx(1,0)
    circ.ch(1,0)
    circ.cx(1,2)
    circ.x(1)
circ.cx(0,2)

print(circ.draw())

IBMQ.load_account()                                                  
provider = IBMQ.get_provider(hub='ibm-q-internal',group='deployed')  
device   = provider.get_backend('ibmq_athens')                       
backend  = Aer.get_backend('qasm_simulator')                         
instance = QuantumInstance(backend                          = backend,
                           shots                            = 8092,
                           noise_model                      = NoiseModel.from_backend(device.properties()),
                           coupling_map                     = device.configuration().coupling_map,
                           optimization_level               = 3,
                           initial_layout                   = [0,1,2])
res_raw = measure_operator(H,circ,instance)
print("QASM, noise model raw ",res_raw)

instance = QuantumInstance(backend                          = backend,
                           shots                            = 8092,
                           noise_model                      = NoiseModel.from_backend(device.properties()),
                           coupling_map                     = device.configuration().coupling_map,
                           measurement_error_mitigation_cls = CompleteMeasFitter,  # <===== ADD CALIBRATION
                           optimization_level               = 3,
                           initial_layout                   = [0,1,2])
res_1 = measure_operator(H,circ,instance)
print("QASM, noise model RO error mitigation ",res_1)

