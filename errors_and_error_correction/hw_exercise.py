from qiskit      import *
from qiskit.aqua import QuantumInstance
from qiskit.providers.aer.backends.aerbackend import AerBackend
from qiskit.providers.ibmq.ibmqbackend import IBMQBackend
from qiskit.providers.aer import noise
from qiskit.providers.aer.noise import NoiseModel
from qiskit.ignis.mitigation.measurement import (complete_meas_cal,CompleteMeasFitter)

from subroutines import *

def make_richardson_circuit(circ,repetitions=1):
    nq = circ.num_qubits
    qr      = QuantumRegister(nq,'q')
    cr      = ClassicalRegister(nq,'c')
    circ_R  = QuantumCircuit(qr,cr)
    for g in circ:
        instruction,qubits,_ = g   
        if(instruction.name=='cx'):
           for k in range(repetitions): circ_R.cx(qubits[0],qubits[1]); circ_R.barrier()
        elif(instruction.name=='u3'):
           t1,t2,t3 = instruction.params
           circ_R.u3(t1,t2,t3,qubits[0])
        elif(instruction.name=='x'):
           circ_R.x(qubits[0])
        elif(instruction.name=='ry'):
           t1 = instruction.params[0]
           circ_R.ry(t1,qubits[0])
        elif(instruction.name=='ch'):
           for k in range(repetitions): circ_R.ch(qubits[0],qubits[1]); circ_R.barrier()
        elif(instruction.name=='barrier'):
           circ_R.barrier()
        else:
           assert(False)
    return circ_R

nqubits = 2
Sx = single_qubit_pauli('x',0,nqubits) + single_qubit_pauli('x',1,nqubits)
Sy = single_qubit_pauli('y',0,nqubits) + single_qubit_pauli('y',1,nqubits)
Sz = single_qubit_pauli('z',0,nqubits) + single_qubit_pauli('z',1,nqubits)
H  = 0.25*(Sx*Sx + Sy*Sy + Sz*Sz)

qr   = QuantumRegister(nqubits,'q')
cr   = ClassicalRegister(nqubits,'c')
circ = QuantumCircuit(qr,cr)
circ.ry(np.pi/6.0,0)
circ.cx(0,1)

# qasm: important to account for statistical uncertainties
# (we will need them in the measurement of H, in the qse algorithms, in the green's functions)

for s in [100,200,500,1000,2000,4000,8000]:
    backend  = Aer.get_backend('qasm_simulator')
    instance = QuantumInstance(backend=backend,shots=s)
    mu,sigma = measure_operator(H,circ,instance)
    print("QASM --- shots = %d, E = %.6f +/- %.6f " % (s,mu,sigma))

# on hardware there is noise
# i \hbar \frac{d \rho(device))}{dt} = [ H(device,t) , \rho(device) ] + \sum_k \gamma_k(t) L_k(t) \rho(device) L_k(t)^\dagger
# H(device,t) = H(drive,t) + H(coherent-error,t)        <=== uncontrolled Hamiltonian, source of coherent (unitary) errors
# \sum_k \gamma_k(t) L_k(t) \rho(device) L_k(t)^\dagger <=== uncontrolled Lindblad, source of incoherent (non-unitary) error

IBMQ.load_account()                                                 
provider = IBMQ.get_provider(hub='ibm-q-internal',group='deployed')
backend  = provider.get_backend('ibmq_mumbai')
instance = QuantumInstance(backend                          = backend,
                           shots                            = 8092,
                           optimization_level               = 0,
                           initial_layout                   = [0,1])
res_raw = measure_operator(H,circ,instance)
print("HW RAW --- shots = %d, E = %.6f +/- %.6f " % (8092,res_raw[0],res_raw[1]))

instance = QuantumInstance(backend                          = backend,
                           shots                            = 8092,
                           optimization_level               = 0,
                           measurement_error_mitigation_cls = CompleteMeasFitter, # <===== readout error mitigation
                           initial_layout                   = [0,1])
res_1 = measure_operator(H,circ,instance)
print("HW READOUT ERROR MITIGATION (complete) --- shots = %d, E = %.6f +/- %.6f " % (8092,res_1[0],res_1[1]))

# ------------------------------------------------------------------
# readout error mitigation only corrects for measurement errors
# but errors occur as the circuit is executed, not just in the measurement phase
# the richardson extrapolation (based on pulse stretching) corrects for gate errors
# the proper tool is PULSE https://qiskit.org/textbook/ch-quantum-hardware/calibrating-qubits-pulse.html but PULSE is difficult/technical
# k = 0  CNOT -> CNOT
# k = 1  CNOT -> CNOT CNOT CNOT
# k = 2  CNOT -> CNOT CNOT CNOT CNOT CNOT
# ------------------------------------------------------------------

circ_R = make_richardson_circuit(circ,repetitions=3)

res_3 = measure_operator(H,circ_R,instance)
print("NOISE MODEL --- 'richardson' circuit with 3 CNOTs instead of 1 CNOT ",res_3)

# E(k) = a + b*k (k = number of repetitions)
# E(1) = a + b
# E(3) = a + 3*b
# E(3)-E(1)/2 = b
# a = E(1) + (E(3)-E(1))/2

res_richardson = ( res_1[0]+(res_1[0]-res_3[0])/2.0 , np.sqrt(res_1[1]**2+(res_1[1]**2+res_3[1]**2)/4.0) )
print("NOISE MODEL --- 'richardson' extrapolation ",res_richardson)

