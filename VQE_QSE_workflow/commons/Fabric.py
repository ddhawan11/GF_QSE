import numpy as np
import itertools
from typing         import List,Optional,Union
from qiskit         import QuantumRegister,QuantumCircuit
from qiskit.circuit import ParameterVector,Parameter
from qiskit.aqua.components.variational_forms import VariationalForm
from qiskit.aqua.components.initial_states    import InitialState
from qiskit.circuit.library import RYGate

def G_gate(qc,domain,phi):
    a,b = domain
    qc.ry(np.pi/2.0,a)
    qc.ry(np.pi/2.0,b)
    qc.cz(a,b)
    qc.ry( phi,a)
    qc.ry(-phi,b)
    qc.cz(a,b)
    qc.ry(-np.pi/2.0,a)
    qc.ry(-np.pi/2.0,b)

def OCNOT(qc,a,b):
    qc.x(a)
    qc.cx(a,b)
    qc.x(a)

def QNP_px(qc,domain,theta):
    a,b,c,d = domain
    OCNOT(qc,b,a)
    OCNOT(qc,c,d)
    # ----
    qc.ry(np.pi/2.0,b)
    qc.ry(np.pi/2.0,c)
    qc.cz(b,c)
    ccRY_p = RYGate( theta).control(2,ctrl_state='11')
    qc.append(ccRY_p,[a,d,b])
    ccRY_m = RYGate(-theta).control(2,ctrl_state='11')
    qc.append(ccRY_m,[a,d,c])
    qc.cz(b,c)
    qc.ry(-np.pi/2.0,b)
    qc.ry(-np.pi/2.0,c)
    # ----
    OCNOT(qc,b,a)
    OCNOT(qc,c,d)

def QNP_or(qc,domain,phi):
    a,b,c,d = domain
    G_gate(qc,[a,c],phi)
    G_gate(qc,[b,d],phi)

def Q_gate(qc,domain,theta,phi):
    QNP_px(qc,domain,theta)
    QNP_or(qc,domain,phi)

class Fabric(VariationalForm):

    def __init__(self,num_orbitals=0,depth=1,initial_state=None,chf=None,entanglement='linear'):
        super().__init__()
        self._num_qubits     = 2*num_orbitals
        self._num_orbitals   = num_orbitals
        self._initial_state  = initial_state
        self.depth           = depth
        self.quartets        = [(a,num_orbitals+a,a+1,num_orbitals+a+1) for a in range(num_orbitals-1)]
        self.quartets       += [(a,num_orbitals+a,a+1,num_orbitals+a+1) for a in range(1,num_orbitals-1)] 
        self._num_parameters = 2*len(self.quartets)*self.depth
        self.chf             = chf
        self._bounds         = [(-np.pi,np.pi)]*self._num_parameters

    def construct_circuit(self,parameters):
        circuit = self._initial_state.copy()
        m       = 0
        circuit.barrier()
        for i in range(self.depth):
            for q in self.quartets:
                Q_gate(circuit,q,parameters[m],parameters[m+1])
                m += 2
                circuit.barrier()
        if(self.chf is not None): circuit = circuit + self.chf
        return circuit.copy()

