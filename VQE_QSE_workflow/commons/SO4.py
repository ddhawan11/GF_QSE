import qiskit
import itertools
import logging
import sys
import numpy as np

from qiskit.quantum_info                      import Pauli
from qiskit.aqua.operators                    import WeightedPauliOperator
from qiskit.aqua.components.variational_forms import VariationalForm
from qiskit                                   import QuantumRegister, QuantumCircuit
from qiskit.aqua                              import QuantumInstance,aqua_globals

def add_unitary_gate(circuit,qubit1,qubit2,parameters,p0):
    circuit.s(qubit1)
    circuit.s(qubit2)
    circuit.h(qubit2)
    circuit.cx(qubit2,qubit1)
    circuit.u3(parameters[p0],parameters[p0+1],parameters[p0+2],qubit1); p0 += 3
    circuit.u3(parameters[p0],parameters[p0+1],parameters[p0+2],qubit2); p0 += 3
    circuit.cx(qubit2,qubit1)
    circuit.h(qubit2)
    circuit.sdg(qubit1)
    circuit.sdg(qubit2)

class var_form_unitary(VariationalForm):
    def __init__(self,nqubit,depth,initial_state,parameters=None, entanglement: str='linear',chf=None):
        super().__init__()
        self._configuration  = {'name': 'var_form_unitary'}
        self._num_qubits     = nqubit
        self._depth          = depth
        self._initial_state  = initial_state
        self._entanglement   = entanglement
        if entanglement == 'linear':
            self._domain     = [(i,i+1) for i in range(0,nqubit-1,2)] + [(i,i+1) for i in range(1,nqubit-1,2)]
        if entanglement == 'full':
            self._domain     = [(i,j) for i in range(0,nqubit-1) for j in range(i+1,nqubit)]
        self._num_parameters = 6*depth*len(self._domain)
        self._bounds         = [(-np.pi,np.pi)]*self._num_parameters
        self.chf             = chf

    def construct_circuit(self,parameters=None,q=None):

        if parameters is None:
           parameters = np.random.rand(self._num_parameters)

        circuit = self._initial_state.copy()
        circuit.barrier()

        p0 = 0
        for k in range(self._depth):
            for (i,j) in self._domain:
                add_unitary_gate(circuit,i,j,parameters,p0); p0 += 6
            circuit.barrier()

        if(self.chf is not None): circuit = circuit + self.chf
        return circuit

