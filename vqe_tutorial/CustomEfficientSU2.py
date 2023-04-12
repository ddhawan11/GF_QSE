import numpy as np
import itertools
from typing         import List,Optional,Union
from qiskit         import QuantumRegister,QuantumCircuit
from qiskit.circuit import ParameterVector,Parameter
from qiskit.aqua.components.variational_forms import VariationalForm
from qiskit.aqua.components.initial_states    import InitialState

class CustomEfficientSU2(VariationalForm):

    def __init__(self,num_qubits=0,reps=1,entanglement='linear',initial_state=None,chf=None):
        super().__init__()
        self._num_qubits     = num_qubits
        self._initial_state  = initial_state
        self._num_parameters = num_qubits*(1+reps)
        self._reps           = reps
        if(entanglement=='linear'):
          self._entanglement = [(i,i+1) for i in range(num_qubits-1)]
        else:
          assert(False)
        self.chf             = chf
        self._bounds         = [(-np.pi,np.pi)]*self._num_parameters

    def construct_circuit(self,parameters):
        circuit = self._initial_state.copy()
        m       = 0
        circuit.barrier()
        for i in range(self._num_qubits):
            circuit.ry(parameters[m],i)
            m += 1
        for j in range(self._reps):
            for (a,b) in self._entanglement:
                circuit.cx(a,b)
            for i in range(self._num_qubits):
                circuit.ry(parameters[m],i)
                m += 1
        circuit.barrier()
        if(self.chf is not None): circuit = circuit + self.chf
        return circuit.copy()

