import numpy as np
import itertools
from typing         import List,Optional,Union
from qiskit         import QuantumRegister,QuantumCircuit
from qiskit.circuit import ParameterVector,Parameter
from qiskit.aqua.components.variational_forms import VariationalForm
from qiskit.aqua.components.initial_states    import InitialState

class CustomVarForm(VariationalForm):

    def __init__(self,num_qubits=0,paulis=[],initial_state=None,chf=None):
        super().__init__()
        self._num_qubits     = num_qubits
        self._paulis         = paulis
        self._initial_state  = initial_state
        self._num_parameters = len(paulis)
        self.chf             = chf
        print(paulis)
        print(len(paulis))
        self._bounds         = [(-np.pi,np.pi)]*self._num_parameters

    def construct_circuit(self,parameters):
        circuit = self._initial_state.copy()
        m       = 0
        circuit.barrier()
        for p in self._paulis:
            idx = [j for j,pj in enumerate(p) if pj!='I']
            for j,pj in enumerate(p):
                if(pj=='X'): circuit.h(j)
                if(pj=='Y'): circuit.s(j); circuit.h(j)
            for R in range(len(idx)-1):
                circuit.cx(idx[R],idx[R+1]) 
            circuit.rz(parameters[m],idx[len(idx)-1])
            for R in range(len(idx)-1)[::-1]:
                circuit.cx(idx[R],idx[R+1])
            for j,pj in enumerate(p):
                if(pj=='X'): circuit.h(j)
                if(pj=='Y'): circuit.h(j); circuit.sdg(j)
            m += 1
            circuit.barrier()
        if(self.chf is not None): circuit = circuit + self.chf
        return circuit.copy()

