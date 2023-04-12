import numpy as np
import itertools
from typing         import List,Optional,Union
from qiskit         import QuantumRegister,QuantumCircuit
from qiskit.circuit import ParameterVector,Parameter
from qiskit.aqua.components.variational_forms import VariationalForm
from qiskit.aqua.components.initial_states    import InitialState

def build_CISD_paulis(mol_data):
    print(mol_data.keys())
    n   = mol_data['n']
    na  = mol_data['na']
    nb  = mol_data['nb']
    hf  = [0]*(2*n)
    occ_a = list(range(na))
    occ_b = list(range(n,n+nb))
    vrt_a = list(set(range(n))-set(occ_a))
    vrt_b = list(set(range(n,2*n))-set(occ_b))
    paulis = []
    for i in occ_a:
        for a in vrt_a:
            pauli_ai = ['I']*(2*n)
            pauli_ai[i] = 'X'
            pauli_ai[a] = 'Y'
            paulis.append(''.join(pauli_ai))
    for i in occ_b:
        for a in vrt_b:
            pauli_ai = ['I']*(2*n)
            pauli_ai[i] = 'X' 
            pauli_ai[a] = 'Y'
            paulis.append(''.join(pauli_ai))
    for i in occ_a:
        for j in occ_a:
               if(i<j):
                  for a in vrt_a:
                      for b in vrt_a:
                          if(a<b):
                             pauli_abij    = ['I']*(2*n)
                             pauli_abij[i] = 'X'
                             pauli_abij[j] = 'X'         
                             pauli_abij[a] = 'X'
                             pauli_abij[b] = 'Y'         
                             paulis.append(''.join(pauli_abij))
    # ---
    for i in occ_b:
        for j in occ_b:
               if(i<j):
                  for a in vrt_b:
                      for b in vrt_b:
                          if(a<b):
                             pauli_abij    = ['I']*(2*n)
                             pauli_abij[i] = 'X'
                             pauli_abij[j] = 'X'
                             pauli_abij[a] = 'X'
                             pauli_abij[b] = 'Y'
                             paulis.append(''.join(pauli_abij))
    # ---
    for i in occ_a:
        for j in occ_b:
               for a in vrt_b:
                   for b in vrt_a:
                          pauli_abij    = ['I']*(2*n)
                          pauli_abij[i] = 'X'
                          pauli_abij[j] = 'X'
                          pauli_abij[a] = 'X'
                          pauli_abij[b] = 'Y'
                          paulis.append(''.join(pauli_abij))
    return paulis[::-1] # first two-electron excitations (ab then aa and bb), then single-electron excitations (b then a)

class PauliExp(VariationalForm):

    def __init__(self,num_qubits=0,paulis=[],initial_state=None,chf=None):
        super().__init__()
        self._num_qubits     = num_qubits
        self._paulis         = [list(p) for p in paulis]
        print(self._paulis)
        self._initial_state  = initial_state
        self._num_parameters = len(paulis)
        self.chf             = chf
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

