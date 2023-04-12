import itertools,functools
import numpy as np
from qiskit.quantum_info import Pauli
from qiskit.aqua.operators import WeightedPauliOperator,commutator
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.aqua import aqua_globals

from qiskit.aqua.components.optimizers import SPSA
from qiskit.tools import parallel_map
from scipy import linalg as LA
import numpy as np
from functools import reduce
import time

def adjoint(WPO):
    ADJ = WPO.copy()
    ADJ._paulis = [[np.conj(weight),pauli] for weight,pauli in WPO._paulis]
    return ADJ

def safe_eigh(h, s, lindep=1e-15):
    seig,t = LA.eigh(s)
    #print(seig)
    if seig[0] < lindep:
        idx = seig >= lindep
        t = t[:,idx] * (1/np.sqrt(seig[idx]))
        heff = reduce(np.dot, (t.T.conj(), h, t))
        w, v = LA.eigh(heff)
        v = np.dot(t, v)
    else:
        w, v = LA.eigh(h, s)
    return w, v, seig

def Identity(n):
    zeros = [0]*n
    zmask = [0]*n
    a_x = np.asarray(zmask,dtype=np.bool)
    a_z = np.asarray(zeros,dtype=np.bool)
    return WeightedPauliOperator([(1.0,Pauli(a_x,a_z))])

def construct_paulis(k,n):
    pauli_list = []
    mu = 0
    for z_idx in [x for x in list(itertools.product([0,1],repeat=n)) if sum(x)<k+1]:
        nu = 0 
        for x_idx in [x for x in list(itertools.product([0,1],repeat=n)) if sum(x)<k+1]:
            print(" >>> ",mu,nu)
            zx_idx = [ max(i,j) for (i,j) in zip(z_idx,x_idx)]
            if(sum(zx_idx)<k+1):
               a_x = np.asarray(z_idx,dtype=np.bool)
               a_z = np.asarray(x_idx,dtype=np.bool)
               pauli_list.append(WeightedPauliOperator([(1.0,Pauli(a_x,a_z))]))
            nu += 1
        mu += 1
    return pauli_list

def construct_fermis(num_particles,num_orbitals,active_occ_list,active_unocc_list,same_spin_doubles,
                     method_singles,method_doubles,excitation_type,qubit_mapping,two_qubit_reduction,z2symmetries,n):
    single_excitations, double_excitations = UCCSD.compute_excitation_lists(num_particles,num_orbitals,active_occ_list,
                                             active_unocc_list,same_spin_doubles,method_singles,method_doubles,excitation_type)
    for s in single_excitations: print("singles ",s)
    for d in double_excitations: print("doubles ",d)
    results = parallel_map(UCCSD._build_hopping_operator,
                                   single_excitations + double_excitations,
                                   task_args=(num_orbitals,num_particles,
                                              qubit_mapping,two_qubit_reduction,
                                              z2symmetries),
                                   num_processes=aqua_globals.num_processes)
    hopping_ops = []
    for op, index in results:
            if op is not None and not op.is_empty():
                hopping_ops.append(op)
    #for ir,r in enumerate(hopping_ops): print("hopping operator %d \n " % ir,r.print_details())
    eye = WeightedPauliOperator([(1.0,Pauli(np.asarray([0]*n,dtype=np.bool),np.asarray([0]*n,dtype=np.bool)))])
    return [eye]+hopping_ops

