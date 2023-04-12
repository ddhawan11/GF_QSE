import numpy as np

from qiskit.chemistry.fermionic_operator import FermionicOperator
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.aqua.operators import Z2Symmetries

def creators_and_destructors(n_spin_orbitals,qubit_mapping,two_qubit_reduction,num_particles):
    F_op   = FermionicOperator(h1=np.zeros((n_spin_orbitals,n_spin_orbitals)))
    if(type(qubit_mapping)==str):
       map_type = qubit_mapping
    else:
       map_type = qubit_mapping.value.lower()
    if map_type == 'jordan_wigner':
        a_list = F_op._jordan_wigner_mode(n_spin_orbitals)
    elif map_type == 'parity':
        a_list = F_op._parity_mode(n_spin_orbitals)
    elif map_type == 'bravyi_kitaev':
        a_list = F_op._bravyi_kitaev_mode(n_spin_orbitals)
    elif map_type == 'bksf':
        return bksf_mapping(self)
    else:
        assert(False)
    # creators
    c_list = [WeightedPauliOperator([(0.5,a1),(-0.5j,a2)]) for a1,a2 in a_list]
    # destructors
    d_list = [WeightedPauliOperator([(0.5,a1),(+0.5j,a2)]) for a1,a2 in a_list]

    if map_type == 'parity' and two_qubit_reduction:
        c_list = [Z2Symmetries.two_qubit_reduction(c,num_particles) for c in c_list]
        d_list = [Z2Symmetries.two_qubit_reduction(d,num_particles) for d in d_list]

    return c_list,d_list

def spin_summed_one_body_operators(n_spin_orbitals,qubit_mapping,two_qubit_reduction,num_particles):
    c_list = []
    if(type(qubit_mapping)==str):
       map_type = qubit_mapping
    else:
       map_type = qubit_mapping.value.lower()
    for p in range(n_spin_orbitals//2):
        for q in range(n_spin_orbitals//2):
            h1   = np.zeros((n_spin_orbitals,n_spin_orbitals))
            h1[p,q] = 1.0
            h1[p+n_spin_orbitals//2,q+n_spin_orbitals//2] = 1.0
            F_op = FermionicOperator(h1=h1)
            c_list.append(F_op.mapping(map_type))
    if map_type == 'parity' and two_qubit_reduction:
        c_list = [Z2Symmetries.two_qubit_reduction(c,num_particles) for c in c_list]
    return c_list

def build_qse_operators(excitations,dressed,mol_info):
    if(type(mol_info)==dict):
       n_spin_orbitals     = mol_info['n_spin_orbitals']
       qubit_mapping       = mol_info['qubit_mapping']
       two_qubit_reduction = mol_info['two_qubit_reduction']
       num_particles       = mol_info['num_particles']
    else:
       n_spin_orbitals     = mol_info._molecule_info['num_orbitals']
       qubit_mapping       = mol_info._qubit_mapping 
       two_qubit_reduction = mol_info._two_qubit_reduction
       num_particles       = mol_info._molecule_info['num_particles']
 
    c_list,d_list = creators_and_destructors(n_spin_orbitals     = n_spin_orbitals    ,
                                             qubit_mapping       = qubit_mapping      ,
                                             two_qubit_reduction = two_qubit_reduction,
                                             num_particles       = num_particles      )

    if(excitations=='ip'):
       pool = d_list
    if(excitations=='ea'):
       pool = c_list
    if(num_particles[0]==num_particles[1]):
       pool = pool[:len(pool)//2]
    if(dressed):
       o_list = spin_summed_one_body_operators(n_spin_orbitals     = n_spin_orbitals    ,
                                               qubit_mapping       = qubit_mapping      ,
                                               two_qubit_reduction = two_qubit_reduction,
                                               num_particles       = num_particles      )
       pool = [x*y for x in pool for y in o_list]
    return pool

