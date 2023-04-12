import numpy as np
from qiskit.chemistry.fermionic_operator import FermionicOperator
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.aqua.operators import Z2Symmetries

def Identity(n):
    from qiskit.aqua.operators import WeightedPauliOperator
    from qiskit.quantum_info import Pauli
    import numpy as np
    zeros = [0]*n
    zmask = [0]*n
    a_x = np.asarray(zmask,dtype=np.bool)
    a_z = np.asarray(zeros,dtype=np.bool)
    return WeightedPauliOperator([(1.0,Pauli(a_x,a_z))])

def creators_destructors(n_spin_orbitals,qubit_mapping,two_qubit_reduction,num_particles):
    F_op = FermionicOperator(h1=np.zeros((n_spin_orbitals,n_spin_orbitals)))
    if(type(qubit_mapping)==str): map_type = qubit_mapping
    else:                         map_type = qubit_mapping.value.lower()
    if(map_type=='jordan_wigner'):   a_list = F_op._jordan_wigner_mode(n_spin_orbitals)
    elif(map_type=='parity'):        a_list = F_op._parity_mode(n_spin_orbitals)
    elif(map_type=='bravyi_kitaev'): a_list = F_op._bravyi_kitaev_mode(n_spin_orbitals)
    elif(map_type=='bksf'):          return bksf_mapping(self)
    else:                            assert(False)

    c_list = [WeightedPauliOperator([(0.5,a1),( 0.5j,a2)]) for a1,a2 in a_list]
    d_list = [WeightedPauliOperator([(0.5,a1),(-0.5j,a2)]) for a1,a2 in a_list]

    if(map_type=='parity' and two_qubit_reduction):
        c_list = [Z2Symmetries.two_qubit_reduction(c,num_particles) for c in c_list]
        d_list = [Z2Symmetries.two_qubit_reduction(d,num_particles) for d in d_list]
    return c_list,d_list

def build_qse_operators(class_of_operators,dressed,spin,mol_info):
    from qiskit.chemistry import FermionicOperator
    operators = []
    n         = mol_info['operators']['orbitals']
    map_type  = mol_info['operators']['mapping']
    tqr       = mol_info['operators']['2qr']
    nelec     = mol_info['operators']['particles']
    if(class_of_operators=='ip'):   operators = creators_destructors(n,map_type,tqr,nelec)[0]
    elif(class_of_operators=='ea'): operators = creators_destructors(n,map_type,tqr,nelec)[1]
    else:                           assert(False)

    if(spin=='alpha'):  operators = operators[:len(operators)//2]
    elif(spin=='beta'): operators = operators[len(operators)//2:]
    elif(spin=='both'): operators = operators
    else:               assert(False)

    if(dressed):
       ob_operators = []
       for p in range(n//2):
           for q in range(n//2):
               h_1                = np.zeros((n,n))
               h_1[p,q]           = 1.0
               h_1[p+n//2,q+n//2] = 1.0
               o_h_1    = FermionicOperator(h1=h_1).mapping(map_type=map_type)
               if(map_type=='parity' and tqr):
                   o_h_1 = Z2Symmetries.two_qubit_reduction(X,nelec)
               ob_operators.append(o_h_1)

       print(ob_operators[0].paulis, operators[0].paulis)
       return [x*y for x in operators for y in ob_operators]
    else:
       return operators

