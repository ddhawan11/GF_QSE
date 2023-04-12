def diagonalize(operator):
    import numpy as np
    from   qiskit.aqua.algorithms import NumPyEigensolver

    res = NumPyEigensolver(operator,k=1).run()
    return np.real(res['eigenvalues'][0])

def taper_principal_operator(qubit_op,target_sector=None):
    from qiskit.aqua.operators import Z2Symmetries

    z2_symmetries = Z2Symmetries.find_Z2_symmetries(qubit_op)
    nsym        = len(z2_symmetries.sq_paulis)
    z2syms      = z2_symmetries
    sqlist      = z2_symmetries.sq_list
    tapered_ops = z2_symmetries.taper(qubit_op)

    if(target_sector is None):
       e_min = 1e10
       i_min = -1
       for i in range(len(tapered_ops)):
           e_i = diagonalize(tapered_ops[i])
           if(e_i<e_min):
              e_min = e_i
              i_min = i
       return tapered_ops[i_min],z2syms,sqlist,i_min
    else:
       return tapered_ops[target_sector],z2syms,sqlist,target_sector

def get_matrix(O):
    from qiskit.aqua.operators.legacy import op_converter
    return op_converter.to_matrix_operator(O)._matrix.todense()

def taper_auxiliary_operators(A_op,z2syms,target_sector):
    import numpy as np
    from qiskit.aqua.operators import WeightedPauliOperator
    from qiskit.quantum_info   import Pauli
    from qiskit.aqua.operators import commutator

    Ws_list = [ s for s in z2syms.symmetries]
    Ws_list = [ WeightedPauliOperator(paulis=[[1.0,s]]) for s in Ws_list] 
    Ws_list = [ get_matrix(s) for s in Ws_list]

    the_ancillas = []
    for A in A_op:
        comm = True
        A_mat = get_matrix(A)
        for s_mat in Ws_list:
            comm_s = np.dot(A_mat,s_mat)-np.dot(s_mat,A_mat)
            comm_s = np.abs(comm_s).max()
            if(comm_s>1e-6): comm=False
        #for s in z2syms.symmetries:
        #    Ws = WeightedPauliOperator(paulis=[[1.0,s]])
        #    s_mat = get_matrix(Ws)
        #    comm_s = commutator(A,WeightedPauliOperator(paulis=[[1.0,s]])).chop(1e-6)
        #    if(not comm_s.is_empty()): comm=False
        if(not comm):
           the_ancillas.append(WeightedPauliOperator(paulis=[[0.0,z2syms.symmetries[0]]]))
        else:
           A_taper = z2syms.taper(A)
           if(type(A_taper)==list): the_ancillas.append(A_taper[target_sector])
           else: the_ancillas.append(A_taper)
    return the_ancillas

