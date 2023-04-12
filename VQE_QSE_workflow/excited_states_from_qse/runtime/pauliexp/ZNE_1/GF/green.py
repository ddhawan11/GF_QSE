def ground_state(matrix):
    import numpy as np
    import itertools
    from scipy import linalg as LA
    from qiskit.aqua.operators.legacy import op_converter
    m = to_matrix(matrix)
    m = np.real(m)
    eps,V  = LA.eigh(m)
    return eps[0],V[:,0]

def eigenpairs(matrix):
    import numpy as np
    import itertools
    from scipy import linalg as LA
    from qiskit.aqua.operators.legacy import op_converter
    m = to_matrix(matrix)
    m = np.real(m)
    eps,V  = LA.eigh(m)
    return eps,V

def to_matrix(oper):
    from qiskit.aqua.operators.legacy import op_converter
    return op_converter.to_matrix_operator(oper)._matrix.todense()

