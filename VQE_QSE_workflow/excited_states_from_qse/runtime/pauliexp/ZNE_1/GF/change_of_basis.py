import numpy as np
from   scipy import linalg as LA
from   qiskit.aqua.operators import WeightedPauliOperator
from   qiskit                import *
from   qiskit.chemistry      import FermionicOperator
from   qiskit.aqua.operators import Z2Symmetries
from   qiskit.aqua.operators import WeightedPauliOperator
from   qiskit.quantum_info   import Pauli

def Identity(n):
    zeros = [0]*n
    zmask = [0]*n
    a_x = np.asarray(zmask,dtype=np.bool)
    a_z = np.asarray(zeros,dtype=np.bool)
    return WeightedPauliOperator([(1.0,Pauli(a_x,a_z))])

def decompose_into_givens(U_original,tol=1e-6):
    detU = LA.det(U_original)
    print("determinant of U ",detU)
    # Hn ... H1 U = 1 (inverse of U)
    U      = U_original.copy()
    nr,nc  = U.shape[0],U.shape[1]
    r,c    = nr-1,0
    G_list = []
    while(c<nc):
       while(r>c):
          [a,b] = U[r-1:r+1,c]
          if(np.abs(b)>tol):
             cx =  a/np.sqrt(a**2+b**2)
             sx = -b/np.sqrt(a**2+b**2)
             G  = np.eye(nr)
             G[r-1,r-1:r+1] = np.array([cx,-sx])
             G[r  ,r-1:r+1] = np.array([sx, cx])
             G_list.append([r,np.arctan2(sx,cx)])
             U  = np.dot(G,U)
          r -= 1
       r = nr-1
       c += 1
    # U = inv(H1) ... inv(Hn) = Gn ... G1
    G_list = [[r,-t] for [r,t] in G_list[::-1]]
    U_givens = np.eye(nr)
    for r,t in G_list:
        G  = np.eye(nr)
        cx,sx          = np.cos(t),np.sin(t)
        G[r-1,r-1:r+1] = np.array([cx,-sx])
        G[r  ,r-1:r+1] = np.array([sx, cx])
        U_givens = np.dot(G,U_givens)
    return G_list

def find_pauli_domain(p):
    str_p  = str(p)
    dom_p  = [(p.num_qubits-1)-i for i in range(len(str_p)) if str_p[i]!='I']
    return len(dom_p),dom_p

def givens(circuit,givens_list):
    dphi = 0.0
    for g in givens_list:
        dom_p = [set(find_pauli_domain(p)[1]) for c,p in g.paulis]
        dom_g = list(set().union(*dom_p))
        print("Givens operator ",g.print_details())
        if(len(dom_g)==0): dphi += g.paulis[0][0]
        if(len(dom_g)==1): apply_compressed_circuit(g,circuit,dom_g,offset=0)
        if(len(dom_g)==2): apply_compressed_circuit(g,circuit,dom_g,offset=0)
        if(len(dom_g)>=3):
           for c,p in g.paulis:
               g_p    = WeightedPauliOperator([[c,p]])
               dom_gp = find_pauli_domain(p)[1]
               assert(len(dom_gp)<=2)
               apply_compressed_circuit(g_p,circuit,dom_gp,offset=1)
    return dphi

def fermi_qubit(qubit_mapping,two_qubit_reduction,nelec,t1,t2=None):
    if(type(qubit_mapping)==str):
       map_type = qubit_mapping
    else:
       map_type = qubit_mapping.value.lower()
    if(t2 is None): X = FermionicOperator(t1).mapping(map_type=map_type)
    else:           X = FermionicOperator(t1,t2).mapping(map_type=map_type)
    if(map_type=='parity' and two_qubit_reduction):
       X = Z2Symmetries.two_qubit_reduction(X,nelec)
    return X

def givens_operators(G_list,operators):
    qubit_mapping       = operators['mapping']
    two_qubit_reduction = operators['2qr']
    nelec               = operators['particles']
    n                   = operators['orbitals']//2
    givens = []
    for r,theta in G_list:
        givens.append(single_givens_operator(r-1,r,theta,qubit_mapping,two_qubit_reduction,nelec,n))
        givens.append(single_givens_operator(n+r-1,n+r,theta,qubit_mapping,two_qubit_reduction,nelec,n))
    return givens

def single_givens_operator(i,j,theta,qubit_mapping,two_qubit_reduction,nelec,n):
    t1 = np.zeros((2*n,2*n))
    t1[i,j] = -theta
    t1[j,i] =  theta
    return fermi_qubit(qubit_mapping,two_qubit_reduction,nelec,t1)

def apply_compressed_circuit(wpo,qc,domain,offset):

    from scipy                        import linalg as LA
    from qiskit.aqua.operators.legacy import op_converter

    from qiskit.quantum_info.synthesis.one_qubit_decompose import OneQubitEulerDecomposer
    from qiskit.quantum_info.synthesis.two_qubit_decompose import TwoQubitBasisDecomposer
    from qiskit.circuit.library.standard_gates.x           import CXGate

    wpo_reduced = WeightedPauliOperator([[c,Pauli(np.take(p.z,domain),np.take(p.x,domain))] for c,p in wpo.paulis])
    mat = LA.expm(op_converter.to_matrix_operator(wpo_reduced)._matrix.todense())
    if(len(domain)==1): C = OneQubitEulerDecomposer().__call__(mat)
    if(len(domain)==2): C = TwoQubitBasisDecomposer(CXGate()).__call__(mat)
    for g in C:
        instruction,q1,q2 = g
        if(instruction.name=='u3'):
           t1,t2,t3 = instruction.params
           qc.u3(t1,t2,t3,domain[q1[0].index]+offset)
        if(instruction.name=='cx'):
           qc.cx(domain[q1[0].index]+offset,domain[q1[1].index]+offset)

def construct_cob_circuit(mol_dict,operators):
    from tapering import taper_auxiliary_operators
    V      = mol_dict['V']
    G_list = decompose_into_givens(V)
    G_list = givens_operators(G_list,operators)
    if(operators['tapering']):
       G_list = taper_auxiliary_operators(G_list,operators['tapering_info'][0],operators['target_sector'])
    qc = QuantumCircuit(G_list[0].num_qubits)
    givens(qc,G_list)
    return qc

