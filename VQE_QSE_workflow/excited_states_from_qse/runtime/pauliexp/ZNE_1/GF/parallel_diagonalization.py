import sys,time,itertools
import numpy as np
# download this library asap
from   mpi4py import MPI
from   scipy  import linalg as LA
from QSE      import QSE


vqe_res      = np.load('vqe_q_uccsd_results.npy',allow_pickle=True).item()

operators_IP_EA = np.load("IP_EA_operators.npy", allow_pickle= True).item()

QSE_IP_calc  = QSE(psi           = vqe_res['vqe_circuit'],
                   operators     = operators_IP_EA["IP_operators"],
                   mol_info      = vqe_res,
                   instance_dict = {'instance':'statevector_simulator','shots':1})

QSE_EA_calc  = QSE(psi           = vqe_res['vqe_circuit'],
                   operators     = operators_IP_EA["EA_operators"],
                   mol_info      = vqe_res,
                   instance_dict = {'instance':'statevector_simulator','shots':1})


size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()
comm = MPI.COMM_WORLD

#t0 = time.time()
outfile  = open('qse.txt','w')

if(rank==0): n = len(QSE_IP_calc.operators)#4#int(np.loadtxt('data.in'))
else:        n = 0
n    = comm.bcast(n,root=0)
args = None

# pairs of elements (i,j) that process k will take care of
print("rank", rank, "size", size)
i_rank = [(i,j) for m,(i,j) in enumerate(itertools.product(range(n),range(n))) if i<=j and m%size==rank]
print("I am processor %d of %d and I will handle matrix elements %s " % (rank,size,str(i_rank)), flush = True)

#t1 = time.time()

S_matrix_IP = np.zeros((n,n,2))
H_matrix_IP = np.zeros((n,n,2))
S_matrix_EA = np.zeros((n,n,2))
H_matrix_EA = np.zeros((n,n,2))
for (i,j) in i_rank:
    QSE_IP_calc.get_qse_element(i, j, outfile=outfile)
    S_matrix_IP[i,j,:] = QSE_IP_calc.matrices["S"]
    H_matrix_IP[i,j,:] = QSE_IP_calc.matrices["H"]
    QSE_EA_calc.get_qse_element(i, j, outfile=outfile)
    S_matrix_EA[i,j,:] = QSE_EA_calc.matrices["S"]
    H_matrix_EA[i,j,:] = QSE_EA_calc.matrices["H"]
    S_matrix_IP[j,i,:] = S_matrix_IP[i,j,:]
    H_matrix_IP[j,i,:] = H_matrix_IP[i,j,:]
    S_matrix_EA[j,i,:] = S_matrix_EA[i,j,:]
    H_matrix_EA[j,i,:] = H_matrix_EA[i,j,:]

   
S_matrix_IP = comm.reduce(S_matrix_IP,root=0)
H_matrix_IP = comm.reduce(H_matrix_IP,root=0)
S_matrix_EA = comm.reduce(S_matrix_EA,root=0)
H_matrix_EA = comm.reduce(H_matrix_EA,root=0)
print("S_matrix_IP", S_matrix_IP)
print("H_matrix_IP", H_matrix_IP)
print("S_matrix_EA", S_matrix_EA)
print("H_matrix_EA", H_matrix_EA)


m      = len(operators_IP_EA["cr_operators"])
i_rank = [(i,j) for p,(i,j) in enumerate(itertools.product(range(m),range(n))) if p%size==rank]
print(i_rank)
print("I am processor %d of %d and I will handle matrix elements %s " % (rank,size,str(i_rank)))

C_matrix_IP = np.zeros((m,n,2), dtype = 'complex')
C_matrix_EA = np.zeros((m,n,2), dtype = 'complex')

for (i,j) in i_rank:
    QSE_IP_calc.compute_transition_matrix_element(i, j,
                                                  operators_IP_EA["cr_operators"],
                                                  outfile=outfile)
    QSE_EA_calc.compute_transition_matrix_element(i, j,
                                                  operators_IP_EA["ds_operators"],
                                                  outfile=outfile)
    C_matrix_IP[i,j,:] = QSE_IP_calc.matrices["C"]
    C_matrix_EA[i,j,:] = QSE_EA_calc.matrices["C"]
#    C_matrix[j,i,:] = C_matrix[i,j,:]

C_matrix_IP = comm.reduce(C_matrix_IP,root=0)
C_matrix_EA = comm.reduce(C_matrix_EA,root=0)
print("C_matrix_IP", C_matrix_IP)
print("C_matrix_EA", C_matrix_EA)


np.save("qse_ip_matrices.npy",{"S":S_matrix_IP, "H":H_matrix_IP, "C":C_matrix_IP},allow_pickle=True)
np.save("qse_ea_matrices.npy",{"S":S_matrix_EA, "H":H_matrix_EA, "C":C_matrix_EA},allow_pickle=True)


#exit()
# t2 = time.time()

# if(rank==0):
#    # then the root diagonalizes the matrix
#    eps,U = LA.eigh(m_rank)
#    np.save('data.eigval',eps)
#    np.save('data.eigvec',U)

# t3 = time.time()

# print("I am processor %d of %d and it took me %.6f, %.6f, %.6f s to diagonalize, compute matrix elements, and prepare" % (rank,size,t3-t2,t2-t1,t1-t0))
