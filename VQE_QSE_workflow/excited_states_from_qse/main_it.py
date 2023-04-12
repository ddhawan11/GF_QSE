import numpy as np
import sys
sys.path.append('../commons/')

from   job_details  import write_job_details
from   subroutines  import build_qse_operators
from   post_process import compute_qse_eigenpairs,compute_qse_greens_function
from   QSE          import QSE

outfile  = open('qse.txt','w')
write_job_details(outfile)

vqe_res      = np.load('../ground_state_from_vqe/results/vqe_q_uccsd_results.npy',allow_pickle=True).item()

IP_operators = build_qse_operators(class_of_operators='ip',dressed=False,spin='alpha',mol_info=vqe_res) # destruction operators
cr_operators = build_qse_operators(class_of_operators='ea',dressed=False,spin='alpha',mol_info=vqe_res) # creation    operators
EA_operators = build_qse_operators(class_of_operators='ea',dressed=False,spin='alpha',mol_info=vqe_res) # creation    operators
ds_operators = build_qse_operators(class_of_operators='ip',dressed=False,spin='alpha',mol_info=vqe_res) # destruction operators

# ------------------------------------------------------------------------------------------------------------

QSE_IP_calc  = QSE(psi           = vqe_res['vqe_circuit'],
                   operators     = IP_operators,
                   mol_info      = vqe_res,
                   instance_dict = {'instance':'statevector_simulator','shots':1})
QSE_IP_calc.construct_qse_matrices(outfile=outfile,resfile='qse_ip_matrices.npy',postselection='none')
QSE_IP_calc.compute_transition_matrix_elements(transition_operators = cr_operators,
                                               outfile              = outfile,
                                               resfile              = 'qse_ip_matrices.npy') # < Psi(VQE) | a*(p) a(q) | Psi(VQE) >

# ------------------------------------------------------------------------------------------------------------

QSE_EA_calc  = QSE(psi           = vqe_res['vqe_circuit'],
                   operators     = EA_operators,
                   mol_info      = vqe_res,
                   instance_dict = {'instance':'statevector_simulator','shots':1})
QSE_EA_calc.construct_qse_matrices(outfile=outfile,resfile='qse_ea_matrices.npy',postselection='none')
QSE_EA_calc.compute_transition_matrix_elements(transition_operators = ds_operators,
                                               outfile              = outfile,
                                               resfile              = 'qse_ea_matrices.npy') # < Psi(VQE) | a(p) a*(q) | Psi(VQE) >

# ------------------------------------------------------------------------------------------------------------

e_qse_ip,c_qse_ip = compute_qse_eigenpairs(vqe_res = vqe_res,
                                           qse_res = np.load('qse_ip_matrices.npy',allow_pickle=True).item(),
                                           outfile = outfile)

e_qse_ea,c_qse_ea = compute_qse_eigenpairs(vqe_res = vqe_res,
                                           qse_res = np.load('qse_ea_matrices.npy',allow_pickle=True).item(),
                                           outfile = outfile)

tau    = 1e-6
nmax   = 100
x_mesh = [tau*float(n)/float(nmax) for n in range(-nmax,nmax+1)]

def ker_ip(x,y):
    return np.heaviside(-x,0.5)*np.exp(x*y)
def ker_ea(x,y):
    return -np.heaviside(x,0.5)*np.exp(-x*y)

# Eq 6 from George's paper, lower line
g_qse_ip = compute_qse_greens_function(vqe_res = vqe_res,
                                       qse_res = np.load('qse_ip_matrices.npy',allow_pickle=True).item(),
                                       x_mesh  = x_mesh,
                                       kernel  = ker_ip,
                                       outfile = outfile)
# transposition, as per George's paper
g_qse_ip = g_qse_ip.transpose((0,2,1))

# Eq 6 from George's paper, upper line
g_qse_ea = compute_qse_greens_function(vqe_res = vqe_res,
                                       qse_res = np.load('qse_ea_matrices.npy',allow_pickle=True).item(),
                                       x_mesh  = x_mesh,
                                       kernel  = ker_ea,
                                       outfile = outfile)

G      = g_qse_ip+g_qse_ea

# G(pq,epsilon+) = - <c(p) c*(q) >
GP     = G[nmax+1,:,:]
GP_ref = -QSE_EA_calc.matrices['S'][:,:,0]
print("imaginary-time green's function matches rdm? (EA) ",np.abs(GP-GP_ref).max())

# G(pq,epsilon-) = <c*(q) c(p) >
GM     = G[nmax-1,:,:]
GM_ref = QSE_IP_calc.matrices['S'][:,:,0]
print("imaginary-time green's function matches rdm? (IP) ",np.abs(GM-GM_ref).max())

exit()
