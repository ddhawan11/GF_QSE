import numpy as np
import subprocess
from subprocess import Popen
import sys, time
sys.path.append('/home/ddhawan/internship_diksha/VQE_QSE_workflow/commons/')
#sys.path.append('/home/ddhawan/internship_diksha/test_mpi/')

from scipy                  import linalg as LA
from job_details            import write_job_details
from subroutines            import build_qse_operators
from post_process           import compute_qse_eigenpairs,compute_qse_greens_function
from QSE                    import QSE
from compute_g0_subroutines import get_Gnot,import_density

numProc = 2
outfile  = open('qse.txt','w')
write_job_details(outfile)

vqe_res      = np.load('vqe_q_uccsd_results.npy',allow_pickle=True).item()

IP_operators = build_qse_operators(class_of_operators='ip',dressed=True, spin='alpha',mol_info=vqe_res) # destruction operators
cr_operators = build_qse_operators(class_of_operators='ea',dressed=False,spin='alpha',mol_info=vqe_res) # creation    operators
EA_operators = build_qse_operators(class_of_operators='ea',dressed=True, spin='alpha',mol_info=vqe_res) # creation    operators
ds_operators = build_qse_operators(class_of_operators='ip',dressed=False,spin='alpha',mol_info=vqe_res) # destruction operators

IP_EA_operators = {"IP_operators": IP_operators,
                   "cr_operators": cr_operators,
                   "EA_operators": EA_operators,
                   "ds_operators": ds_operators}
np.save('IP_EA_operators.npy', IP_EA_operators, allow_pickle=True)
print("operators saved")

t0 = time.time()
print("We got operators fine")
p = subprocess.Popen("mpiexec -n " +str(numProc) + " python -m mpi4py parallel_diagonalization.py", shell=True)
print("p", "mpiexec -n " +str(numProc) + " python -m mpi4py parallel_diagonalization.py")
p.communicate()
print("Done")
t1 = time.time()
print("Time taken by parallel code: ", t1-t0)
e_qse_ip,c_qse_ip = compute_qse_eigenpairs(vqe_res = vqe_res,
                                           qse_res = np.load('qse_ip_matrices.npy',allow_pickle=True).item(),
                                           outfile = outfile)

e_qse_ea,c_qse_ea = compute_qse_eigenpairs(vqe_res = vqe_res,
                                           qse_res = np.load('qse_ea_matrices.npy',allow_pickle=True).item(),
                                           outfile = outfile)

beta   = 100.0
nmax   = 3000
x_mesh = [(2*n+1)*np.pi/beta for n in range(0,nmax+1)]

def ker_ip(x,y):
    return 1.0/(1j*x+y)
def ker_ea(x,y):
    return 1.0/(1j*x-y)

# Eq 6 from George's paper, lower line
g_qse_ip = compute_qse_greens_function(vqe_res = vqe_res,
                                       qse_res = np.load('qse_ip_matrices.npy',allow_pickle=True).item(),
                                       x_mesh  = x_mesh,
                                       kernel  = ker_ip,   # 1/(Iw+dE)
                                       outfile = outfile)
# transposition, as per George's paper
g_qse_ip = g_qse_ip.transpose((0,2,1))

# Eq 6 from George's paper, upper line
g_qse_ea = compute_qse_greens_function(vqe_res = vqe_res,
                                       qse_res = np.load('qse_ea_matrices.npy',allow_pickle=True).item(),
                                       x_mesh  = x_mesh,
                                       kernel  = ker_ea,   # 1/(Iw-dE)
                                       outfile = outfile)


# -------------------------------------------------------------------------------------------------------

ng = g_qse_ip.shape[2]
#print(np.abs(g_qse_ip[:,:ng//2,ng//2:]).max())
#print(np.abs(g_qse_ip[:,ng//2:,:ng//2]).max())
#print(np.abs(g_qse_ea[:,:ng//2,ng//2:]).max())
#print(np.abs(g_qse_ea[:,ng//2:,:ng//2]).max())
#g_qse_ip = g_qse_ip[:,:ng//2,:ng//2] + g_qse_ip[:,ng//2:,ng//2:]
#g_qse_ea = g_qse_ea[:,:ng//2,:ng//2] + g_qse_ea[:,ng//2:,ng//2:]

np.save('qse_green_function_freq.npy',g_qse_ip+g_qse_ea)
print((g_qse_ip+g_qse_ea).shape)
exit()

import matplotlib.pyplot as plt
fig,axs = plt.subplots(2,2)
plt.subplots_adjust(wspace=0.4,hspace=0.4)

axs[0,0].set_title('Re G(IP), G(EA)')
axs[0,1].set_title('Im G(IP), G(EA)')
axs[1,0].set_title('Re G(tot)')
axs[1,1].set_title('Im G(tot)')

ng = g_qse_ip.shape[2]
for p in range(ng):
    for q in range(ng):
        axs[0,0].plot(x_mesh,np.real(g_qse_ip[:,p,q]),label='G(IP,%d,%d)'%(p,q))
        axs[0,1].plot(x_mesh,np.imag(g_qse_ip[:,p,q]),label='G(IP,%d,%d)'%(p,q))
        axs[0,0].plot(x_mesh,np.real(g_qse_ea[:,p,q]),label='G(EA,%d,%d)'%(p,q))
        axs[0,1].plot(x_mesh,np.imag(g_qse_ea[:,p,q]),label='G(EA,%d,%d)'%(p,q))
        axs[1,0].plot(x_mesh,np.real(g_qse_ip[:,p,q]+g_qse_ea[:,p,q]),label='G(%d,%d)'%(p,q))
        axs[1,1].plot(x_mesh,np.imag(g_qse_ip[:,p,q]+g_qse_ea[:,p,q]),label='G(%d,%d)'%(p,q))

for r in range(2):
    for c in range(2):
        axs[r,c].set_xlabel('matsubara frequency [a.u.]')
        axs[r,c].set_ylabel('greens function')
        axs[r,c].legend(ncol=2)
plt.show()



B
