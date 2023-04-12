import numpy as np
import sys, time
import qiskit
from qiskit import IBMQ
IBMQ.load_account()
sys.path.append('/Users/mario/Documents/GitHub/UMich/internship_diksha/VQE_QSE_workflow/commons')
#/home/ddhawan/GF_extrapolate_2/VQE_QSE_workflow/commons/')
#/Users/mario/Documents/GitHub/UMich/internship_diksha/VQE_QSE_workflow/excited_states_from_qse

from scipy                  import linalg as LA
from job_details            import write_job_details
from subroutines            import build_qse_operators
from post_process           import compute_qse_eigenpairs,compute_qse_greens_function,statistical_analysis
from QSE                    import QSE
from compute_g0_subroutines import get_Gnot,import_density

outfile  = open('qse_HW.txt','w')
write_job_details(outfile)

vqe_res      = np.load('vqe_q_uccsd_results.npy',allow_pickle=True).item()

IP_operators = build_qse_operators(class_of_operators='ip',dressed=False, spin='alpha',mol_info=vqe_res) # destruction operators
cr_operators = build_qse_operators(class_of_operators='ea',dressed=False,spin='alpha',mol_info=vqe_res) # creation    operators
EA_operators = build_qse_operators(class_of_operators='ea',dressed=False, spin='alpha',mol_info=vqe_res) # creation    operators
ds_operators = build_qse_operators(class_of_operators='ip',dressed=False,spin='alpha',mol_info=vqe_res) # destruction operators

t0 = time.time()
#------------------------------------------------------------------------------------------------------------
provider  = IBMQ.get_provider(hub='ibm-q-internal',group='performance',project='paper-priority')
QSE_IP_calc  = QSE(psi           = vqe_res['vqe_circuit'],
                   operators     = IP_operators,
                   mol_info      = vqe_res,
                   instance_dict = {'instance':'hardware','provider':provider,'quantum_machine':'ibm_auckland',
                                    'error_mitigation':True,'shots':100000,'initial_layout':[0,1,2,3],'optimization':3})
QSE_IP_calc.construct_qse_matrices(outfile=outfile,resfile='qse_ip_matrices_HW.npy')
QSE_IP_calc.compute_transition_matrix_elements(transition_operators = cr_operators,
                                               outfile              = outfile,
                                               resfile              = 'qse_ip_matrices_HW.npy') # < Psi(VQE) | a*(p) a(q) | Psi(VQE) >

#------------------------------------------------------------------------------------------------------------

QSE_EA_calc  = QSE(psi           = vqe_res['vqe_circuit'],
                   operators     = EA_operators,
                   mol_info      = vqe_res,
                   instance_dict = {'instance':'hardware','provider':provider,'quantum_machine':'ibm_auckland',
                                    'error_mitigation':True,'shots':100000,'initial_layout':[0,1,2,3],'optimization':3})
QSE_EA_calc.construct_qse_matrices(outfile=outfile,resfile='qse_ea_matrices_HW.npy')
QSE_EA_calc.compute_transition_matrix_elements(transition_operators = ds_operators,
                                               outfile              = outfile,
                                               resfile              = 'qse_ea_matrices_HW.npy') # < Psi(VQE) | a(p) a*(q) | Psi(VQE) >

#------------------------------------------------------------------------------------------------------------
t1 = time.time()
print("time taken by serial code: ", t1-t0)
e_qse_ip,c_qse_ip = compute_qse_eigenpairs(vqe_res = vqe_res,
                                           qse_res = np.load('qse_ip_matrices_HW.npy',allow_pickle=True).item(),
                                           outfile = outfile, shots=30000)


beta   = 100.0
nmax   = 3000
x_mesh = [(2*n+1)*np.pi/beta for n in range(0,nmax+1)]

def ker_ip(x,y):
    return 1.0/(1j*x+y)
def ker_ea(x,y):
    return 1.0/(1j*x-y)

# Eq 6 from George's paper, lower line
g_qse_ip = compute_qse_greens_function(vqe_res = vqe_res,
                                       qse_res = np.load('qse_ip_matrices_HW.npy',allow_pickle=True).item(),
                                       x_mesh  = x_mesh,
                                       kernel  = ker_ip,   # 1/(Iw+dE)
                                       outfile = outfile, shots=30000)
# transposition, as per George's paper
print(g_qse_ip.shape)

#g_qse_ip = g_qse_ip[:,:,:,:].transpose((0,2,1,3))

# Eq 6 from George's paper, upper line
g_qse_ea = compute_qse_greens_function(vqe_res = vqe_res,
                                       qse_res = np.load('qse_ea_matrices_HW.npy',allow_pickle=True).item(),
                                       x_mesh  = x_mesh,
                                       kernel  = ker_ea,   # 1/(Iw-dE)
                                       outfile = outfile, shots=30000)

#print(g_qse_ea.shape)
#print((g_qse_ip+g_qse_ea).shape)
np.save('qse_green_function_freq.npy',g_qse_ip+g_qse_ea)
print("Starting statistical analysis:")
gf_jk_full = statistical_analysis(g_qse_ip+g_qse_ea, x_mesh)
print(gf_jk_full)
exit()
# -------------------------------------------------------------------------------------------------------

ng = g_qse_ip.shape[2]

#g_qse_ip = g_qse_ip[:,:ng//2,:ng//2,0] + g_qse_ip[:,ng//2:,ng//2:,0]
#g_qse_ea = g_qse_ea[:,:ng//2,:ng//2,0] + g_qse_ea[:,ng//2:,ng//2:,0]
print((g_qse_ip+g_qse_ea).shape)
exit()


import matplotlib.pyplot as plt
plt.plot(x_mesh,np.real(g_qse_ip[:,1,1,0]+g_qse_ea[:,1,1,0]),label='G(%d,%d)_real'%(1,1))
plt.plot(x_mesh,np.imag(g_qse_ip[:,1,1,0]+g_qse_ea[:,1,1,0]),label='G(%d,%d)_imag'%(1,1))
#plt.save(GF_00_real.png)
plt.legend()
plt.show()
exit()
# fig,axs = plt.subplots(2,2)
# plt.subplots_adjust(wspace=0.4,hspace=0.4)

# axs[0,0].set_title('Re G(IP), G(EA)')
# axs[0,1].set_title('Im G(IP), G(EA)')
# axs[1,0].set_title('Re G(tot)')
# axs[1,1].set_title('Im G(tot)')

# ng = g_qse_ip.shape[2]
# for p in range(ng):
#     for q in range(ng):
#         axs[0,0].plot(x_mesh,np.real(g_qse_ip[:,p,q,0]),label='G(IP,%d,%d)'%(p,q))
#         axs[0,1].plot(x_mesh,np.imag(g_qse_ip[:,p,q,0]),label='G(IP,%d,%d)'%(p,q))
#         axs[0,0].plot(x_mesh,np.real(g_qse_ea[:,p,q,0]),label='G(EA,%d,%d)'%(p,q))
#         axs[0,1].plot(x_mesh,np.imag(g_qse_ea[:,p,q,0]),label='G(EA,%d,%d)'%(p,q))
#         axs[1,0].plot(x_mesh,np.real(g_qse_ip[:,p,q,0]+g_qse_ea[:,p,q,0]),label='G(%d,%d)'%(p,q))
#         axs[1,1].plot(x_mesh,np.imag(g_qse_ip[:,p,q,0]+g_qse_ea[:,p,q,0]),label='G(%d,%d)'%(p,q))


print(g_qse_ip.shape, g_qse_ea.shape)
fout = open("GF_mean.txt", "w")
for w in range(g_qse_ip.shape[0]):
    for i in range(2):
        for j in range(2):
            fout.write(str(w)+"   "+str(i)+"   "+str(j)+"   "+str(g_qse_ip[w,i,j,0]+g_qse_ea[w,i,j,0])+"\n")

fout.close()

for r in range(2):
    for c in range(2):
        axs[r,c].set_xlabel('matsubara frequency [a.u.]')
        axs[r,c].set_ylabel('greens function')
        axs[r,c].legend(ncol=2)
plt.show()



