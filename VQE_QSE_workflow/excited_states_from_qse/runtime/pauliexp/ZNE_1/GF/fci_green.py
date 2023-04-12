import numpy as np
import itertools
import sys
sys.path.append('../commons/')

from   job_details  import write_job_details
from   subroutines  import build_qse_operators
from   post_process import compute_qse_eigenpairs,compute_qse_greens_function
from   QSE          import QSE
from   green        import eigenpairs,to_matrix

outfile  = open('qse.txt','w')
write_job_details(outfile)
vqe_res  = np.load('vqe_q_uccsd_results.npy',allow_pickle=True).item()

H    = vqe_res['operators']['h_op']
E,V  = eigenpairs(H)                          # <======
neig = V.shape[1]
n    = V.shape[0]

cr_operators = build_qse_operators(class_of_operators='ea',dressed=False,spin='both',mol_info=vqe_res) # creation    operators
ds_operators = build_qse_operators(class_of_operators='ip',dressed=False,spin='both',mol_info=vqe_res) # destruction operators

#for c in cr_operators:
#    print(c.print_details())
#exit()

n_op   = len(cr_operators)
cr_mat = np.zeros((n,n,n_op),dtype=complex)
ds_mat = np.zeros((n,n,n_op),dtype=complex)

for i in range(n_op):
    cr_mat[:,:,i] = to_matrix(cr_operators[i])
    ds_mat[:,:,i] = to_matrix(ds_operators[i])

cr_trans = np.einsum('pi,pqx,q->xi',V,cr_mat,V[:,0])
ds_trans = np.einsum('pi,pqx,q->xi',V,ds_mat,V[:,0])

print("<abcd|")
print("a = left  orbital, spin-up")
print("b = right orbital, spin-up")
print("c = left  orbital, spin-down")
print("d = right orbital, spin-down")

for i in range(neig):
    print("eigenstate %d, energy = %.6f " % (i,E[i]))
    nup,ndn = 0,0
    for mu,idx in enumerate(itertools.product([0,1],repeat=H.num_qubits)):
        if(np.abs(V[mu,i])>1e-4):
           print("<%s|Psi(%d)> = %s " % (''.join([str(x) for x in idx[::-1]]),i,str(V[mu,i])))
           nup += sum(idx[::-1][:len(idx)//2])*np.abs(V[mu,i])**2
           ndn += sum(idx[::-1][len(idx)//2:])*np.abs(V[mu,i])**2
    print("N(up),N(down) = %.1f %.1f" % (nup,ndn))
    for p in range(n_op):
        print("<Psi(%d)| c(%d)|Psi0> = %s " % (i,p,str(ds_trans[p,i])))
        print("<Psi(%d)|c*(%d)|Psi0> = %s " % (i,p,str(cr_trans[p,i])))
    print("-------------")



beta   = 100.0
nmax   = 100
x_mesh = [(2*n+1)*np.pi/beta for n in range(0,nmax+1)]
nx     = len(x_mesh)

ker_ea = np.zeros((nx,neig),dtype=complex)
ker_ip = np.zeros((nx,neig),dtype=complex)

for j in range(nx):
    for m in range(neig):
        ker_ea[j,m] = 1.0/(1j*x_mesh[j]+E[0]-E[m])
        ker_ip[j,m] = 1.0/(1j*x_mesh[j]+E[m]-E[0])
        print("E0,Em,Em-E0 ",E[0],E[m],E[m]-E[0])
    print("-----")

g_qse_ea = np.einsum('am,xm,bm->xab',np.conj(cr_trans),ker_ea,cr_trans)
g_qse_ip = np.einsum('bm,xm,am->xab',np.conj(ds_trans),ker_ip,ds_trans)

norb     = g_qse_ea.shape[2]//2
#print(np.abs(g_qse_ea[:,:norb,norb:]).max(),np.abs(g_qse_ea[:,norb:,:norb]).max())
#print(np.abs(g_qse_ip[:,:norb,norb:]).max(),np.abs(g_qse_ip[:,norb:,:norb]).max())
g_qse_ea = (g_qse_ea[:,:norb,:norb] + g_qse_ea[:,norb:,norb:])/2.0
g_qse_ip = (g_qse_ip[:,:norb,:norb] + g_qse_ip[:,norb:,norb:])/2.0
#exit()

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

np.save('qse_green_function_freq_fci.npy',g_qse_ip+g_qse_ea)


