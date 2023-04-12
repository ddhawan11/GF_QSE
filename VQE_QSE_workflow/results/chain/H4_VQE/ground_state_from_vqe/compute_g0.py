import numpy as np
from scipy import linalg as LA

def import_density(n,fname):
    ell = open(fname,'r').readlines()
    rho = np.zeros((n,n))
    m   = 0
    for p in range(n):
        for q in range(n):
            rho[p,q] = float(ell[m].split()[2])
            m += 1
    return rho

def G(omega,mu=0.0):
    return LA.inv((1j*omega+mu)*S-F)

def get_Gnot(mol_data,rho,x_mesh):
    n        = mol_data['n']
    S        = np.eye(n)
    h1       = mol_data['h1']
    h2       = mol_data['h2']
    F        = h1+(2*np.einsum('ikjl,jl->ik',h2,rho)-np.einsum('iljk,jl->ik',h2,rho))
    Gtens = np.zeros((len(x_mesh),n,n),dtype=complex)
    for ix,x in enumerate(x_mesh):
        Gtens[ix,:,:] = G(x,0.0)
    return Gtens

mol_data = np.load('../generate_hamiltonian/h_dict.npy',allow_pickle=True).item()
rho      = import_density(n,'vqe_q_uccsd_1rdm.txt')/2.0
 
beta   = 100.0
nmax   = 100
x_mesh = [(2*n+1)*np.pi/beta for n in range(0,nmax+1)]

Gtens = get_Gnot(mol_data,rho,x_mesh)

import matplotlib.pyplot as plt
fig,axs = plt.subplots(1,2)
plt.subplots_adjust(wspace=0.4,hspace=0.4)

axs[0].set_title('Re G')
axs[1].set_title('Im G')

Gqse = np.load('../excited_states_from_qse/qse_green_function_freq.npy')

ng = Gtens.shape[2]
for p in range(ng):
    for q in range(ng):
        axs[0].plot(x_mesh,np.real(Gtens[:,p,q]),label='G0(%d,%d)'%(p,q))
        axs[1].plot(x_mesh,np.imag(Gtens[:,p,q]),label='G0(%d,%d)'%(p,q))
        axs[0].plot(x_mesh,np.real(Gqse[:,p,q]), label='G(qse,%d,%d)'%(p,q))
        axs[1].plot(x_mesh,np.imag(Gqse[:,p,q]), label='G(qse,%d,%d)'%(p,q))

for c in range(2):
    axs[c].set_xlabel('matsubara frequency [a.u.]')
    axs[c].set_ylabel('greens function')
    axs[c].legend(ncol=2)
plt.show()

