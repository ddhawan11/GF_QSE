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

def get_Gnot(mol_data,rho,x_mesh):
    n        = mol_data['n']
    S        = np.eye(n)
    h1       = mol_data['h1']
    h2       = mol_data['h2']
    F        = h1+(2*np.einsum('ikjl,jl->ik',h2,rho)-np.einsum('iljk,jl->ik',h2,rho))

    def G(omega,mu=0.0):
        return LA.inv((1j*omega+mu)*S-F)

    Gtens = np.zeros((len(x_mesh),n,n),dtype=complex)
    for ix,x in enumerate(x_mesh):
        Gtens[ix,:,:] = G(x,0.0)
    return Gtens

