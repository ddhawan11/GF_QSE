import numpy as np
import pyscf
from pyscf import lo
from scipy import linalg as LA

def gs(A,S):
    n = A.shape[1]
    for j in range(n):
        for k in range(j):
            A[:,j] -= np.dot(A[:,k],np.dot(S,A[:,j]))*A[:,k]
        A[:,j] = A[:,j]/np.sqrt(np.dot(A[:,j],np.dot(S,A[:,j])))
    return A

# ----

mol = pyscf.M(
    atom     = 'H 0 0 0; Li 0 0 1.593',
    basis    = '6-31g',
    symmetry = 'Coov')

mf  = mol.RHF().run()
n   = mol.nao_nr()
S   = mf.get_ovlp()

MO  = mf.mo_coeff
AO  = np.eye(n)
# for lack of fantasy, initialize the SAO as the AO
SAO = np.eye(n)

# force the first SAO to be the lowest-energy (core 1s) 
SAO[:,0] = MO[:,0]
# ortonormalize with Gram-Schmidt
SAO = gs(SAO,S)
print(SAO[:,0]-MO[:,0])
# relocalize (optional)
SAO[:,1:] = lo.Boys(mol,SAO[:,1:]).kernel()

print("orthonormality of SAOs        ",np.abs(np.einsum('px,pq,qy->xy',SAO,S,SAO)-np.eye(n)).max())
print("overlaps between SAOs and AOs ")
M = np.einsum('px,pq,qy->xy',SAO,S,AO)
for x in range(n):
    y = np.argmax(np.abs(M[x,:]))
    print("max overlap with ",mol.ao_labels()[y],M[x,y])

print("overlaps between SAOs and MOs ")
M = np.einsum('px,pq,qy->xy',SAO,S,MO)
for x in range(n):
    y = np.argmax(np.abs(M[x,:]))
    print("max overlap with MO number ",y,M[x,y])

# matrix expanding the MOs in the SAO basis
C = np.dot(LA.inv(SAO),MO)
print("C[0,0]  ",C[0,0])
print("C[0,1:] ",np.abs(C[0,1:]).max())
print("C[1:,0] ",np.abs(C[1:,0]).max())
print("C[1:,1:] unitary ",LA.det(C[1:,1:]))
print(C)


