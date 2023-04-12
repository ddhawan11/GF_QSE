import numpy as np
from   pyscf import gto,scf,ao2mo
from   scipy import linalg as LA
import pickle
mol   = gto.M(verbose=4,atom=[['H',(0,0,0)],['H',(0,0,1.0)],['H',(0,0.0,2.0)],['H',(0,0.0,3.0)]],basis='sto-6g')
mf    = scf.RHF(mol)
E     = mf.kernel()

# AO hamiltonian

E0  = mol.energy_nuc()
s1  = mf.get_ovlp()
h1  = mf.get_hcore()
h2  = ao2mo.restore(1,mf._eri,mol.nao_nr())

basis = 'sao'
print("Nuclear energy", E0)
if(basis=='sao'):
   sgm,U = LA.eigh(s1)
   C     = np.einsum('ij,j->ij',U,1.0/np.sqrt(sgm))
   C     = np.dot(C,U.T)
   if(LA.det(mf.mo_coeff)<0):
      c0,c1 = C[:,0].copy(),C[:,1].copy()
      C[:,0],C[:,1] = c1,c0
if(basis=='mo'):
   C     = mf.mo_coeff

# orthonormal basis hamiltonian

s1  = np.einsum('pi,pq,qj->ij',C,s1,C)
h1  = np.einsum('pi,pq,qj->ij',C,h1,C)
h2  = np.einsum('pi,rj,qk,sl,prqs->ijkl',C,C,C,C,h2)

CHF = mf.mo_coeff
VHF = np.einsum('am,la->lm',CHF,LA.inv(C))

t_file = open("T.obj",'rb')
h1 = pickle.load(t_file)
t_file.close()
v_file = open("V.obj",'rb')
h2 = pickle.load(v_file)
v_file.close()
print(h1,h2)


mol_data = {'n'  : C.shape[1],
            'na' : mol.nelectron//2,
            'nb' : mol.nelectron//2,
            'E0' : E0,
            'h1' : h1,
            'h2' : h2,
            'V'  : VHF}

np.save('h_dict.npy',mol_data,allow_pickle=True)

