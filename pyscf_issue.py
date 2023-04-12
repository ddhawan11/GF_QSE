import numpy as np
import scipy as sp
import pickle

def with_custom_hamiltonian(mol_data):
    import pyscf
    from   pyscf import gto,scf,lo,ao2mo,cc,fci,mcscf,mp

    n     = mol_data['n']
    na,nb = mol_data['na'],mol_data['nb']
    mol               = gto.M(verbose=2)
    mol.nelectron     = na+nb
    mol.spin          = na-nb
    mol.incore_anyway = True
    mol.nao_nr        = lambda *args : n
    mol.energy_nuc    = lambda *args : mol_data['E0']
    mf            = scf.RHF(mol)
    mf.get_hcore  = lambda *args: mol_data['h1'].copy()
    mf.get_ovlp   = lambda *args: np.eye(n)
    mf._eri       = ao2mo.restore(8,mol_data['h2'].copy(),n)

    rho_0 = np.zeros((n,n))
    for i in range(na): rho_0[i,i] = 2.0
    E0 = mf.kernel(rho_0)
    if(not mf.converged):
       mf = scf.newton(mf)
       E0 = mf.kernel(mf.make_rdm1())
    print("SCF    computed/expected %.12f %.6f " % (E0,-1.125315))
    mc = mcscf.CASSCF(mf,n,(na,nb))
    Ea = mc.kernel()[0]
    print("CASSCF computed/expected %.12f %.6f " % (Ea,-1.145927))
    mc = mcscf.CASCI(mf,n,(na,nb))
    Eb = mc.kernel()[0]
    print("CASCI  computed/expected %.12f %.6f " % (Eb,-1.145927))
    h1 = np.einsum('pi,pq,qj->ij',mf.mo_coeff,mf.get_hcore(),mf.mo_coeff,optimize=True)
    h2 = np.einsum('pi,rj,qk,sl,prqs->ijkl',mf.mo_coeff,mf.mo_coeff,mf.mo_coeff,mf.mo_coeff,mol_data['h2'],optimize=True)
    Ea = fci.direct_spin1.kernel(h1.copy(),h2.copy(),n,(na,nb))[0]+mol.energy_nuc()
    mc = fci.FCI(mf)
    Ec = mc.kernel()[0]
    print("FCI    computed/expected %.12f %.6f " % (Ec,-1.145927))
    mm = mp.MP2(mf)
    Ed = mm.kernel()[0]+E0
    print("MP2    computed/expected %.12f %.6f " % (Ed,-1.138507))
    mc = cc.CCSD(mf)
    Ee = mc.kernel()[0]+E0
    print("CCSD   computed/expected %.12f %.6f " % (Ee,-1.145927))

# H2 at STO-6G level from a custom, tabulated Hamiltonian in the MO basis

n = 2
c = 0.714139
t_file = open("T.obj","rb")
v_file = open("V.obj","rb")
t = pickle.load(t_file)
v = pickle.load(v_file)
t_file.close()
v_file.close()

mol_data = {'n'  : n,
            'na' : 1,
            'nb' : 1,
            'E0' : c,
            'h1' : np.float64(t),
            'h2' : np.float64(v)}

import pyscf
import platform
print("PySCF  version ",pyscf.__version__)
print("Python version ",platform.python_version())
print("numpy  version ",np.__version__)
print("scipy  version ",sp.__version__)

with_custom_hamiltonian(mol_data)

