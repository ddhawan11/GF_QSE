import numpy as np
import pyscf
from   pyscf import gto,scf,lo,ao2mo,cc,fci,mcscf
from   scipy import linalg as LA


def do_scf(mol_data,rho_0=None):
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
    #print(n,na,nb,mf.get_hcore(),mf.get_ovlp(),mf._eri,mol_data['E0'])
    #exit()
    if(rho_0 is None):
       rho_0 = np.zeros((n,n))
       for i in range(na): rho_0[i,i] = 2.0
    E0 = mf.kernel(rho_0)
    if(not mf.converged):
       mf = scf.newton(mf)
       E0 = mf.kernel(mf.make_rdm1())
    h1 = np.einsum('pi,pq,qj->ij',mf.mo_coeff,mf.get_hcore(),mf.mo_coeff)
    h2 = np.einsum('pi,rj,qk,sl,prqs->ijkl',mf.mo_coeff,mf.mo_coeff,mf.mo_coeff,mf.mo_coeff,mol_data['h2'])
    print(mol.energy_nuc())
    Ea = fci.direct_spin1.kernel(h1.copy(),h2.copy(),n,(na,nb))[0]+mol.energy_nuc()
    mc = mcscf.CASSCF(mf,2,2)
    Eb = mc.kernel()[0]
    mc = fci.FCI(mf)
    Ec = mc.kernel()[0]
    print("SCF       ",E0)
    print("SCF, rho  ",mf.make_rdm1())
    print("FCI       ",Ea,Eb,Ec)
    mc = cc.CCSD(mf)
    Eccsd = mc.kernel()[0]
    exit()
    return E0,E


