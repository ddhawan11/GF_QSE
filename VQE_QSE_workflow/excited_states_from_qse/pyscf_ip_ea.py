import pyscf
E = []
for charge,spin in zip([0,1,-1,-2],[0,1,1,0]):
    mol = pyscf.M(
        atom = 'H 0 0 0; H 0 0 0.741',
        basis = 'sto-6g',
        charge = charge,
        spin = spin)
    
    myhf = mol.ROHF().run()
    
    mycas = myhf.CASCI(mol.nao_nr(),mol.nelectron)
    mycas.fcisolver.nroots = 999
    E.append(mycas.kernel()[0])
print("E(IP,n)-E(EE,0) ",E[1]-E[0][0])
print("E(EA,n)-E(EE,0) ",E[2]-E[0][0])


