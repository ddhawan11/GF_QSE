import numpy as np
from qiskit.chemistry import QMolecule

def make_qmolecule_from_data(mol_data):
    n,na,nb,h0,h1,h2 = mol_data['n'],mol_data['na'],mol_data['nb'],mol_data['E0'],mol_data['h1'],mol_data['h2']
    m = QMolecule()
    m.nuclear_repulsion_energy = h0
    m.num_orbitals             = n
    m.num_alpha                = na
    m.num_beta                 = nb
    m.mo_coeff                 = np.eye(n)
    m.mo_onee_ints             = h1
    m.mo_eri_ints              = h2
    return m

def transpilation(c,machine,opt,layout):
    from qiskit import IBMQ
    provider = IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q-internal',group='deployed',project='default')
    backend  = provider.get_backend(machine)
    from qiskit.compiler import transpile
    transpiled_circuit = transpile(c,backend,initial_layout=layout,optimization_level=opt)
    return transpiled_circuit


