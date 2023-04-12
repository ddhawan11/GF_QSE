def make_qmolecule_from_data(mol_data):
    from qiskit.chemistry import QMolecule
    import numpy as np
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

def Identity(n):
    from qiskit.aqua.operators import WeightedPauliOperator
    from qiskit.quantum_info import Pauli
    import numpy as np
    zeros = [0]*n
    zmask = [0]*n
    a_x = np.asarray(zmask,dtype=np.bool)
    a_z = np.asarray(zeros,dtype=np.bool)
    return WeightedPauliOperator([(1.0,Pauli(a_x,a_z))])

def rdm_operators(n,nelec,map_type,tqr):
    import numpy as np
    from qiskit.chemistry.fermionic_operator import FermionicOperator
    from qiskit.aqua.operators import Z2Symmetries
    ob_operators = []
    for p in range(n//2):
        for q in range(n//2):
            h_1                = np.zeros((n,n))
            h_1[p,q]           = 1.0
            h_1[p+n//2,q+n//2] = 1.0
            o_h_1    = FermionicOperator(h1=h_1).mapping(map_type=map_type)
            if(map_type=='parity' and tqr):
               o_h_1 = Z2Symmetries.two_qubit_reduction(X,nelec)
            ob_operators.append(o_h_1)
    return ob_operators

def map_to_qubits(mol_data={},mapping='jordan_wigner',two_qubit_reduction=False,tapering=True,measure_1rdm=False):
    import sys
    sys.path.append('../commons/')

    from qiskit.chemistry.core import Hamiltonian,TransformationType,QubitMappingType
    from tapering import taper_principal_operator,taper_auxiliary_operators

    if(mapping=='jordan_wigner'): qubit_mapping = QubitMappingType.JORDAN_WIGNER
    elif(mapping=='parity'):      qubit_mapping = QubitMappingType.PARITY
    else:                         assert(False)

    molecule  = make_qmolecule_from_data(mol_data)
    core      = Hamiltonian(transformation      = TransformationType.FULL,
                            qubit_mapping       = qubit_mapping,
                            two_qubit_reduction = two_qubit_reduction,
                            freeze_core         = False,
                            orbital_reduction   = [])
    H_op,A_op = core.run(molecule)
    dE        = core._energy_shift + core._ph_energy_shift + core._nuclear_repulsion_energy
    H_op      = H_op + dE * Identity(H_op.num_qubits)
    A_op      = A_op[:3]
    if(measure_1rdm):
       A_op += rdm_operators(core._molecule_info['num_orbitals'],
                             core._molecule_info['num_particles'],
                             core._qubit_mapping,
                             two_qubit_reduction)

    operators = {'orbitals'  : core._molecule_info['num_orbitals'],
                 'particles' : core._molecule_info['num_particles'],
                 'mapping'   : core._qubit_mapping,
                 '2qr'       : two_qubit_reduction,
                 'tapering'  : tapering}
    if(tapering):
       H_op_tapered,z2syms,sqlist,target_sector = taper_principal_operator(H_op,target_sector=None)
       A_op_tapered                             = taper_auxiliary_operators(A_op,z2syms,target_sector)
       operators['h_op']           = H_op_tapered
       operators['a_op']           = A_op_tapered
       operators['names']          = ['number','spin-2','spin-z']
       operators['target_sector']  = target_sector
       operators['tapering_info']  = [z2syms,sqlist]
       operators['untapered_h_op'] = H_op
    else:
       operators['h_op']           = H_op
       operators['a_op']           = A_op
       operators['names']          = ['number','spin-2','spin-z']
       operators['target_sector']  = None
       operators['tapering_info']  = [None,None]

    return molecule,operators

def produce_variational_form(molecule,operators,ansatz={}):
    import sys
    sys.path.append('../commons/')
    from change_of_basis                            import construct_cob_circuit
    from qiskit.chemistry.components.initial_states import HartreeFock

    initial_state = HartreeFock(num_orbitals  = operators['orbitals'],
                                qubit_mapping = operators['mapping'],
                                num_particles = operators['particles'],
                                sq_list       = operators['tapering_info'][1])
    final_state   = construct_cob_circuit(molecule,operators)
    if(ansatz['type']=='q_uccsd'):
       from local_uccsd import UCCSD
       var_form = UCCSD(num_orbitals        = operators['orbitals'],
                        num_particles       = operators['particles'],
                        active_occupied     = None,
                        active_unoccupied   = None,
                        initial_state       = initial_state,
                        qubit_mapping       = operators['mapping'],
                        two_qubit_reduction = operators['2qr'],
                        num_time_slices     = ansatz['reps'],
                        z2_symmetries       = operators['tapering_info'][0],
                        target_sector       = operators['target_sector'],
                        qf                  = final_state)
    elif(ansatz['type']=='efficient_su2'):
       from CustomEfficientSU2 import CustomEfficientSU2 
       var_form = CustomEfficientSU2(num_qubits    = operators['h_op'].num_qubits,
                                     reps          = ansatz['reps'],
                                     entanglement  = ansatz['entanglement'],
                                     initial_state = initial_state.construct_circuit(),
                                     chf           = final_state)
    elif(ansatz['type']=='pauli_exp'):
       from PauliExp import PauliExp
       var_form  = PauliExp(num_qubits    = operators['h_op'].num_qubits,
                            initial_state = initial_state.construct_circuit(),
                            paulis        = ansatz['paulis'],
                            chf           = final_state)
    elif(ansatz['type']=='so4'):
       from SO4 import var_form_unitary
       var_form = var_form_unitary(nqubit        = initial_state.construct_circuit().num_qubits,
                                   depth         = ansatz['reps'],
                                   initial_state = initial_state.construct_circuit(),
                                   entanglement  = ansatz['entanglement'],
                                   chf           = final_state)
    elif(ansatz['type']=='fabric'):
       from Fabric import Fabric
       var_form = Fabric(num_orbitals = initial_state.construct_circuit().num_qubits//2,
                         depth         = ansatz['reps'],
                         initial_state = initial_state.construct_circuit(),
                         entanglement  = ansatz['entanglement'],
                         chf           = final_state)
    else:
       assert(False)
    return var_form

def run_vqe(mol_data,operators,var_form,optimizer_dict,instance_dict,fname_prefix=None,outfile=None,measure_1rdm=False):
    import sys
    sys.path.append('../commons/')
    from qiskit.aqua.components.optimizers import L_BFGS_B,CG,SPSA,COBYLA,ADAM
    from qiskit.aqua.algorithms            import VQE
    from qiskit                            import Aer
    from qiskit.aqua                       import QuantumInstance
    from harvest                           import measure_operators
    from qiskit.aqua.algorithms            import NumPyEigensolver
    from prettytable                       import PrettyTable
    import os
    import numpy as np

    if(optimizer_dict['name']=='cg'):
       optimizer = CG(maxiter=optimizer_dict['max_iter'])
    elif(optimizer_dict['name']=='adam'):
       optimizer = ADAM(maxiter=optimizer_dict['max_iter'])
    elif(optimizer_dict['name']=='spsa'):
       optimizer = SPSA(maxiter=1000)
    elif(optimizer_dict['name']=='bfgs'):
       optimizer = L_BFGS_B(maxiter=optimizer_dict['max_iter'],iprint=1001)
    else:
       assert(False)

    backend          = Aer.get_backend(instance_dict['instance'])
    quantum_instance = QuantumInstance(backend=backend,shots=instance_dict['shots'])

    if(os.path.isfile(fname_prefix+'_input_parameters.txt')):
       p0 = np.loadtxt(fname_prefix+'_input_parameters.txt')
    else:
       p0 = None
    algo             = VQE(operators['h_op'],var_form,optimizer,aux_operators=operators['a_op'],include_custom=True,initial_point=p0)
    algo_result      = algo.run(quantum_instance)

    p1      = algo._ret['opt_params']
    res_vqe = measure_operators([operators['h_op']]+operators['a_op'],var_form.construct_circuit(p1),quantum_instance)

    ee      = NumPyEigensolver(operator=operators['h_op'],k=1,aux_operators=operators['a_op']).run()
    res_ee  = [ee['eigenvalues'][0]]+[ee['aux_operator_eigenvalues'][0][i][0] for i in range(len(operators['a_op']))]

    n_aux   = 3
    t = PrettyTable(['method','energy']+operators['names'])
    t.add_row(['VQE']+[str(round(np.real(x),8))+' +/- '+str(round(np.real(y),6)) for x,y in res_vqe[:n_aux+1]])
    t.add_row(['FCI']+[str(round(np.real(x),8)) for x in res_ee[:n_aux+1]])
    outfile.write("\nVQE %s\n" % fname_prefix)
    outfile.write(str(t))
    outfile.write("\n")
    if(measure_1rdm):
       from scipy import linalg as LA
       f = open(fname_prefix+"_1rdm.txt",'w')
       n   = mol_data['n'] 
       rho = np.zeros((n,n,2))
       m   = 0
       for p in range(n):
           for q in range(n):
               rho[p,q,:] = res_vqe[n_aux+1+m]
               m         += 1
               f.write('%d %d %f +/- %f \n' % (p,q,rho[p,q,0],rho[p,q,1]))
       f.write('trace (sum diag) %f \n' % np.einsum('pp',rho[:,:,0]))
       nu,U = LA.eigh(rho[:,:,0])
       f.write('trace (sum eigs) %f \n'   % np.sum(nu))
       f.write('occupation numbers %s \n' % str(nu)) 

    np.savetxt(fname_prefix+"_output_parameters.txt",p1)

    np.save(fname_prefix+'_results.npy',{'mol_data'            : mol_data,
                                         'operators'           : operators,
                                         'num_parameters'      : var_form.num_parameters,
                                         'results_vqe'         : res_vqe,
                                         'vqe_circuit'         : var_form.construct_circuit(p1),
                                         'results_eigensolver' : res_ee},allow_pickle=True)

