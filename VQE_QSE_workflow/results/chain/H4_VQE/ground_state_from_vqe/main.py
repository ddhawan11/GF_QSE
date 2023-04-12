import numpy as np
import sys
sys.path.append('/Users/dikshadhawan/internship_diksha/VQE_QSE_workflow/commons/')
from   subroutines import map_to_qubits,produce_variational_form,run_vqe
from   job_details import write_job_details
from   PauliExp    import build_CISD_paulis

mol_data           = np.load('h_dict.npy',allow_pickle=True).item()
outfile            = open('vqe_rep1.txt','w')
write_job_details(outfile)
molecule,operators = map_to_qubits(mol_data,mapping='jordan_wigner',two_qubit_reduction=False,tapering=False,measure_1rdm=True)


var_form           = produce_variational_form(mol_data,operators,ansatz={'type':'q_uccsd','reps':1})
results            = run_vqe(mol_data,operators,var_form,optimizer_dict={'name':'bfgs','max_iter':1000},
                             instance_dict={'instance':'statevector_simulator','shots':1024},fname_prefix='vqe_q_uccsd',outfile=outfile,
                             measure_1rdm=True)

var_form           = produce_variational_form(mol_data,operators,ansatz={'type':'efficient_su2','reps':1,'entanglement':'linear'})
results            = run_vqe(mol_data,operators,var_form,optimizer_dict={'name':'bfgs','max_iter':1000},
                             instance_dict={'instance':'statevector_simulator','shots':1024},fname_prefix='vqe_efficient_su2',outfile=outfile,
                             measure_1rdm=True)

#paulis             = build_CISD_paulis(mol_data)
#var_form           = produce_variational_form(mol_data,operators,ansatz={'type':'pauli_exp','paulis':paulis})
#results            = run_vqe(mol_data,operators,var_form,optimizer_dict={'name':'bfgs','max_iter':1000},
 #                            instance_dict={'instance':'statevector_simulator','shots':1024},fname_prefix='vqe_pauli_exp',outfile=outfile,
  #                           measure_1rdm=True)

#var_form           = produce_variational_form(mol_data,operators,ansatz={'type':'so4','entanglement':'full','reps':1})
#results            = run_vqe(mol_data,operators,var_form,optimizer_dict={'name':'cg','max_iter':1000},
#                             instance_dict={'instance':'statevector_simulator','shots':1024},fname_prefix='vqe_so4',outfile=outfile,
#                             measure_1rdm=True)

var_form           = produce_variational_form(mol_data,operators,ansatz={'type':'fabric','entanglement':'full','reps':1})
results            = run_vqe(mol_data,operators,var_form,optimizer_dict={'name':'bfgs','max_iter':1000},
                             instance_dict={'instance':'statevector_simulator','shots':1024},fname_prefix='fabric',outfile=outfile,
                             measure_1rdm=True)



