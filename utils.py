from qiskit                                        import *
from qiskit.aqua.operators                         import Z2Symmetries
from qiskit.aqua.algorithms                        import ExactEigensolver
from qiskit.aqua.algorithms                        import NumPyEigensolver
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.aqua                                   import aqua_globals
from qiskit.tools                                  import parallel_map
from prettytable                                   import PrettyTable
from qiskit.aqua.operators import WeightedPauliOperator,commutator

import numpy as np
import functools

def get_offsets(molecule,core,mgsr):
    dE    = core._energy_shift + core._ph_energy_shift + core._nuclear_repulsion_energy
    dA    = np.zeros(6)
    return dE,dA

def taper(molecule,core,qubit_op,A_op,outf,return_taper_sector=False,target_sector=None):

    z2_symmetries  = Z2Symmetries.find_Z2_symmetries(qubit_op)
    the_ancillas   = A_op
    nsym           = len(z2_symmetries.sq_paulis)
    the_tapered_op = qubit_op
    sqlist         = None
    z2syms         = None

    outf.write('\n\nstart tapering... \n\n')
    
    if(nsym>0):
       outf.write('Z2 symmetries found:\n')
       for symm in z2_symmetries.symmetries:
           outf.write(symm.to_label()+'\n')
       outf.write('single qubit operators found:\n')
       for sq in z2_symmetries.sq_paulis:
           outf.write(sq.to_label()+'\n')
       outf.write('cliffords found:\n')
       for clifford in z2_symmetries.cliffords:
           outf.write(clifford.print_details()+'\n')
       outf.write('single-qubit list: {}\n'.format(z2_symmetries.sq_list))
   
       tapered_ops = z2_symmetries.taper(qubit_op)
       for tapered_op in tapered_ops:
           outf.write("Number of qubits of tapered qubit operator: {}\n".format(tapered_op.num_qubits))
    
       ee = NumPyEigensolver(qubit_op, k=1)
       result = core.process_algorithm_result(ee.run())
       for line in result[0]:
           outf.write(line+'\n')
   
       if(target_sector is not None):
          smallest_idx = target_sector
          ee = NumPyEigensolver(tapered_ops[smallest_idx],k=1)
          curr_value = ee.run()['eigenvalues'][0]
          outf.write("Lowest eigenvalue of the {}-th tapered operator (computed part) is {:.12f}\n".format(smallest_idx, curr_value))
       else:
          smallest_eig_value = 99999999999999
          smallest_idx = -1
          for idx in range(len(tapered_ops)):
              ee = NumPyEigensolver(tapered_ops[idx], k=1)
              curr_value = ee.run()['eigenvalues'][0]
              if curr_value < smallest_eig_value:
                  smallest_eig_value = curr_value
                  smallest_idx = idx
              outf.write("Lowest eigenvalue of the {}-th tapered operator (computed part) is {:.12f}\n".format(idx, curr_value))   
   
       the_tapered_op = tapered_ops[smallest_idx]
       the_coeff      = tapered_ops[smallest_idx].z2_symmetries.tapering_values
       outf.write("The {}-th tapered operator matches original ground state energy, with corresponding symmetry sector of {}\n".format(smallest_idx,the_coeff))
       sqlist = the_tapered_op.z2_symmetries.sq_list
       z2syms = the_tapered_op.z2_symmetries

       the_ancillas = []
       for A in A_op:

           cmm = True
           for s in z2_symmetries.symmetries:
               cmm_s = commutator(A,WeightedPauliOperator(paulis=[[1.0,s]])).chop(1e-6)
               if(not cmm_s.is_empty()): cmm=False

           print("ancilla op commutes? ",cmm)

           if(not cmm):
              the_ancillas.append(WeightedPauliOperator(paulis=[[0.0,z2_symmetries.symmetries[0]]]))
           else:
              A_taper = z2_symmetries.taper(A)
              if(type(A_taper)==list): the_ancillas.append(A_taper[smallest_idx])
              else: the_ancillas.append(A_taper)

    outf.write('\n\n...finish tapering \n\n')

    if(return_taper_sector):
       return the_tapered_op,the_ancillas,z2syms,sqlist,smallest_idx
    else:
       return the_tapered_op,the_ancillas,z2syms,sqlist

    #   the_ancillas = []
    #   for A in A_op:
    #       A_taper = z2_symmetries.taper(A)
    #       if(type(A_taper)==list): the_ancillas.append(A_taper[smallest_idx])
    #       else: the_ancillas.append(A_taper)
    #
    #outf.write('\n\n...finish tapering \n\n')
    #
    #return the_tapered_op,the_ancillas,z2syms,sqlist 

def get_results(H_op,A_op,molecule,core,algo_result,outfile):

    mgsr = core._process_algorithm_result_ground_state(algo_result)
    dE,dA = get_offsets(molecule,core,mgsr)

    res_vqe = [algo_result['optimal_value']+dE,
               algo_result['aux_operator_eigenvalues'][0][0]+dA[0],
               algo_result['aux_operator_eigenvalues'][1][0]+dA[1],
               algo_result['aux_operator_eigenvalues'][2][0]+dA[2]]

    ee = NumPyEigensolver(operator=H_op,k=1,aux_operators=A_op).run()
    res_ee = [ee['eigenvalues'][0]+dE,
              ee['aux_operator_eigenvalues'][0][0][0]+dA[0],
              ee['aux_operator_eigenvalues'][0][1][0]+dA[1],
              ee['aux_operator_eigenvalues'][0][2][0]+dA[2]]

    t = PrettyTable(['method','Energy','N','S^2','Sz'])
    t.add_row(['VQE']+[str(round(np.real(x),6)) for x in res_vqe])
    t.add_row(['FCI']+[str(round(np.real(x),6)) for x in res_ee])
    outfile.write(str(t))
    outfile.write("\n")

    return res_vqe, res_ee

def measure_operators(operators,wfn_circuit,instance):
    circuits = []
    for idx,oper in enumerate(operators):
        circuit = oper.construct_evaluation_circuit(
                  wave_function               = wfn_circuit,
                  statevector_mode            = instance.is_statevector,
                  use_simulator_snapshot_mode = instance.is_statevector,
                  circuit_name_prefix         = 'oper_'+str(idx))
        circuits.append(circuit)
    if circuits:
        to_be_simulated_circuits = \
            functools.reduce(lambda x, y: x + y, [c for c in circuits if c is not None])
        result = instance.execute(to_be_simulated_circuits)
    # ---
    results_list = []
    for idx,oper in enumerate(operators):
        mean,std = oper.evaluate_with_result(
                   result = result,statevector_mode = instance.is_statevector,
                   use_simulator_snapshot_mode = instance.is_statevector,
                   circuit_name_prefix         = 'oper_'+str(idx))
        results_list.append([np.real(mean),np.abs(std)])
    # ---
    return results_list




'''
def print_UCCSD_parameters(molecule,core,var_form,algo_result,z2syms,sqlist,deleted_orbitals,outfile):
    sd,dd = UCCSD.compute_excitation_lists([var_form._num_alpha, var_form._num_beta],var_form._num_orbitals,
                                               None, None,
                                               same_spin_doubles=var_form.same_spin_doubles,
                                               method_singles=var_form._method_singles,
                                               method_doubles=var_form._method_doubles,
                                               excitation_type=var_form._excitation_type)

    kept_orbitals = [x for x in range(molecule.num_orbitals) if x not in deleted_orbitals]

    ed = sd+dd
    results = parallel_map(UCCSD._build_hopping_operator,ed,
                           task_args=(var_form._num_orbitals,[var_form._num_alpha, var_form._num_beta],
                                      core._qubit_mapping,core._two_qubit_reduction,
                                      z2syms),
                           num_processes=aqua_globals.num_processes)

    def convert_index(idx):
        idx_converted = []
        for i in idx:
            if(i<len(kept_orbitals)): idx_converted.append(str(kept_orbitals[i])+'u')
            else:                     idx_converted.append(str(kept_orbitals[i-len(kept_orbitals)])+'d')
        return idx_converted

    t   = PrettyTable(['excitation','amplitude'])
    lst = []
    im=0
    for m,(op,index) in enumerate(results):
        if op is not None and not op.is_empty():
           lst.append([convert_index(index),algo_result['optimal_point'][im]])
           im += 1
    for i in range(len(lst)):
        for j in range(i+1,len(lst)):
            if(np.abs(lst[j][1])>np.abs(lst[i][1])):
               lst[i],lst[j]=lst[j],lst[i]
        t.add_row([str(lst[i][0])]+[str(lst[i][1])])
    outfile.write(str(t))
    outfile.write("\n")
   
def print_circuit_requirements(c,machine,opt,layout,outfile):
    provider = IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q-internal',group='deployed',project='default')
    backend  = provider.get_backend(machine)
    from qiskit.compiler import transpile
    circuit = transpile(c,backend,initial_layout=layout,optimization_level=opt)
    outfile.write("operations %s \n" % circuit.count_ops())
    outfile.write("depth      %s \n" % circuit.depth())
    outfile.write("qubit      %s \n" % len(layout))

def build_bitstr(num_orbitals,qubit_mapping,two_qubit_reduction,qubit_tapering,sq_list,bitstr):
    half_orbitals = num_orbitals // 2
    #bitstr = np.zeros(self._num_orbitals, bool)
    #bitstr[-self._num_alpha:] = True
    #bitstr[-(half_orbitals + self._num_beta):-half_orbitals] = True
    if qubit_mapping == 'parity':
        new_bitstr = bitstr.copy()
        t_r        = np.triu(np.ones((num_orbitals,num_orbitals))) # transform to parity with upper triangular matrix
        new_bitstr = t_r.dot(new_bitstr.astype(np.int))%2
        if two_qubit_reduction:
           bitstr = np.append(new_bitstr[1:half_orbitals],new_bitstr[half_orbitals+1:]) # get rid of last alpha and beta
        else:
           bitstr = new_bitstr # don't
    else:
        assert(False)

    if qubit_tapering:
        sq_list = (len(bitstr) - 1) - np.asarray(sq_list)
        bitstr  = np.delete(bitstr, sq_list)

    return bitstr
'''

