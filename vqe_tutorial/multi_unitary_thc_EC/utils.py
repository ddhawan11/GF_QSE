import numpy as np
import functools
from   qiskit.aqua.operators import WeightedPauliOperator
from   qiskit.quantum_info   import Pauli
from   qiskit.result         import postprocess,models
from   qiskit                import *

def safe_eigh(h, s, lindep=1e-15):
    from scipy     import linalg as LA
    from functools import reduce
    seig,t = LA.eigh(s)
    if seig[0] < lindep:
        idx = seig >= lindep
        t = t[:,idx] * (1/np.sqrt(seig[idx]))
        heff = reduce(np.dot, (t.T.conj(), h, t))
        w, v = LA.eigh(heff)
        v = np.dot(t, v)
    else:
        w, v = LA.eigh(h, s)
    return w, v, seig

def find_first_sublist(seq, sublist, start=0):
    length = len(sublist)
    for index in range(start, len(seq)):
        if seq[index:index+length] == sublist:
            return index, index+length

def replace_sublist(seq, sublist, replacement):
    length = len(replacement)
    index = 0
    for start, end in iter(lambda: find_first_sublist(seq, sublist, index), None):
        seq[start:end] = replacement
        index = start + length

def find_pauli_domain(p):
    str_p  = str(p)
    dom_p  = [(p.num_qubits-1)-i for i in range(len(str_p)) if str_p[i]!='I']
    return len(dom_p),dom_p

def apply_z_exp(qc,dom,angle,ctrl,i_ctrl=0):
    n = len(dom)
    for i in range(n-1): qc.cx(dom[i],dom[i+1])
    if(ctrl): qc.crz(angle,i_ctrl,dom[n-1])
    else:     qc.rz(angle,dom[n-1])
    for i in range(n-1)[::-1]: qc.cx(dom[i],dom[i+1])

def find(ch,str):
    return [len(str)-1-i for i, c in enumerate(str) if c == ch]

def nz_set(wpo):
    n  = wpo.num_qubits
    cz = [np.real(c) for c,p in wpo.paulis]
    iz = [find('Z',str(p)) for c,p in wpo.paulis]
    nz = [len(i) for i in iz]
    df = 0.0
    if(max(nz)==0):              # DEBUG: case when the echo operator is prop to the identity
       v = []
       for m,l in enumerate(iz):
           df += cz[m]
    if(max(nz)==1):
       v = np.zeros(n)
       for m,l in enumerate(iz):
           if(len(l)>0): v[l[0]]=cz[m]
           else:         df += cz[m]
    if(max(nz)==2):
       v = np.zeros((n,n))
       for m,l in enumerate(iz): 
           if(len(l)>0): v[l[1],l[0]]=cz[m]
           else:         df += cz[m]
    return iz,nz,v,df

def diagonal_operator(circuit,wpo,controlled):
    #print("DIAG OP DEBUG -- circuit")
    #print(circuit.draw())
    #print("DIAG OP DEBUG -- wpo")
    #print(wpo.print_details())
    dphi        = 0.0
    iz,nz,cz,df = nz_set(wpo)
    if(wpo.num_qubits==4 and max(nz)==1 and controlled):
       dphi += df
       apply_z_exp(circuit,[1],2*cz[0],controlled,i_ctrl=0)
       circuit.swap(0,1)
       apply_z_exp(circuit,[2],2*cz[1],controlled,i_ctrl=1)
       circuit.swap(1,2)
       apply_z_exp(circuit,[3],2*cz[2],controlled,i_ctrl=2)
       circuit.swap(3,4)
       apply_z_exp(circuit,[3],2*cz[3],controlled,i_ctrl=2)
       circuit.swap(3,4)
       circuit.swap(1,2)
       circuit.swap(0,1)
    elif(wpo.num_qubits==4 and max(nz)==2 and controlled==False):
       dphi += df
       apply_z_exp(circuit,[1,2],2*cz[0,1],controlled)
       apply_z_exp(circuit,[3,4],2*cz[2,3],controlled)
       apply_z_exp(circuit,[2,3],2*cz[1,2],controlled)
       circuit.swap(1,2)
       circuit.swap(3,4)
       circuit.swap(2,3)
       apply_z_exp(circuit,[2,3],2*cz[0,3],controlled)
       apply_z_exp(circuit,[1,2],2*cz[1,3],controlled)
       apply_z_exp(circuit,[3,4],2*cz[0,2],controlled)
       circuit.swap(2,3)
       circuit.swap(3,4)
       circuit.swap(1,2)
    elif(wpo.num_qubits==4 and max(nz)==2 and controlled):
       apply_z_exp(circuit,[2,1],2*cz[0,1],controlled,i_ctrl=0)
       circuit.swap(2,3)
       apply_z_exp(circuit,[2,1],2*cz[0,2],controlled,i_ctrl=0)
       circuit.swap(2,3)
       circuit.swap(0,1)
       apply_z_exp(circuit,[3,2],2*cz[1,2],controlled,i_ctrl=1)
       circuit.swap(3,4)
       apply_z_exp(circuit,[3,2],2*cz[1,3],controlled,i_ctrl=1)
       circuit.swap(1,2)
       apply_z_exp(circuit,[4,3],2*cz[2,3],controlled,i_ctrl=2)
       circuit.swap(1,2)
       circuit.swap(0,1)
       circuit.swap(2,3)
       apply_z_exp(circuit,[2,1],2*cz[0,3],controlled,i_ctrl=0)
       circuit.swap(2,3)
       circuit.swap(3,4)
    else:
       for c,p in wpo.paulis:
           phi_p = np.real(c)
           n_dom_p,dom_p = find_pauli_domain(p)
           if(n_dom_p==0): dphi += np.real(phi_p)
           else:
               apply_z_exp(circuit,[1+i for i in dom_p],2*phi_p,controlled)
    return dphi

def apply_compressed_circuit(wpo,qc,domain,offset):

    from scipy                        import linalg as LA
    from qiskit.aqua.operators.legacy import op_converter

    from qiskit.quantum_info.synthesis.one_qubit_decompose import OneQubitEulerDecomposer
    from qiskit.quantum_info.synthesis.two_qubit_decompose import TwoQubitBasisDecomposer
    from qiskit.circuit.library.standard_gates.x           import CXGate

    wpo_reduced = WeightedPauliOperator([[c,Pauli(np.take(p.z,domain),np.take(p.x,domain))] for c,p in wpo.paulis])
    mat = LA.expm(op_converter.to_matrix_operator(wpo_reduced)._matrix.todense())
    if(len(domain)==1): C = OneQubitEulerDecomposer().__call__(mat)
    if(len(domain)==2): C = TwoQubitBasisDecomposer(CXGate()).__call__(mat)
    for g in C:
        instruction,q1,q2 = g
        if(instruction.name=='u3'):
           t1,t2,t3 = instruction.params
           qc.u3(t1,t2,t3,domain[q1[0].index]+offset)
        if(instruction.name=='cx'):
           qc.cx(domain[q1[0].index]+offset,domain[q1[1].index]+offset)

def givens(circuit,givens_list):
    dphi = 0.0
    for g in givens_list:
        dom_p = [set(find_pauli_domain(p)[1]) for c,p in g.paulis]
        dom_g = list(set().union(*dom_p))
        if(len(dom_g)==0): dphi += g.paulis[0][0]
        if(len(dom_g)==1): apply_compressed_circuit(g,circuit,dom_g,offset=1)
        if(len(dom_g)==2): apply_compressed_circuit(g,circuit,dom_g,offset=1)
        if(len(dom_g)>=3):
           for c,p in g.paulis:
               g_p    = WeightedPauliOperator([[c,p]])
               dom_gp = find_pauli_domain(p)[1]
               assert(len(dom_gp)<=2)
               apply_compressed_circuit(g_p,circuit,dom_gp,offset=1)
    return dphi

def single_step(psi_ij,H,D,M,dictionary,controlled=True):
    relative_phase = 0.0
    # ----- if an echo sequence is present, count the number of pulses and sample phases
    idx_echo    = len([x for x in D.circuit_instruction if 'echo' in x])//2
    echo_phases = 2*np.pi*np.random.random(idx_echo)
    for instruction in D.circuit_instruction:
        # ----- echo
        if('echo'   in instruction):
           idx_echo = int(instruction.split('_')[1])
           sgn_echo = 1
           if('inv' in instruction): sgn_echo = -1
           #psi_ij.barrier()
           #print("ECHO DEBUG -- particle number operator ",M.observables['N'])
           diagonal_operator(psi_ij,sgn_echo*echo_phases[idx_echo]*M.observables['N'],controlled=False)
           psi_ij.rz(sgn_echo*echo_phases[idx_echo],0)
           #psi_ij.barrier()
        # ----- givens
        if('<-'     in instruction):
           givens(psi_ij,M.givens[instruction])
        # ----- one-body
        if('evolve_t' in instruction):
           #psi_ij.barrier()
           #print("EVOLUTION DEBUG ")
           d_phi = diagonal_operator(psi_ij,-dictionary['time_step']*M.hamiltonian_terms['t1'],controlled=controlled)
           if(controlled): relative_phase += d_phi
           #psi_ij.barrier()
        # ----- two-body
        if('evolve_u' in instruction):
           idx_u = int(instruction.split('_')[1][1:])
           #psi_ij.barrier()
           d_phi = diagonal_operator(psi_ij,-dictionary['time_step']*M.hamiltonian_terms['u%d'%idx_u],controlled=controlled)
           if(controlled): relative_phase += d_phi
           #psi_ij.barrier()
    return relative_phase

def time_evolution_circuit(psi_0,i_bra,i_ket,k,H,D,M,dictionary):
    assert(i_bra<=i_ket)
    phase  = 0.0
    psi_ij = psi_0.copy()
    for mu in range(i_bra):
        single_step(psi_ij,H,D,M,dictionary,controlled=False)
    for mu in range(i_ket-i_bra):
        phase += single_step(psi_ij,H,D,M,dictionary,controlled=True)
    return [('t_bra_%d|t_ket_%d|echo_%d'%(i_bra,i_ket,k),psi_ij.copy(),phase)]

# ------------------------------------------------------------

def measure_operators(operator_dict,circuit_dict,instance,post_select=False,rf=(None,None),particle_number=0,ps_fun='none',ps_mapping='jordan_wigner',ps_tqr=False):
    if(post_select): return measure_operators_with_post_selection(operator_dict,circuit_dict,instance,rf,particle_number,ps_fun,ps_mapping,ps_tqr)
    else:            return measure_operators_without_post_selection(operator_dict,circuit_dict,instance,rf)

def measure_operators_without_post_selection(operator_dict,circuit_dict,instance,rf):
    circuits = []
    for ck in circuit_dict.keys():
        circ_k = circuit_dict[ck]
        for ok in operator_dict.keys():
            oper_k = operator_dict[ok]
            if(not oper_k.is_empty()):
               circuit = oper_k.construct_evaluation_circuit(
                         wave_function               = circ_k,
                         statevector_mode            = instance.is_statevector,
                         use_simulator_snapshot_mode = instance.is_statevector,
                         circuit_name_prefix         = 'oper_'+ok+'_wfn_'+ck)
               circuits.append(circuit)
    if circuits:
        to_be_simulated_circuits = \
            functools.reduce(lambda x, y: x + y, [c for c in circuits if c is not None])
        fname,operation = rf
        import pickle
        if(operation=='read'):
           with open(fname,'rb') as input:
                result = pickle.load(input)

           nr = len(result.results)
           for ir in range(nr):
               print("DICTIONARY BEING READ..... ",result.results[ir].header.name)

        if(operation=='write'):
           result = instance.execute(to_be_simulated_circuits)
           with open(fname,'wb') as output:
                pickle.dump(result,output,pickle.HIGHEST_PROTOCOL)
    # -----
    res = {}
    for ck in circuit_dict.keys():
        for ok in operator_dict.keys():
            oper_k = operator_dict[ok]
            if(not oper_k.is_empty()):
               mean,std = oper_k.evaluate_with_result(
                          result                      = result,
                          statevector_mode            = instance.is_statevector,
                          use_simulator_snapshot_mode = instance.is_statevector,
                          circuit_name_prefix         = 'oper_'+ok+'_wfn_'+ck)
               mean,std = np.real(mean),np.abs(std)
               res[ok+'|'+ck] = (mean,std)
            else:
               res[ok+'|'+ck] = (0,0)
    return res

def measure_operators_with_post_selection(operator_dict,circuit_dict,instance,rf,particle_number,ps_fun,ps_mapping,ps_tqr):
    def get_circuit_oper(ck):
        ck_list = ck.split('|')
        oper_ck = ck_list[len(ck_list)-1].split('<-')[0].split('_')[1]
        ck_list = ck_list[:len(ck_list)-1]
        return '|'.join(ck_list),oper_ck
    # -----
    res_ovlp = {}
    circuits = []
    for ck in circuit_dict.keys():
        ck_list,oper_ck = get_circuit_oper(ck)
        for oj in ['re_S','im_S']:
            ckoj = '%s|%s' % (oj,ck_list)
            res_ovlp[ckoj] = [0,0]
        for oj in ['re_%s'%oper_ck,'im_%s'%oper_ck]:
            ckoj = oj+'|'+ck_list
            if(not operator_dict[oj].is_empty()):
               circuit = operator_dict[oj].construct_evaluation_circuit(
                         wave_function               = circuit_dict[ck],
                         statevector_mode            = instance.is_statevector,
                         use_simulator_snapshot_mode = instance.is_statevector,
                         circuit_name_prefix         = ckoj+'|')
               circuits.append(circuit)
    # -----

    #for c in circuits[::-1]:
    #    for cj in c:
    #        print("name of the circuit ",cj.name)
    #        print(cj.draw())
    #        print("BEFORE transpiling OP, DEPTH ",cj.count_ops(),cj.depth())
    #        from qiskit.compiler import transpile
    #        IBMQ.load_account()
    #        provider     = IBMQ.get_provider(hub='ibm-q-internal', group='deployed', project='default')
    #        real_device  = provider.get_backend('ibmq_boeblingen')
    #        cj_tilde = transpile(cj,real_device,optimization_level=3,initial_layout=[0,1,2])
    #        print("AFTER transpiling (ol=3,0->4) OP, DEPTH ",cj_tilde.count_ops(),cj_tilde.depth())
    #        #cj_tilde = transpile(cj,real_device,optimization_level=0,initial_layout=[0,1,2])
    #        #print("AFTER transpiling (ol=0,0->4) OP, DEPTH ",cj_tilde.count_ops(),cj_tilde.depth())
    #        #cj_tilde = transpile(cj,real_device,optimization_level=3,initial_layout=[2,0,1])
    #        #print("AFTER transpiling (ol=3,2->4) OP, DEPTH ",cj_tilde.count_ops(),cj_tilde.depth())
    #        #cj_tilde = transpile(cj,real_device,optimization_level=0,initial_layout=[2,0,1])
    #        #print("AFTER transpiling (ol=0,2->4) OP, DEPTH ",cj_tilde.count_ops(),cj_tilde.depth())
    #        print(cj_tilde.draw())
    #        print("="*53)
    #exit()

    if circuits:
        to_be_simulated_circuits = \
            functools.reduce(lambda x, y: x + y, [c for c in circuits if c is not None])
        fname,operation = rf
        import pickle
        if(operation=='read'):
           with open(fname,'rb') as input:
                result = pickle.load(input)

           nr = len(result.results)
           for ir in range(nr):
               print("DICTIONARY BEING READ..... ",result.results[ir].header.name)

        if(operation=='write'):
           result = instance.execute(to_be_simulated_circuits)
           with open(fname,'wb') as output:
                pickle.dump(result,output,pickle.HIGHEST_PROTOCOL)
    # -----
    result,res_ovlp = post_select(result,operator_dict[oj].num_qubits,particle_number,ps_fun,res_ovlp,ps_mapping,ps_tqr)
    # -----
    res = {}
    for ck in circuit_dict.keys():
        ck_list,oper_ck = get_circuit_oper(ck)
        for oj in ['re_%s'%oper_ck,'im_%s'%oper_ck]:
            ckoj = oj+'|'+ck_list
            if(not operator_dict[oj].is_empty()):
               mean,std = operator_dict[oj].evaluate_with_result(
                          result                      = result,
                          statevector_mode            = instance.is_statevector,
                          use_simulator_snapshot_mode = instance.is_statevector,
                          circuit_name_prefix         = ckoj+'|')
               mean,std = np.real(mean),np.abs(std)
               res[ckoj] = [mean,std]
            else:
               res[ckoj] = [0,0]
    res = {**res,**res_ovlp}
    return res

def is_postselected(ps_fun,x,particle_number,ps_mapping,ps_tqr):
    if(ps_mapping=='jordan_wigner' and ps_tqr==False):
       # only for jordan-wigner and without tapering
       na,nb = x[:len(x)-1],x[:len(x)-1]
       na,nb = na[:len(na)//2],nb[len(nb)//2:]
       na,nb = [int(x) for x in str(na)],[int(x) for x in str(nb)]
       na,nb = sum(na),sum(nb)
       if(ps_fun=='number'): return(na+nb==particle_number[0]+particle_number[1])
       elif(ps_fun=='spin'): return(na==particle_number[0] and nb==particle_number[1])
       elif(ps_fun=='none'): return(True)
       else:
            print("ERROR: ps_fun = ",ps_fun," is not valid")
            assert(False)
    elif(ps_mapping=='parity' and ps_tqr==True):
         #print("DEBUG PS, original string ",x)
         x_parity = [int(xi) for xi in list(x[:len(x)-1])]
         x_parity = x_parity[::-1]
         #print("DEBUG PS, original string, removing ancilla and having left-to-right order ",x_parity)
         x_parity = x_parity[:len(x_parity)//2]+[particle_number[0]%2]+x_parity[len(x_parity)//2:]+[(particle_number[0]+particle_number[1])%2]
         #print("DEBUG PS, parity string ",x_parity)
         x_parity = [x_parity[0]]+[(x_parity[i]+x_parity[i-1])%2 for i in range(1,len(x_parity))]
         #print("DEBUG PS, JW ",x_parity)
         na,nb = sum(x_parity[:len(x_parity)//2]),sum(x_parity[len(x_parity)//2:])
         #print("DEBUG PS, na,nb ",na,nb)
         #print("=============================")
         if(ps_fun=='number'): return(na+nb==particle_number[0]+particle_number[1])
         elif(ps_fun=='spin'): return(na==particle_number[0] and nb==particle_number[1])
         elif(ps_fun=='none'): return(True)
         else:
            print("ERROR: ps_fun = ",ps_fun," is not valid")
            assert(False)
    else:
            print("ERROR: ps_mapping/ps_tqr ",ps_mapping,ps_tqr," not valid")
            assert(False)


def post_select(result,num_qubits,particle_number,ps_fun,res_ovlp,ps_mapping,ps_tqr):
    nr = len(result.results)
    for ir in range(nr):
        r = result.results[ir]
        d = result.results[ir].data
        data = d.counts
        data = postprocess.format_counts(data,header={'memory_slots':num_qubits})
        h    = result.results[ir].header.name.split('|')
        # postselect
        data = {x: data[x] for x in data.keys() if is_postselected(ps_fun,x,particle_number,ps_mapping,ps_tqr)}
        if('re' in h[0]): ovlp_part = 're_S'
        if('im' in h[0]): ovlp_part = 'im_S'
        # overlap contribution from post-selected counts
        for x in data.keys():
            key = '%s|%s' % (ovlp_part,'|'.join(h[1:len(h)-1]))
            bx = x[len(x)-1]
            if(bx=='0'): res_ovlp[key][0] += data[x]
            else:        res_ovlp[key][1] += data[x]
        # overwriting results, showing post-selected counts only
        data = {hex(int(x,2)): data[x] for x in data.keys() if is_postselected(ps_fun,x,particle_number,ps_mapping,ps_tqr)}
        result.results[ir].data.counts = data
        result.results[ir].shots = sum([data[x] for x in data.keys()]) 
    for k in res_ovlp.keys():
        num_p,num_m = res_ovlp[k]
        n,m = num_p+num_m,(num_p-num_m)/(num_p+num_m)
        res_ovlp[k] = (m,np.sqrt(1-m**2)/np.sqrt(n))
    return result,res_ovlp
