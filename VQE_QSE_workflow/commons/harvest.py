import numpy as np
import functools
from   qiskit.result         import postprocess,models

def is_postselected(x,ps_fun,z_meas,particle_number):
    if(not z_meas):
       return(True)
    na,nb = x[:len(x)-1],x[:len(x)-1]
    na,nb = na[:len(na)//2],nb[len(nb)//2:]
    na,nb = [int(x) for x in str(na)],[int(x) for x in str(nb)]
    na,nb = sum(na),sum(nb)
    if(ps_fun=='number'): return(na+nb==particle_number[0]+particle_number[1])
    elif(ps_fun=='spin'): return(na==particle_number[0] and nb==particle_number[1])
    elif(ps_fun=='none'): return(True)
    else:
        assert(False)

def is_zeta_measurement(oper):
    isz=[]
    for c,P in oper._paulis:
        isPz = True
        for xm in P._x[0]:
            if(xm): isPz = False
        isz.append(isPz)
    return isz

def post_select(result,postselection,nelec,z_meas,num_qubits): 
    nr = len(result.results)
    for ir in range(nr):
        r = result.results[ir]
        d = result.results[ir].data
        data = d.counts
        data = postprocess.format_counts(data,header={'memory_slots':num_qubits})
        data = {x: data[x]             for x in data.keys() if is_postselected(x,postselection,z_meas[ir],nelec)}
        data = {hex(int(x,2)): data[x] for x in data.keys() if is_postselected(x,postselection,z_meas[ir],nelec)}
        result.results[ir].data.counts = data
        result.results[ir].shots = sum([data[x] for x in data.keys()])
    return result

def measure_operators(operators,wfn_circuit,instance,postselection='none',nelec=(0,0)):
    z_meas = []
    circuits = []
    for idx,oper in enumerate(operators):
        if(not oper.is_empty()):
           circuit = oper.construct_evaluation_circuit(
                     wave_function               = wfn_circuit,
                     statevector_mode            = instance.is_statevector,
                     use_simulator_snapshot_mode = instance.is_statevector,
                     circuit_name_prefix         = 'oper_'+str(idx))
           circuits.append(circuit)
        z_meas += is_zeta_measurement(oper)
    if circuits:
        to_be_simulated_circuits = \
            functools.reduce(lambda x, y: x + y, [c for c in circuits if c is not None])
        result = instance.execute(to_be_simulated_circuits)
    # ---
    if(postselection=='none'):
       result = results
    elif(postselection=='number'):
       result = post_select(result,postselection,nelec,z_meas,operators[0].num_qubits)
    # ---
    results_list = []
    for idx,oper in enumerate(operators):
        print(idx,oper.print_details())
        if(not oper.is_empty()):
           mean,std = oper.evaluate_with_result(
                      result = result,statevector_mode = instance.is_statevector,
                      use_simulator_snapshot_mode = instance.is_statevector,
                      circuit_name_prefix         = 'oper_'+str(idx))
           if(np.abs(np.imag(mean))>1e-4): print("attention: IMAG",mean)
           results_list.append([np.real(mean),np.abs(std)])
        else:
           results_list.append([0,0])
        print(idx,results_list)
    # ---
    return results_list

