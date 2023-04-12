import numpy as np
from scipy import linalg as LA
from qiskit.chemistry.core                         import Hamiltonian,TransformationType,QubitMappingType
from qiskit.chemistry.components.initial_states    import HartreeFock
from qiskit.aqua.components.optimizers             import L_BFGS_B,CG,SPSA,COBYLA,ADAM
from qiskit.aqua.algorithms                        import VQE
from qiskit.aqua                                   import QuantumInstance,aqua_globals
from qiskit.circuit.library                        import EfficientSU2
from subroutines                                   import *
from utils                                         import *
from CustomVarForm                                 import *
from givens import *

#from qiskit.chemistry                                              import set_qiskit_chemistry_logging
#import logging
#set_qiskit_chemistry_logging(logging.DEBUG)

# ------- dictionary with Hamiltonian information

#
# H = c + t(a,b) C*(a) C(b) + v(prqs) C*(p) C*(q) C(s) C(r)
#
# the orbitals need to be orthonormal 
#

import numpy as np
from   pyscf import gto,scf,ao2mo
from   scipy import linalg as LA
from   local_uccsd import *
from   CustomEfficientSU2 import *

mol = gto.M(verbose=4,atom=[['H',(0,0,0)],['H',(0,0,0.741)]],basis='sto-6g')
mf  = scf.RHF(mol)
E   = mf.kernel()
print("HF energy ",E)

# AO hamiltonian
E0  = mol.energy_nuc()
s1  = mf.get_ovlp()
h1  = mf.get_hcore()
h2  = ao2mo.restore(1,mf._eri,mol.nao_nr())

# LO hamiltonian
sgm,U = LA.eigh(s1)
print("overlap matrix eigenvalues ",sgm)
C     = np.einsum('ij,j->ij',U,1.0/np.sqrt(sgm))
C     = np.dot(C,U.T)
n     = C.shape[0]

# AO2LO transformation
s1  = np.einsum('pi,pq,qj->ij',C,s1,C)
h1  = np.einsum('pi,pq,qj->ij',C,h1,C)
h2  = np.einsum('pi,rj,qk,sl,prqs->ijkl',C,C,C,C,h2)

# LO2MO transformation unitary
# |m) = \sum_a CHF(am) |a)
# |l) = \sum_a C(al)   |a) ---> |b) = \sum_l C^{-1}(lb) |l) = \sum_{al} C^{-1}(lb) C(al) |a) = |b)
# |m) = \sum_{al} CHF(am) C^{-1}(lb) |l) = \sum_l [ \sum_a CHF(am) C^{-1}(lb) ] |l)
CHF = mf.mo_coeff
VHF = np.einsum('am,lb->lm',CHF,LA.inv(C))

'''
#x = PrettyTable()
#x.field_names = ["i","j","S[i,j]","T[i,j]"]
#
#n   = h1.shape[0] 
#for i in range(n):
#    for j in range(n):
#        x.add_row(['%d'%i,'%d'%j,'%.6f'%s1[i,j],'%.6f'%h1[i,j]])
#print(str(x))
# MOs
#C   = mf.mo_coeff
# eigenvectors of S
sgm,U = LA.eigh(s1)
print("overlap matrix eigenvalues ",sgm)
C     = np.einsum('ij,j->ij',U,1.0/np.sqrt(sgm))
C     = np.dot(C,U.T)
n     = C.shape[0]

# AO2MO transformation
s1  = np.einsum('pi,pq,qj->ij',C,s1,C)
h1  = np.einsum('pi,pq,qj->ij',C,h1,C)
h2  = np.einsum('pi,rj,qk,sl,prqs->ijkl',C,C,C,C,h2)

import prettytable

x = PrettyTable()
x.field_names = ["i","j","S[i,j]","T[i,j]","C[i,j]"]

for i in range(n):
    for j in range(n):
        x.add_row(['%d'%i,'%d'%j,'%.6f'%s1[i,j],'%.6f'%h1[i,j],'%.6f'%C[i,j]])
print(str(x))

for p in range(n):
    for r in range(n):
        for q in range(n):
            for s in range(n):
                print('%d %d %d %d %.6f ' % (p,r,q,s,h2[p,r,q,s]))
#exit()
'''

import numpy as np
n = n
c = E0
t = h1
v = h2

mol_data = {'n'  : n,
            'na' : 1,                    # number of spin-up
            'nb' : 1,                    # number of spin-down 
            'E0' : c,
            'h1' : t,
            'h2' : v}

# ----------------------------------------------- mapping to qubits

outfile   = open('vqe.txt','w')
molecule  = make_qmolecule_from_data(mol_data)
core      = Hamiltonian(transformation=TransformationType.FULL,qubit_mapping=QubitMappingType.JORDAN_WIGNER,
                        two_qubit_reduction=False,freeze_core=False,orbital_reduction=[])
H_op,A_op = core.run(molecule)
H_op      = H_op

# transformation to qubits https://arxiv.org/abs/1208.5986
# H_op = qubit Hamiltonian
# A_op = set of 3 auxiliary operators: particle number, spin-squared, spin-z

init_state = HartreeFock(num_orbitals=core._molecule_info['num_orbitals'],qubit_mapping=core._qubit_mapping,
                         two_qubit_reduction=core._two_qubit_reduction,num_particles=core._molecule_info['num_particles'])
HVF_circuit = construct_LO2MO_circuit(VHF,nelec=(1,1),n=2,qubit_mapping=QubitMappingType.JORDAN_WIGNER,two_qubit_reduction=False)
#c0          = init_state.construct_circuit()
#init_state.construct_circuit = lambda mode='circuit',register=None : c0 + HVF_circuit

outfile.write("\nHartree-Fock circuit\n")
outfile.write(str(init_state.construct_circuit().draw())+"\n")

backend          = Aer.get_backend('statevector_simulator')
quantum_instance = QuantumInstance(backend=backend,shots=1024*8)

res = measure_operators([H_op],init_state.construct_circuit()+HVF_circuit,quantum_instance) # function to measure a list of operators [H] on a circuit
                                                                                            # returns a list [r] where r=[m,s]
outfile.write("HF energy = %.6f +/- %.6f \n" % (res[0][0]+E0,res[0][1]))

# ----------------------------------------------- vqe with q-UCCSD Ansatz

#
# q-UCCSD https://arxiv.org/abs/1805.04340
# 
# |Psi) = exp(T-T*) |HF) = \prod_{mu} exp(t(mu)P(mu)) |HF)
#
# T = \sum_{ai} t(ai) C*(a) C(i) + \sum_{abij} t(abij) C*(a) C*(b) C(j) C(i)
#

var_form  = UCCSD(num_orbitals=core._molecule_info['num_orbitals'],num_particles=core._molecule_info['num_particles'],
                  active_occupied=None,active_unoccupied=None,initial_state=init_state,qubit_mapping=core._qubit_mapping,
                  two_qubit_reduction=core._two_qubit_reduction,num_time_slices=1,qf=HVF_circuit)
#
# g(x) = [f(x+h)-f(x)]/h   ,   h=1e-8
# Var[g(x)] = [ Var[f(x+h)]+Var[f(x)] ] / h**2 = (2 s**2/h**2)
# 
# 
# https://qiskit.org/documentation/stubs/qiskit.aqua.components.optimizers.SPSA.html
#
#optimizer = SPSA(maxiter=1000) #L_BFGS_B(maxiter=1000,iprint=1001)
#optimizer = ADAM(maxiter=1000)
optimizer = CG(maxiter=1000)
if(os.path.isfile('quccsd_input_parameters.txt')): p0 = np.loadtxt('quccsd_input_parameters.txt')
else:                                              p0 = np.zeros(var_form.num_parameters)
algo             = VQE(H_op,var_form,optimizer,aux_operators=A_op,include_custom=True,initial_point=p0)
#backend          = Aer.get_backend('qasm_simulator')
#quantum_instance = QuantumInstance(backend=backend,shots=1024*8)
algo_result      = algo.run(quantum_instance)

p1 = algo._ret['opt_params']
np.savetxt("quccsd_output_parameters.txt",p1)

res = measure_operators([H_op]+A_op,var_form.construct_circuit(p1),quantum_instance)

outfile.write("\nVQE q-UCCSD\n")
res_vqe,res_ee = get_results(H_op,A_op,molecule,core,algo_result,outfile,results_with_error_bars=res)
np.save('quccsd_results.npy',{'molecule_info'       : mol_data,
                              'map_type'            : core._qubit_mapping,
                              'two_qubit_reduction' : core._two_qubit_reduction,
                              'num_qubits'          : H_op.num_qubits,
                              'hamiltonian'         : H_op,
                              'num_parameters'      : var_form.num_parameters,
                              'results_vqe'         : res_vqe,
                              'vqe_circuit'         : var_form.construct_circuit(p1),
                              'results_eigensolver' : res_ee},allow_pickle=True)

#for t in np.arange(-10,10,0.001):
#    res_t = algo._energy_evaluation(t*p1)
#    print(t,res_t+E0)
#exit()

#C = var_form.construct_circuit(p1)
#C = transpilation(C,machine='ibmq_athens',opt=0,layout=[0,1])
#outfile.write("\nVQE-q-UCCSD circuit\n")
#outfile.write(str(C.draw())+"\n")

# ----------------------------------------------- vqe with a "hardware-efficient" Ansatz

#
# layers of su2 (i.e. single-qubit unitary) gates of Ry type su2_gates=['ry']
# alternating layers of entangling CNOT gates, assuming linear connectivity entanglement='linear'
# reps=1 ===> one layer of entangling gates (cheap, but inaccurate)
#
# Ry C     Ry C     Ry
# Ry X C   Ry X C   Ry
# Ry   X C Ry   X C Ry
# Ry     X Ry     X Ry
# 

var_form  = CustomEfficientSU2(num_qubits=H_op.num_qubits,reps=1,entanglement='linear',
                               initial_state=init_state.construct_circuit(),chf=HVF_circuit)
#optimizer = SPSA(maxiter=1000) #CG(maxiter=1000)
optimizer = CG(maxiter=1000)
if(os.path.isfile('he_input_parameters.txt')): p0 = np.loadtxt('he_input_parameters.txt')
else:                                          p0 = np.random.random(var_form.num_parameters)
algo             = VQE(H_op,var_form,optimizer,aux_operators=A_op,include_custom=True,initial_point=p0)
#backend          = Aer.get_backend('statevector_simulator')
#quantum_instance = QuantumInstance(backend=backend)
algo_result      = algo.run(quantum_instance)

p1 = algo._ret['opt_params']
np.savetxt("he_output_parameters.txt",p1)

res = measure_operators([H_op]+A_op,var_form.construct_circuit(p1),quantum_instance)

outfile.write("\nVQE hardware-efficient Ansatz\n")
res_vqe,res_ee = get_results(H_op,A_op,molecule,core,algo_result,outfile,results_with_error_bars=res)
np.save('he_results.npy',{'num_qubits'          : H_op.num_qubits,
                          'hamiltonian'         : H_op,
                          'num_parameters'      : var_form.num_parameters,
                          'results_vqe'         : res_vqe,
                          'vqe_circuit'         : var_form.construct_circuit(p1),
                          'results_eigensolver' : res_ee},allow_pickle=True)

#for t in np.arange(-10,10,0.01):
#    res_t = algo._energy_evaluation(t*p1)
#    print(t,res_t+E0)
#exit()
#C = algo.get_optimal_circuit() #var_form.construct_circuit(p1)
#C = transpilation(C,machine='ibmq_athens',opt=0,layout=[0,1])
#outfile.write("\nVQE-EfficientSU2 circuit\n")
#outfile.write(str(C.draw())+"\n")

# ----------------------------------------------- vqe with a "custom" Ansatz
#
# e.g. https://arxiv.org/pdf/1809.03827.pdf

var_form  = CustomVarForm(num_qubits=H_op.num_qubits,initial_state=init_state.construct_circuit(),
                          paulis=[['X','X','X','Y']],chf=HVF_circuit)
#optimizer = SPSA(maxiter=1000) #CG(maxiter=1000)
optimizer = CG(maxiter=1000)
if(os.path.isfile('custom_input_parameters.txt')): p0 = np.loadtxt('custom_input_parameters.txt')
else:                                          p0 = np.zeros(var_form.num_parameters)
algo             = VQE(H_op,var_form,optimizer,aux_operators=A_op,include_custom=True,initial_point=p0)
#backend          = Aer.get_backend('statevector_simulator')
#quantum_instance = QuantumInstance(backend=backend)
algo_result      = algo.run(quantum_instance)

p1 = algo._ret['opt_params']
np.savetxt("custom_output_parameters.txt",p1)

res = measure_operators([H_op]+A_op,var_form.construct_circuit(p1),quantum_instance)

outfile.write("\nVQE custom Ansatz\n")
res_vqe,res_ee = get_results(H_op,A_op,molecule,core,algo_result,outfile,results_with_error_bars=res)
np.save('custom_results.npy',{'num_qubits'          : H_op.num_qubits,
                              'hamiltonian'         : H_op,
                              'num_parameters'      : var_form.num_parameters,
                              'results_vqe'         : res_vqe,
                              'vqe_circuit'         : algo.get_optimal_circuit(),
                              'results_eigensolver' : res_ee},allow_pickle=True)

#C = algo.get_optimal_circuit()
#print(C.draw())

#for t in np.arange(-10,10,0.01):
#    res_t = algo._energy_evaluation(t*p1)
#    print(t,res_t+E0)
#exit()

#C = transpilation(C,machine='ibmq_athens',opt=0,layout=[0,1])
#outfile.write("\nCustom circuit\n")
#outfile.write(str(C.draw())+"\n")

# ----------------------------------------------- 

