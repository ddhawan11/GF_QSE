import numpy as np
import scipy
import pickle
from qiskit.chemistry.core                         import Hamiltonian,TransformationType,QubitMappingType
from qiskit.chemistry.components.initial_states    import HartreeFock
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.aqua.components.optimizers             import L_BFGS_B,CG,SPSA
from qiskit.aqua.algorithms                        import VQE
from qiskit.aqua                                   import QuantumInstance,aqua_globals
from qiskit.circuit.library                        import EfficientSU2
from subroutines                                   import *
from utils                                         import *
from CustomVarForm                                 import *
import qiskit

import scf
# ------- dictionary with Hamiltonian information

#
# H = c + t(a,b) C*(a) C(b) + v(prqs) C*(p) C*(q) C(s) C(r)
#
# the orbitals need to be orthonormal 
#
print(qiskit.__version__)
import numpy as np
n = 2                                    # number of orbitals
t_file = open("T.obj","rb")
v_file = open("V.obj","rb")

c =  0.714139#np.random.random()                   # energy offset
#t = np.random.random((n,n))              # integrals of the 1-body part of H
#t = (t+t.T)/2.0
t = pickle.load(t_file)

eigval,eigvec = scipy.linalg.eigh(t)
print("eigval", eigval)
print("eigvec", eigvec)
#v = np.random.random((n,n,10))
#v = (v+v.transpose((1,0,2)))
v = pickle.load(v_file)
#v = np.einsum('ijkl->iklj',v)   # ERI in the Chemist's notation

# import prettytable

# x = PrettyTable()
# x.field_names = ["i","j","T[i,j]"]

# for i in range(n):
#     for j in range(n):
#         x.add_row(['%d'%i,'%d'%j,'%.6f'%t[i,j]])
# print(str(x))

# for p in range(n):
#     for r in range(n):
#         for q in range(n):
#             for s in range(n):
#                 print('%d %d %d %d %.6f ' % (p,r,q,s,v[p,r,q,s]))



t_file.close()
v_file.close()
mol_data = {'n'  : n,
            'na' : 1,                    # number of spin-up
            'nb' : 1,                    # number of spin-down 
            'E0' : c,
            'h1' : float(t),
            'h2' : float(v)}


print ("SCF energy:", scf.do_scf(mol_data))

exit()
# ----------------------------------------------- mapping to qubits

outfile   = open('vqe.txt','w')
molecule  = make_qmolecule_from_data(mol_data)
core      = Hamiltonian(transformation=TransformationType.FULL,qubit_mapping=QubitMappingType.PARITY,
                        two_qubit_reduction=True,freeze_core=False,orbital_reduction=[])
H_op,A_op = core.run(molecule)

# transformation to qubits https://arxiv.org/abs/1208.5986
# H_op = qubit Hamiltonian
# A_op = set of 3 auxiliary operators: particle number, spin-squared, spin-z

init_state = HartreeFock(num_orbitals=core._molecule_info['num_orbitals'],qubit_mapping=core._qubit_mapping,
                         two_qubit_reduction=core._two_qubit_reduction,num_particles=core._molecule_info['num_particles'])

outfile.write("\nHartree-Fock circuit\n")
outfile.write(str(init_state.construct_circuit().draw())+"\n")

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
                  two_qubit_reduction=core._two_qubit_reduction,num_time_slices=1)
optimizer = L_BFGS_B(maxiter=1000,iprint=1001)
if(os.path.isfile('quccsd_input_parameters.txt')): p0 = np.loadtxt('quccsd_input_parameters.txt')
else:                                              p0 = np.zeros(var_form.num_parameters)
algo             = VQE(H_op,var_form,optimizer,aux_operators=A_op,include_custom=True,initial_point=p0)
backend          = Aer.get_backend('statevector_simulator')
quantum_instance = QuantumInstance(backend=backend)
algo_result      = algo.run(quantum_instance)

p1 = algo._ret['opt_params']
np.savetxt("quccsd_output_parameters.txt",p1)

outfile.write("\nVQE q-UCCSD\n")
res_vqe,res_ee = get_results(H_op,A_op,molecule,core,algo_result,outfile)
np.save('quccsd_results.npy',{'num_qubits'          : H_op.num_qubits,
                              'hamiltonian'         : H_op,
                              'num_parameters'      : var_form.num_parameters,
                              'results_vqe'         : res_vqe,
                              'vqe_circuit'         : var_form.construct_circuit(p1),
                              'results_eigensolver' : res_ee},allow_pickle=True)

C = var_form.construct_circuit(p1)
C = transpilation(C,machine='ibmq_athens',opt=0,layout=[0,1])
outfile.write("\nVQE-q-UCCSD circuit\n")
outfile.write(str(C.draw())+"\n")

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

var_form  = EfficientSU2(num_qubits=H_op.num_qubits,reps=3,entanglement='linear',su2_gates=['ry'],initial_state=init_state)
optimizer = CG(maxiter=1000) #,iprint=1)
if(os.path.isfile('h2_input_parameters.txt')): p0 = np.loadtxt('h2_input_parameters.txt')
else:                                          p0 = np.random.random(var_form.num_parameters)
algo             = VQE(H_op,var_form,optimizer,aux_operators=A_op,include_custom=True,initial_point=p0)
backend          = Aer.get_backend('statevector_simulator')
quantum_instance = QuantumInstance(backend=backend)
algo_result      = algo.run(quantum_instance)

p1 = algo._ret['opt_params']
np.savetxt("h2_output_parameters.txt",p1)

outfile.write("\nVQE hardware-efficient Ansatz\n")
res_vqe,res_ee = get_results(H_op,A_op,molecule,core,algo_result,outfile)
np.save('h2_results.npy',{'num_qubits'          : H_op.num_qubits,
                          'hamiltonian'         : H_op,
                          'num_parameters'      : var_form.num_parameters,
                          'results_vqe'         : res_vqe,
                          'vqe_circuit'         : algo.get_optimal_circuit(),
                          'results_eigensolver' : res_ee},allow_pickle=True)

C = algo.get_optimal_circuit() #var_form.construct_circuit(p1)
C = transpilation(C,machine='ibmq_athens',opt=0,layout=[0,1])
outfile.write("\nVQE-EfficientSU2 circuit\n")
outfile.write(str(C.draw())+"\n")

# ----------------------------------------------- vqe with a "custom" Ansatz
#
# e.g. https://arxiv.org/pdf/1809.03827.pdf

#var_form  = CustomVarForm(num_qubits=H_op.num_qubits,initial_state=init_state.construct_circuit())
var_form = CustomVarForm(num_qubits=H_op.num_qubits,initial_state=init_state.construct_circuit(),
             paulis=[['X','Y']],chf=None)
optimizer = CG(maxiter=1000)
if(os.path.isfile('custom_input_parameters.txt')): p0 = np.loadtxt('custom_input_parameters.txt')
else:                                          p0 = np.random.random(var_form.num_parameters)#np.zeros(var_form.num_parameters)
algo             = VQE(H_op,var_form,optimizer,aux_operators=A_op,include_custom=True,initial_point=p0)
backend          = Aer.get_backend('statevector_simulator')
quantum_instance = QuantumInstance(backend=backend)
algo_result      = algo.run(quantum_instance)

p1 = algo._ret['opt_params']
np.savetxt("custom_output_parameters.txt",p1)

outfile.write("\nVQE custom Ansatz\n")
res_vqe,res_ee = get_results(H_op,A_op,molecule,core,algo_result,outfile)
np.save('custom_results.npy',{'num_qubits'          : H_op.num_qubits,
                              'hamiltonian'         : H_op,
                              'num_parameters'      : var_form.num_parameters,
                              'results_vqe'         : res_vqe,
                              'vqe_circuit'         : algo.get_optimal_circuit(),
                              'results_eigensolver' : res_ee},allow_pickle=True)

C = algo.get_optimal_circuit()
C = transpilation(C,machine='ibmq_athens',opt=0,layout=[0,1])
outfile.write("\nCustom circuit\n")
outfile.write(str(C.draw())+"\n")

# ----------------------------------------------- 

