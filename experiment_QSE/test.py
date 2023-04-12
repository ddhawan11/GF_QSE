from initial_imports import *
from creators_and_destructors import *

from qiskit import *
import subprocess
from datetime import datetime

logfile=open('info.txt','w')
dateTimeObj = datetime.now()
logfile.write("date and time "+str(dateTimeObj)+"\n")
logfile.write("\n")
label = subprocess.check_output(['git','log','-1','--format="%H"']).strip()
logfile.write("git commit "+str(label)+"\n")
logfile.write("\n")
qiskit_dict = qiskit.__qiskit_version__
logfile.write("qiskit version \n")
logfile.write(str(qiskit_dict))
logfile.write("\n")

calculation = ['initial','matrix_elements','post_processing']

# to run a QSE calculation, first we need a Hamiltonian operator
x    = np.load('../vqe_tutorial/quccsd_results.npy',allow_pickle=True).item()
H_op = x['hamiltonian']
H_op = H_op + x['molecule_info']['E0'] * Identity(H_op.num_qubits)
core = {'n_spin_orbitals'     : 2*x['molecule_info']['n'],
        'qubit_mapping'       : x['map_type'],
        'two_qubit_reduction' : x['two_qubit_reduction'],
        'num_particles'       : (x['molecule_info']['na'],x['molecule_info']['nb'])}

#driver = PySCFDriver(atom='''H 0 0 -0.37; H 0 0 0.37''',unit=UnitsType.ANGSTROM,charge=0,spin=0,
#                     basis='6-31g',hf_method=HFMethodType.ROHF,symgroup='Dooh')
#mol    = driver.run()
#core   = Hamiltonian(transformation=TransformationType.FULL,qubit_mapping=QubitMappingType.JORDAN_WIGNER,
#                     two_qubit_reduction=False,freeze_core=False,orbital_reduction=[])
#H_op,_ = core.run(mol)
#dE     = core._energy_shift + core._ph_energy_shift + core._nuclear_repulsion_energy
#H_op   = H_op + dE * Identity(H_op.num_qubits)
operator_pool = build_qse_operators(excitations='ea',dressed=False,mol_info=core)
#print(len(operator_pool))
#exit()
#init_state = HartreeFock(num_orbitals=core._molecule_info['num_orbitals'],
#                    qubit_mapping=core['qubit_mapping'],two_qubit_reduction=core['two_qubit_reduction'],
#                    num_particles=core['num_particles'])
#circuit    = init_state.construct_circuit()
circuit  = x['vqe_circuit']

backend  = Aer.get_backend('statevector_simulator')
instance = QuantumInstance(backend=backend,shots=10000)

# ========== QSE ========== #

QSE_calc = QSE(hamiltonian=H_op,psi=circuit,operators=operator_pool,instance=instance,offset=0.0,threshold=1e-6)
if('initial' in calculation):
   check    = QSE_calc.diagonalize_H() #mol)
   result   = QSE_calc.run('qse_result.npy',trace_of_s=x['molecule_info']['na'],stabilization_factor=0,ntry=1)
result = np.load('qse_result.npy',allow_pickle=True).item()
QSE_calc.print_information(result)

d_list = build_qse_operators(excitations='ea',dressed=False,mol_info=core)

if('matrix_elements' in calculation):
   QSE_calc.measure_matrix_elements(operators=d_list,  # compute (Psi_mu|a(i)|Psi_VQE)
                                    fname='qse_matrix_elements.npy',ipea=True)

matrix_elements = np.load('qse_matrix_elements.npy',allow_pickle=True)
if('post_processing' in calculation):
   #options_1 = {'kind':'real_time','nt':100,'dt':0.25}
   options_2 = {'kind':'imaginary_frequency_ip','beta':1000.0,'n_max':1000}
   #options_3 = {'kind':'imaginary_time','nt':100,'dt':0.01}
#x1,y1,dy1 = QSE_calc.measure_correlation_function(d_list,options=options_1,ipea=True,matrix_elements=matrix_elements,vqe_energy=x['results_vqe'][0])
   x2,y2,dy2 = QSE_calc.measure_correlation_function(d_list,options=options_2,ipea=True,matrix_elements=matrix_elements,vqe_energy=x['results_vqe'][0])
   #x3,y3,dy3 = QSE_calc.measure_correlation_function(d_list,options=options_3,ipea=True,matrix_elements=matrix_elements,vqe_energy=x['results_vqe'][0],samples=1)
   np.save('qse_greens_functions.npy',{'imaginary_frequency':(x2,y2,dy2)},allow_pickle=True)

#{'real_time':(x1,y1,dy1),'frequency':(x2,y2,dy2),'imaginary_time':(x3,y3,dy3)},allow_pickle=True)

   green = np.load('qse_greens_functions.npy',allow_pickle=True).item()
   import matplotlib.pyplot as plt
   fig,axs = plt.subplots(2)
   plt.subplots_adjust(wspace=0.4,hspace=0.4)
   # ---
   #axs[0].plot(green['real_time'][0],np.real(green['real_time'][1]))
   #axs[0].plot(green['real_time'][0],np.imag(green['real_time'][1]))
   #axs[0].set_title('real-time Green function')
   ## ---
   #axs[1].plot(green['frequency'][0],np.real(green['frequency'][1]),label='re')
   #axs[1].plot(green['frequency'][0],np.imag(green['frequency'][1]),label='im')
   #axs[1].set_title('frequency-domain Green function')
   # ---
   for i,di in enumerate(d_list):
       for j,dj in enumerate(d_list):
           x  = green['imaginary_frequency'][0]
           y  = green['imaginary_frequency'][1][:,i,j,0] # avg, re
           dy = green['imaginary_frequency'][2][:,i,j,0] # std, re
           axs[0].errorbar(x,y,yerr=dy,label='re G(%d,%d)'%(i,j))
           y  = green['imaginary_frequency'][1][:,i,j,1] # avg, im
           dy = green['imaginary_frequency'][2][:,i,j,1] # std, im
           axs[1].errorbar(x,y,yerr=dy,label='im G(%d,%d)'%(i,j))
   axs[0].set_title('imaginary-freq Green function, real part')
   axs[1].set_title('imaginary-freq Green function, imag part')
   # ---
   axs[0].legend()
   axs[1].legend()
   # ---
   plt.show()

