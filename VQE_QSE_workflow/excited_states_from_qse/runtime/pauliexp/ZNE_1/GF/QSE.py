def adjoint(WPO):
    import numpy as np
    ADJ = WPO.copy()
    ADJ._paulis = [[np.conj(weight),pauli] for weight,pauli in WPO._paulis]
    return ADJ

class QSE:

      def __init__(self,psi=None,operators=None,mol_info=None,instance_dict={}):
          import qiskit
          from qiskit                                import Aer, IBMQ
          from qiskit.aqua                           import QuantumInstance
          from qiskit.ignis.mitigation.measurement   import (complete_meas_cal,tensored_meas_cal,CompleteMeasFitter,TensoredMeasFitter)
          self.psi         = psi
          self.operators   = operators
          self.mol_info    = mol_info
          self.matrices    = {'S':None,'H':None}
          self.result      = {}
          if(mol_info['operators']['tapering']):
             self.hamiltonian   = mol_info['operators']['untapered_h_op']
             self.tapering      = mol_info['operators']['tapering']
             self.z2syms        = mol_info['operators']['tapering_info'][0]
             self.target_sector = mol_info['operators']['target_sector']
          else:
             self.hamiltonian   = mol_info['operators']['h_op']
             self.tapering      = mol_info['operators']['tapering']

#          runtype    = ['qasm',   'ibmq_quito',True, 20000]
          if ("simulator" in instance_dict['instance']):
              print("Running on simulator ")
              backend  = Aer.get_backend(instance_dict['instance'])
              self.instance = QuantumInstance(backend=backend,shots=instance_dict['shots'])
          elif(instance_dict['instance']=='noise_model'): # ==== QASM simulator, with noise model from an actual machine
              print("Running on qasm simulator with noise of a real hardware")
              provider         = IBMQ.load_account()
              backend          = provider.get_backend(instance_dict['quantum_machine'])
              noise_model      = NoiseModel.from_backend(backend)
              coupling_map     = backend.configuration().coupling_map
              basis_gates      = noise_model.basis_gates
              simulator        = Aer.get_backend('qasm_simulator')
              if(instance_dict['error_mitigation']):  # ==== with readout error mitigation
                  print("with error mitigation")
                  instance = QuantumInstance(backend=simulator,noise_model=noise_model,coupling_map=coupling_map,basis_gates=basis_gates,
                                             measurement_error_mitigation_cls=CompleteMeasFitter,shots=instance_dict['shots'])
              else:
                  # ==== without readout error mitigation
                  print("without error mitigation")
                  self.instance = QuantumInstance(backend=simulator,noise_model=noise_model,coupling_map=coupling_map,basis_gates=basis_gates,shots=instance_dict['shots'])
          elif(instance_dict['instance']=='hardware'): # ==== on HARDWARE, I would do it with readout error mitigation
              print("Running on hardware")
              #provider  = IBMQ.get_provider(hub='ibm-q',group='open',project='main')
              simulator = instance_dict['provider'].get_backend(instance_dict['quantum_machine'])
              self.instance  = QuantumInstance(backend=simulator,measurement_error_mitigation_cls=CompleteMeasFitter,shots=instance_dict['shots'])

      def construct_qse_matrices(self,outfile,resfile='results.npy'):

          import numpy as np
          import sys
          sys.path.append('/Users/ddhawan/Documents/internship_diksha/VQE_QSE_workflow/commons/')
          from tapering import taper_auxiliary_operators
          from harvest import measure_operators
          import time

          n = len(self.operators)
          H = np.zeros((n,n,2))
          S = np.zeros((n,n,2))
          print("Entering the loop")
          for i,Pi in enumerate(self.operators):
              for j,Pj in enumerate(self.operators):
                  if(i<=j):
#                     t0 = time.time()
                     Sij_oper = adjoint(Pi)*Pj
                     Hij_oper = adjoint(Pi)*(self.hamiltonian*Pj)
#                     t1 = time.time()
#                     print(t1)
                     if(self.tapering):
                        Sij_oper,Hij_oper = taper_auxiliary_operators([Sij_oper,Hij_oper],self.z2syms,self.target_sector)
#                     t2 = time.time()
#                     print(t2)
                     S[i,j,:] = measure_operators([Sij_oper],self.psi,self.instance)[0]
                     S[j,i,:] = S[i,j,:]
                     np.save("S_matrix_"+str(i)+str(j)+".npy",S[i,j,:],allow_pickle=True)

                     H[i,j,:] = measure_operators([Hij_oper],self.psi,self.instance)[0]
                     H[j,i,:] = H[i,j,:]
                     np.save("H_matrix_"+str(i)+str(j)+".npy",H[i,j,:],allow_pickle=True)

#                     t3 = time.time()
#                     print(t3)
                     outfile.write("S,H matrix elements (%d/%d,%d/%d) computed \n" % (i+1,n,j+1,n,))
#                     outfile.write("times op,taper,measure %.6f %.6f %.6f  \n" % (t1-t0,t2-t1,t3-t2))

          H_saved = np.zeros((n,n,2))
          S_saved = np.zeros((n,n,2))

          for i in range(n):
              for j in range(n):
                if(i<=j):
                    H_saved[i,j,:] = np.load("H_matrix_"+str(i)+str(j)+".npy", allow_pickle=True) 
                    S_saved[i,j,:] = np.load("S_matrix_"+str(i)+str(j)+".npy", allow_pickle=True) 
                    S_saved[j,i,:] = S_saved[i,j,:]
                    H_saved[j,i,:] = H_saved[i,j,:]
          self.matrices['S'] = S_saved
          self.matrices['H'] = H_saved

          np.save(resfile,self.matrices,allow_pickle=True)

      def compute_transition_matrix_elements(self,transition_operators,outfile,resfile='results.npy'):

          import numpy as np
          import sys
          sys.path.append('../commons/')
          from tapering import taper_auxiliary_operators
          from harvest import measure_operators
          import time

          n = len(self.operators)
          m = len(transition_operators)
          C = np.zeros((m,n,2),dtype=complex)

          # C(m,I) = < Psi(vqe) | T(m) E(I) | Psi(vqe) >
          for i,Pi in enumerate(transition_operators):
              for j,Qj in enumerate(self.operators):
#                  t0 = time.time()
                  Cij_oper = Pi*Qj
#                  t1 = time.time()
                  if(self.tapering):
                     Cij_oper = taper_auxiliary_operators([Cij_oper],self.z2syms,self.target_sector)[0]
#                  t2 = time.time()
                  C[i,j,:] = measure_operators([Cij_oper],self.psi,self.instance)[0]
#                  t3 = time.time()
                  np.save("C_matrix_"+str(i)+str(j)+".npy",C[i,j,:],allow_pickle=True)
                  outfile.write("C matrix elements (%d/%d,%d/%d) computed \n" % (i+1,m,j+1,n,))
#                  outfile.write("times op,taper,measure %.6f %.6f %.6f  \n" % (t1-t0,t2-t1,t3-t2))

          C_saved = np.zeros((m,n,2))

          for i in range(m):
              for j in range(n):
                 C_saved[i,j,:] = np.load("C_matrix_"+str(i)+str(j)+".npy", allow_pickle=True) 

          self.matrices['C'] = C_saved

          np.save(resfile,self.matrices,allow_pickle=True)


