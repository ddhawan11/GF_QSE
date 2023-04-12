def adjoint(WPO):
    import numpy as np
    ADJ = WPO.copy()
    ADJ._paulis = [[np.conj(weight),pauli] for weight,pauli in WPO._paulis]
    return ADJ

class QSE:

      def __init__(self,psi=None,operators=None,mol_info=None,instance_dict={}):
          from qiskit                            import Aer
          from qiskit.aqua                       import QuantumInstance
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
          backend          = Aer.get_backend(instance_dict['instance'])
          quantum_instance = QuantumInstance(backend=backend,shots=instance_dict['shots'])
          self.instance    = quantum_instance

      def construct_qse_matrices(self,outfile,resfile='results.npy'):

          import numpy as np
          import sys
          sys.path.append('/home/ddhawan/internship_diksha/VQE_QSE_workflow/commons/')
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
                     t0 = time.time()
                     Sij_oper = adjoint(Pi)*Pj
                     Hij_oper = adjoint(Pi)*(self.hamiltonian*Pj)
                     t1 = time.time()
                     if(self.tapering):
                        Sij_oper,Hij_oper = taper_auxiliary_operators([Sij_oper,Hij_oper],self.z2syms,self.target_sector)
                     t2 = time.time()
                     S[i,j,:] = measure_operators([Sij_oper],self.psi,self.instance)[0]
                     H[i,j,:] = measure_operators([Hij_oper],self.psi,self.instance)[0]
                     t3 = time.time()
                     S[j,i,:] = S[i,j,:]
                     H[j,i,:] = H[i,j,:]
                     outfile.write("S,H matrix elements (%d/%d,%d/%d) computed \n" % (i+1,n,j+1,n,))
                     outfile.write("times op,taper,measure %.6f %.6f %.6f  \n" % (t1-t0,t2-t1,t3-t2))


          self.matrices['S'] = S
          self.matrices['H'] = H
          np.save(resfile,self.matrices,allow_pickle=True)




      def get_qse_element(self,i,j,outfile):

          import numpy as np
          import sys
          sys.path.append('/home/ddhawan/internship_diksha/VQE_QSE_workflow/commons/')
          from tapering import taper_auxiliary_operators
          from harvest import measure_operators
          import time

#          n = len(self.operators)
          H = np.zeros((2))
          S = np.zeros((2))

#          for i,Pi in enumerate(self.operators):
#              for j,Pj in enumerate(self.operators):
#                  if(i<=j):
          t0 = time.time()
          Sij_oper = adjoint(self.operators[i])*self.operators[j]
          Hij_oper = adjoint(self.operators[i])*(self.hamiltonian*self.operators[j])
          t1 = time.time()

          if(self.tapering):
              Sij_oper,Hij_oper = taper_auxiliary_operators([Sij_oper,Hij_oper],self.z2syms,self.target_sector)
          t2 = time.time()

          S[:] = measure_operators([Sij_oper],self.psi,self.instance)[0]
          H[:] = measure_operators([Hij_oper],self.psi,self.instance)[0]
          t3 = time.time()

#          S[j,i,:] = S[i,j,:]
#          H[j,i,:] = H[i,j,:]
          outfile.write("S,H matrix elements (%d,%d) computed \n" % (i+1,j+1))
          outfile.write("times op,taper,measure %.6f %.6f %.6f  \n" % (t1-t0,t2-t1,t3-t2))


          self.matrices['S'] = S
          self.matrices['H'] = H


      def compute_transition_matrix_elements(self,transition_operators,outfile,resfile='results.npy'):

          import numpy as np
          import sys
          sys.path.append('home/ddhawan/internship_diksha/VQE_QSE_workflow/commons/')
          from tapering import taper_auxiliary_operators
          from harvest import measure_operators
          import time

          n = len(self.operators)
          m = len(transition_operators)
          C = np.zeros((m,n,2),dtype=complex)

          # C(m,I) = < Psi(vqe) | T(m) E(I) | Psi(vqe) >
          for i,Pi in enumerate(transition_operators):
              for j,Qj in enumerate(self.operators):
                  t0 = time.time()
                  Cij_oper = Pi*Qj
                  t1 = time.time()
                  if(self.tapering):
                     Cij_oper = taper_auxiliary_operators([Cij_oper],self.z2syms,self.target_sector)[0]
                  t2 = time.time()
                  C[i,j,:] = measure_operators([Cij_oper],self.psi,self.instance)[0]
                  t3 = time.time()
                  outfile.write("C matrix elements (%d/%d,%d/%d) computed \n" % (i+1,m,j+1,n,))
                  outfile.write("times op,taper,measure %.6f %.6f %.6f  \n" % (t1-t0,t2-t1,t3-t2))
          self.matrices['C'] = C
          np.save(resfile,self.matrices,allow_pickle=True)



      def compute_transition_matrix_element(self,i,j,transition_operators,outfile):

          import numpy as np
          import sys
          sys.path.append('/home/ddhawan/internship_diksha/VQE_QSE_workflow/commons/')
          from tapering import taper_auxiliary_operators
          from harvest import measure_operators
          import time

 #         n = len(self.operators)
 #         m = len(transition_operators)
          C = np.zeros((2),dtype=complex)

          # C(m,I) = < Psi(vqe) | T(m) E(I) | Psi(vqe) >

          Cij_oper = transition_operators[i]*self.operators[j]
  
          if(self.tapering):
              Cij_oper = taper_auxiliary_operators([Cij_oper],self.z2syms,self.target_sector)[0]
          C[:] = measure_operators([Cij_oper],self.psi,self.instance)[0]

          outfile.write("C matrix elements (%d,%d) computed \n" % (i+1,j+1))
           
          self.matrices['C'] = C
#           np.save(resfile,self.matrices,allow_pickle=True)
