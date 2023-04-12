from subroutines import *

def sample_matrix(M,is_hermitian=False):
    r = M.shape[0]
    c = M.shape[1]
    M_sample = np.zeros((r,c))
    for i in range(r):
        for j in range(c):
            M_sample[i,j] = np.random.normal(M[i,j,0],M[i,j,1])
    if(is_hermitian):
       M_sample = (M_sample+M_sample.T)/2.0
    return M_sample

def sample_energy_differences(E_vec,E_0):
    n = len(E_vec)
    E_sample = np.zeros(n)
    for i,(mi,si) in enumerate(E_vec):
        E_sample[i] = np.random.normal(mi,si)-np.random.normal(E_0[0],E_0[1])
    return E_sample

class QSE:

      # QSE formulates an Ansatz of the form
      # \sum_m x(m) Em | Psi )
      # the coefficients x(m) are found by solving the eigenvalue equation
      # Hx = epsilon Sx
      # S(mn) = (Psi | Em*   En | Psi) <== measured on HW
      # H(mn) = (Psi | Em* H En | Psi) <== measured on HW
      # S[i,j,0] = mean value, S[i,j,1] = error bar

      def __init__(self,hamiltonian=None,psi=None,operators=None,instance=None,offset=None,threshold=None):
          self.hamiltonian = hamiltonian
          self.psi         = psi
          self.operators   = operators
          self.instance    = instance
          self.offset      = offset
          self.threshold   = threshold
          self.matrices    = {'S':None,'H':None}
          self.result      = {}

      def get_vqe_energy(self):
          return self.measure(self.hamiltonian,[self.psi],self.instance)[0]

      def measure(self,operator,wfn_circuits,instance):
          circuits = []
          for idx,wfn_circuit in enumerate(wfn_circuits):
              circuit = operator.construct_evaluation_circuit(
                        wave_function               = wfn_circuit,
                        statevector_mode            = instance.is_statevector,
                        use_simulator_snapshot_mode = False,
                        circuit_name_prefix         = 'wfn_'+str(idx))
              circuits.append(circuit)
          if circuits:
              to_be_simulated_circuits = \
                  functools.reduce(lambda x, y: x + y, [c for c in circuits if c is not None])
              result = instance.execute(to_be_simulated_circuits)
          # ---
          results_list = []
          for idx,wfn_circuit in enumerate(wfn_circuits):
              mean,std = operator.evaluate_with_result(
                         result = result,statevector_mode = instance.is_statevector,
                         use_simulator_snapshot_mode = False,
                         circuit_name_prefix         = 'wfn_'+str(idx))
              results_list.append([np.real(mean),np.abs(std)])
          # ---
          return results_list

      def construct_qse_matrices(self):
          n = len(self.operators)
          H = np.zeros((n,n,2))
          S = np.zeros((n,n,2))
          for i,Pi in enumerate(self.operators):
              for j,Pj in enumerate(self.operators):
                  print("qse matrices construction, row, col, tot = ",i,j,n)
                  if(i<=j):
                     t0 = time.time()
                     Sij_oper = adjoint(Pi)*Pj                     # S(i,j) = (Psi|dag(Pi)   Pj|Psi)
                     t1 = time.time()
                     Hij_oper = adjoint(Pi)*(self.hamiltonian*Pj)  # H(i,j) = (Psi|dag(Pi) H Pj|Psi)
                     t2 = time.time()
                     S[i,j,:] = self.measure(Sij_oper,[self.psi],self.instance)[0]
                     t3 = time.time()
                     H[i,j,:] = self.measure(Hij_oper,[self.psi],self.instance)[0]
                     t4 = time.time()
                     S[j,i,:] = S[i,j,:]
                     H[j,i,:] = H[i,j,:]
                     t5 = time.time()
                     print("indices ij ",i,j," --- times [s] for Sij prep, Hij prep, Sij meas, Hij meas ",t1-t0,t2-t1,t3-t2,t4-t3)
          self.matrices['S'] = S
          self.matrices['H'] = H 

      def stochastic_diagonalization(self,H,S=None,ntry=10000,stabilization_factor=0,target_trace=None):
          n = H.shape[0]
          # -----
          #def sample_matrix(M):
          #    M_sample = np.zeros((n,n))
          #    for i in range(n):
          #        for j in range(n):
          #            M_sample[i,j] = np.random.normal(M[i,j,0],M[i,j,1])
          #    M_sample = (M_sample+M_sample.T)/2.0
          #    return M_sample
          # -----
          def stabilize(M,target_trace,stabilization_factor):
              M += stabilization_factor*np.eye(n)
              if(target_trace is not None):
                 M *= (target_trace/np.trace(M))
              return M
          # -----
          eig_val_ave = np.zeros(n)
          eig_val_std = np.zeros(n)
          eig_vec_ave = np.zeros((n,n))
          eig_vec_std = np.zeros((n,n))
          for itry in range(ntry):
              H_sample = sample_matrix(H,is_hermitian=True)
              if(S is not None):
                 S_sample = sample_matrix(S,is_hermitian=True)
                 S_sample = stabilize(S_sample,target_trace,stabilization_factor) 
              else:
                 H_sample = stabilize(H_sample,target_trace,stabilization_factor)
              if(S is not None): 
                 eta,V = safe_eigh(H_sample,S_sample,lindep=self.threshold)[:2]
              else:
                 eta,V = LA.eigh(H_sample)
              eig_val_ave += eta
              eig_val_std += eta**2
              eig_vec_ave += V
              eig_vec_std += V**2
          eig_val_ave /= ntry
          eig_val_std /= ntry
          eig_vec_ave /= ntry
          eig_vec_std /= ntry
          eig_val_std -= eig_val_ave**2
          eig_vec_std -= eig_vec_ave**2
          print("eigenvecs, average ",eig_vec_ave)
          print("eigenvecs, std     ",np.sqrt(np.abs(eig_vec_std)))
          return eig_val_ave,np.sqrt(np.abs(eig_val_std)),eig_vec_ave,np.sqrt(np.abs(eig_vec_std))

      def run(self,fname,trace_of_s=None,stabilization_factor=0.0,ntry=10000):
          result = {}
          self.construct_qse_matrices()
          result['overlap']     = self.matrices['S']
          result['hamiltonian'] = self.matrices['H']
          from pyscf import lib
          from scipy import linalg as LA
          n    = self.matrices['S'][:,:,0].shape[0]
          sgm_ave,sgm_std,_,_ = self.stochastic_diagonalization(self.matrices['S'],stabilization_factor=stabilization_factor,target_trace=trace_of_s,ntry=ntry)
          for m,s in zip(sgm_ave,sgm_std):
              print("eigenvalues of QSE overlap matrix ",m,s)
          sgm_ave,sgm_std,x_ave,x_std = self.stochastic_diagonalization(self.matrices['H'],self.matrices['S'],stabilization_factor=stabilization_factor,target_trace=trace_of_s)
          #exit()
          #n        = self.matrices['S'][:,:,0].shape[0]
          #w,X,seig = safe_eigh(self.matrices['H'][:,:,0],
          #                     self.matrices['S'][:,:,0]+stabilization_factor*np.eye(n),lindep=self.threshold)
          sgm_ave += self.offset
          for m,s in zip(sgm_ave,sgm_std):
              print("eigenvalues of QSE schrodinger eq ",m,s)
          result['eigenvalues']  = [(m,s) for m,s in zip(sgm_ave,sgm_std)]
          result['eigenvectors'] = [(x_ave[:,i],x_std[:,i]) for i in range(x_std.shape[1])]
          self.result = result
          np.save(fname,result,allow_pickle=True)
          return result

      def print_information(self,result):
          self.result = result
          for m,Em in enumerate(self.operators):
              print("QSE operator ",m)
              print(Em.print_details())
          print("QSE matrices")
          H,S = result['hamiltonian'][:,:,:],result['overlap'][:,:,:]
          n   = H.shape[0]
          for i in range(n):
              for j in range(n):
                  print("r,c,S[r,c] = ",i,j,S[i,j,0]," +/- ",S[i,j,1])
          for i in range(n):
              for j in range(n):
                  print("r,c,H[r,c] = ",i,j,H[i,j,0]," +/- ",H[i,j,1])
          for i,(wi,xi) in enumerate(zip(result['eigenvalues'],result['eigenvectors'])):
              print("eigenvalue   ",i,wi[0]," +/- ",wi[1])
              print("components   ")
              for m,(xim,xis) in enumerate(zip(xi[0],xi[1])):
                  print(m,xim,xis)


      def diagonalize_H(self,mol=None):
          from qiskit.aqua.algorithms import ExactEigensolver
          ee         = ExactEigensolver(self.hamiltonian,k=2**self.hamiltonian.num_qubits)
          curr_value = ee.run()['energies']
          print("Hamiltonian eigen-energies      ",curr_value)
          if(mol is not None): print("Hamiltonian Hartree-Fock energy ",mol.hf_energy)

      def build_excitation_operator(self,i):
          # build the operator E(i) going from GS to excited state i
          # E(i) = \sum_j X[i,j] BASIS[j]
          xim,xis = self.result['eigenvectors'][i]
          for j,oj in enumerate(self.operators):
              if(j==0): excitation_operator  = xim[j]*self.operators[j]
              else:     excitation_operator += xim[j]*self.operators[j]
          return excitation_operator

      def compute_matrix_element(self,X,i,j,ipea):
          Ei = self.build_excitation_operator(i)
          Ej = self.build_excitation_operator(j)
          if(ipea): O = adjoint(Ei)*X
          else:     O = adjoint(Ei)*X*Ej
          return self.measure(O,[self.psi],self.instance)[0]

      def compute_matrix_elements(self,X,ipea):
          n_psi = len(self.result['eigenvalues'])
          c = np.zeros((n_psi,2)) #,dtype=complex)
          for i in range(n_psi):
              c[i,:] = self.compute_matrix_element(X,i,0,ipea)
          return c

      def measure_matrix_elements(self,operators=[],fname='',ipea=False):
          n_oper   = len(operators)
          n_states = len(self.result['eigenvalues'])
          Psi_mu_A_Psi_0 = np.zeros((n_states,n_oper,2)) #,dtype=complex)
          for i,A in enumerate(operators):
              Psi_mu_A_Psi_0[:,i,:] = self.compute_matrix_elements(A,ipea)
              for j in range(n_states):
                  print("<excited(%d)|a(%d)|GS> = %.6f +/- %.6f " % (j,i,Psi_mu_A_Psi_0[j,i,0],Psi_mu_A_Psi_0[j,i,1]))
          np.save(fname,Psi_mu_A_Psi_0,allow_pickle=True)
          return Psi_mu_A_Psi_0

      def measure_correlation_function(self,A_list,options={},ipea=False,matrix_elements=None,vqe_energy=None,samples=10):
          if(matrix_elements is None): 
             Psi_mu_A_Psi_0 = self.measure_matrix_elements(A_list,'tmp.npy',ipea)
          else:
             Psi_mu_A_Psi_0 = matrix_elements
          if(ipea):
             if(vqe_energy is None): E0 = self.get_vqe_energy()[0]
             else:                   E0 = vqe_energy
          else:
             E0 = self.result['eigenvalues'][0]
          # ------
          if(options['kind']=='imaginary_time'):
             def fun(w,t):
                 return np.exp(-t*w)
          elif(options['kind']=='real_time'):
             def fun(w,t):
                 return np.exp(-1j*t*w)
          elif(options['kind']=='imaginary_frequency_ip'):
             def fun(dE,wn):
                 return 1.0/(1j*wn+dE)
          elif(options['kind']=='imaginary_frequency_ea'):
             def fun(dE,wn):
                 return 1.0/(1j*wn-dE)
          else:
             assert(False)
          # ------
          n = len(self.result['eigenvalues'])
          if('time' in options['kind']):
             t_mesh = [options['dt']*i for i in range(options['nt']+1)]
             n_oper = len(Psi_mu_A_Psi_0)
             f_mesh = np.zeros((len(t_mesh),n_oper,n_oper,2))
             for i_sample in range(samples):
                 matrix_elements = sample_matrix(Psi_mu_A_Psi_0,is_hermitian=False)
                 energies        = sample_energy_differences(self.result['eigenvalues'],E0)
                 f_sample = np.zeros((len(t_mesh),n_oper,n_oper))
                 for jt,t in enumerate(t_mesh):
                     # G(ij,t) = \sum_k \langle <Psi_k|O(i)|Psi_0> <Psi_k|O(j)|Psi_0> f(E[k],t) 
                     for i in range(n_oper):      # i operator
                         for j in range(n_oper):  # j operator
                             f_ij_k = 0
                             for k in range(n):   # eigenstate
                                 f_ij_k += matrix_elements[k,i]*matrix_elements[k,j]*fun(energies[k],t)
                             f_sample[jt,i,j] = f_ij_k
                 f_mesh[:,:,:,0] += f_sample
                 f_mesh[:,:,:,1] += f_sample**2
             f_mesh[:,:,:,0] /= samples
             f_mesh[:,:,:,1] /= samples
             f_mesh[:,:,:,1]  = f_mesh[:,:,:,1] - f_mesh[:,:,:,0]**2
             f_mesh[:,:,:,1]  = np.sqrt(np.abs(f_mesh[:,:,:,1]))
             for ti,fi,dfi in zip(t_mesh,f_mesh[:,:,:,0],f_mesh[:,:,:,1]):
                 for i in range(n_oper):      # i operator
                     for j in range(n_oper):
                         print("G(%d,%d,%.6f) = %.6f +/- %.6f " % (i,j,ti,fi[i,j],dfi[i,j]))
                 print("---")
             return t_mesh,f_mesh[:,:,:,0],f_mesh[:,:,:,1]
          elif(options['kind']=='imaginary_frequency_ip' or options['kind']=='imaginary_frequency_ea'):
             w_mesh = [(2*n+1)*np.pi/options['beta'] for n in range(-options['n_max'],options['n_max']+1)]
             n_oper = len(Psi_mu_A_Psi_0)
             f_mesh = np.zeros((len(w_mesh),n_oper,n_oper,2,2)) # frequency,orbital i,orbital j,real and imag, mean value, std
             for i_sample in range(samples):
                 matrix_elements = sample_matrix(Psi_mu_A_Psi_0,is_hermitian=False)
                 energies        = sample_energy_differences(self.result['eigenvalues'],E0)
                 f_sample = np.zeros((len(f_mesh),n_oper,n_oper),dtype=complex)
                 for jw,w in enumerate(w_mesh):
                     # G(ij,i omega_n) = \sum_k \langle <Psi_k|O(i)|Psi_0> <Psi_k|O(j)|Psi_0> f(E[k],i omega_n)
                     for i in range(n_oper):      # i operator
                         for j in range(n_oper):  # j operator
                             f_ij_k = 0
                             for k in range(n):   # eigenstate
                                 f_ij_k += matrix_elements[k,i]*matrix_elements[k,j]*fun(energies[k],w) 
                             f_sample[jw,i,j] = f_ij_k # fill green function at img frequency jw, entries ij, with f_ij_k
                 f_mesh[:,:,:,0,0] += np.real(f_sample)
                 f_mesh[:,:,:,0,1] += np.real(f_sample)**2
                 f_mesh[:,:,:,1,0] += np.imag(f_sample)
                 f_mesh[:,:,:,1,1] += np.imag(f_sample)**2
             for l in range(2):
                 f_mesh[:,:,:,l,:] /= samples
                 f_mesh[:,:,:,l,1]  = f_mesh[:,:,:,l,1] - f_mesh[:,:,:,l,0]**2
                 f_mesh[:,:,:,l,1]  = np.sqrt(np.abs(f_mesh[:,:,:,l,1]))
             return w_mesh,f_mesh[:,:,:,:,0],f_mesh[:,:,:,:,1] # frequencies, gf real/imag avg, gf real/imag std
 
