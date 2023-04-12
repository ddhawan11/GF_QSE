import numpy as np
from   qiskit                 import *
from   qiskit.chemistry       import FermionicOperator
from   qiskit.aqua.operators  import Z2Symmetries
from   qiskit.aqua.operators  import WeightedPauliOperator
from   qiskit.quantum_info    import Pauli
from   qiskit.aqua.algorithms import NumPyEigensolver
class Mapper:

      def __init__(self,operator_dict,H,D):
          self.qubit_mapping       = operator_dict['qubit_mapping']
          self.two_qubit_reduction = operator_dict['two_qubit_reduction']
          self.chop                = operator_dict['chop']
          self.nelec               = list(H.ne)
          self.norbs               = H.no
          self.observables         = {}
          self.hamiltonian_terms   = {}
          self.givens              = {}
          self.log                 = ''
          self.build_observables(H)       # WPOs measured (ovlp, part n, ham)
          self.build_hamiltonian_terms(H) # THC
          self.build_givens(D)            # log(Givens)
#          print("NUMBER OPERATOR ",self.observables['N'].print_details())
#          exit()

      # -----------------------------------------------------------------------------------

      def build_observables(self,H):
          self.observables['H'] = self.convert_hamiltonian_to_qubit(H)
          nq                    = self.observables['H'].num_qubits
          self.observables['N'] = self.fermi_qubit(np.eye(2*self.norbs))
          self.observables['S'] = WeightedPauliOperator([[1.0,Pauli([False]*nq,[False]*nq)]])
          self.attach_ancilla_operators(self.observables,'H')
          self.attach_ancilla_operators(self.observables,'S')
          self.log += 'OBSERVABLES: \n'
          for k in self.observables.keys():
              self.observables[k] = self.observables[k].chop(self.chop)
              self.log += 'observable %s: \n%s----- \n' % (k,self.observables[k].print_details())

      def convert_hamiltonian_to_qubit(self,H):
          t1,t2 = self.orbital_to_spin_orbital(H.h1,0.5*H.h2)
          return self.fermi_qubit(t1,t2)

      def orbital_to_spin_orbital(self,h1,h2=None):
          n = 2*h1.shape[0]
          t1 = np.zeros((n,n))
          t2 = np.zeros((n,n,n,n))
          t1[:n//2,:n//2] = h1.copy()
          t1[n//2:,n//2:] = h1.copy()
          if(h2 is None):
             return t1,t2
          else:
             t2 = np.zeros((n,n,n,n))
             t2[:n//2,:n//2, :n//2,:n//2] += h2.copy()
             t2[:n//2,:n//2, n//2:,n//2:] += h2.copy()
             t2[n//2:,n//2:, :n//2,:n//2] += h2.copy()
             t2[n//2:,n//2:, n//2:,n//2:] += h2.copy()
             return t1,t2

      def fermi_qubit(self,t1,t2=None):
          if(t2 is None): X = FermionicOperator(t1).mapping(map_type=self.qubit_mapping)
          else:           X = FermionicOperator(t1,t2).mapping(map_type=self.qubit_mapping)
          if(self.qubit_mapping=='parity' and self.two_qubit_reduction):
             X = Z2Symmetries.two_qubit_reduction(X,self.nelec)
          return X

      def attach_ancilla_operators(self,wpo_dictionary,key):
          ancilla_x = WeightedPauliOperator([[1.0,Pauli([False],[True])]])
          ancilla_y = WeightedPauliOperator([[1.0,Pauli( [True],[True])]])
          re = WeightedPauliOperator([[c*d,Pauli(np.concatenate((p.z,q.z)),np.concatenate((p.x,q.x)))] for [c,p] in ancilla_x.paulis for [d,q] in wpo_dictionary[key].paulis])
          im = WeightedPauliOperator([[c*d,Pauli(np.concatenate((p.z,q.z)),np.concatenate((p.x,q.x)))] for [c,p] in ancilla_y.paulis for [d,q] in wpo_dictionary[key].paulis])
          wpo_dictionary['re_'+key] = re
          wpo_dictionary['im_'+key] = im
 
      # -----------------------------------------------------------------------------------

      def build_hamiltonian_terms(self,H):
          self.hamiltonian_terms['t1'] = self.number_operator(H.eta)
          self.attach_ancilla_operators(self.hamiltonian_terms,'t1')
          for i in range(H.U.shape[0]):
              # number-number operators
              self.hamiltonian_terms['u%d'%i] = self.number_number_operator(0.5*H.Z[i,:,:])
              self.attach_ancilla_operators(self.hamiltonian_terms,'u%d'%i)
          self.log += 'HAMILTONIAN TERMS: \n'
          for k in self.hamiltonian_terms.keys():
              self.hamiltonian_terms[k] = self.hamiltonian_terms[k].chop(self.chop)
              self.log += 'term %s: \n%s----- \n' % (k,self.hamiltonian_terms[k].print_details())

      def number_operator(self,eta):
          n  = len(eta)
          t1 = np.zeros((n,n))
          for i in range(n):
              t1[i,i] = eta[i]
          t1,t2   = self.orbital_to_spin_orbital(t1)
          return self.fermi_qubit(t1)

      def number_number_operator(self,zeta):
          n  = zeta.shape[0]
          t1 = np.zeros((n,n))
          for i in range(n):
              t1[i,i]  = zeta[i,i]
          for i in range(n):
              t1[i,i] -= (np.einsum('ij->i',zeta)[i]+np.einsum('ji->i',zeta)[i])
          t2 = np.zeros((n,n,n,n))
          for i in range(n):
              for j in range(n):
                  t2[i,i,j,j] = zeta[i,j]
          t1,t2   = self.orbital_to_spin_orbital(t1,t2)
          return self.fermi_qubit(t1,t2)

      # -----------------------------------------------------------------------------------

      def build_givens(self,D):
          for k in D.g.keys():
              self.givens[k] = self.givens_operators(D.g[k])
          self.log += 'GIVENS TERMS: \n'
          for k in self.givens.keys():
              self.log += 'term %s: \n' % k
              for gk in self.givens[k]:
                  self.log += gk.print_details()+'\n'
              self.log += '----- \n'

      def givens_operators(self,G_list):
          givens = []
          for r,theta in G_list:
              givens.append(self.single_givens_operator(r-1,r,theta))
              givens.append(self.single_givens_operator(self.norbs+r-1,self.norbs+r,theta))
          return givens

      def single_givens_operator(self,i,j,theta):
          t1 = np.zeros((2*self.norbs,2*self.norbs))
          t1[i,j] = -theta
          t1[j,i] =  theta
          return self.fermi_qubit(t1).chop(self.chop)

      def diagonalize_H(self,H):
          nq = self.observables['H'].num_qubits
          ee = NumPyEigensolver(self.observables['H'],k=2**nq)
          from prettytable import PrettyTable
          t = PrettyTable(['index','eigenvalue'])
          for i,ei in enumerate(np.real(ee.run()['eigenvalues'])):
              t.add_row([str(i),str(ei+H.h0)])
          return str(t)

      # -----------------------------------------------------------------------------------

      def print_details(self):
          return self.log


