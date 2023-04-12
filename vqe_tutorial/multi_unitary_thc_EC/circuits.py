import numpy  as np
from   scipy  import linalg as LA
from   qiskit import *
from   utils  import time_evolution_circuit,measure_operators,givens

class QFD_THC_circuits:

      def __init__(self,H,D,M,circuit_dict):
          self.H    = H
          self.D    = D
          self.M    = M
          self.dict = circuit_dict
          self.log  = ''
          self.circuitry = {}
          self.res       = {}

      def print_details(self):
          return self.log

      def initial_circuit(self):
          bs,theta = self.dict['initial_circuit']
          self.np  = len(bs[0])
          self.nq  = self.np+1
          qr       = QuantumRegister(self.nq,"q")
          cr       = ClassicalRegister(self.nq,"c")
          circuit  = QuantumCircuit(qr)
          circuit.h(0)
          if(len(bs)==1):
             for i,xi in enumerate(bs[0]):
                 if(xi==1): circuit.x(1+i)
          if(len(bs)==2):
             x,y = bs
             S   = [i for i in range(self.np) if x[i]!=y[i]]
             i0  = S[0]
             if(x[i0]==1): x,y = y,x
             for i,xi in enumerate(x):
                 if(x[i]==1): circuit.x(1+i)
             circuit.ry(2*theta,1+i0)
             for i in S[1:]:
                 circuit.cx(1+i0,1+i)
          return circuit.copy()

      def prepare_circuits(self):
          psi_0     = self.initial_circuit()
          for i_bra in range(self.dict['num_steps']+1):
              for i_ket in range(i_bra,self.dict['num_steps']+1):
                  for i_echo in range(max(self.D.dict['echo_pulses'],1)):
                      psi_bke = time_evolution_circuit(psi_0,i_bra,i_ket,i_echo,self.H,self.D,self.M,self.dict)
                      for label,circuit,phase in psi_bke:
                          if(self.D.dict['post_selection']): # attach a final givens transformation
                             for m in self.D.measure_instruction:
                                 phase_m   = phase
                                 circuit_m = circuit.copy()
                                 givens(circuit_m,self.M.givens[m])
                                 circuit_m.rz(phase_m,0)
                                 self.circuitry[label+'|ps_'+m] = circuit_m
                          else:
                             circuit.rz(phase,0)
                             self.circuitry[label] = circuit
          for k in self.circuitry.keys():
              self.log += 'operations %s\ndepth: %s\n' % (str(self.circuitry[k].count_ops()),str(self.circuitry[k].depth()))

      def perform_measurements(self):
          rf = self.dict['dump_data']
          if(self.D.dict['post_selection']): # measure hamiltonian components in their eigenbases 
             self.res = measure_operators({x:self.M.hamiltonian_terms[x] for x in self.M.hamiltonian_terms.keys() if 're_' in x or 'im_' in x},
                                      self.circuitry,self.dict['instance'],post_select=True,rf=rf,particle_number=self.H.ne,ps_fun=self.dict['post_selection_function'],
                                      ps_mapping=self.M.qubit_mapping,ps_tqr=self.M.two_qubit_reduction)
          else:                              # measure hamiltonian as WPO
             self.res = measure_operators({x:self.M.observables[x]       for x in self.M.observables.keys()       if 're_' in x or 'im_' in x},
                                      self.circuitry,self.dict['instance'],post_select=False,rf=rf)

      def print_results(self,message):
          self.log += message+'\n'
          for k in sorted(self.res.keys()):
              self.log += '%s %s\n' % (k,str(self.res[k]))

      def run(self,post_process):
          self.prepare_circuits()
          self.perform_measurements(); self.print_results('pre-processed')
          self.post_process_results(); self.print_results('post-processed')
          self.finalize(post_process)
 
      def post_process_results(self):
          self.echo_average()
          self.print_results('after-echo')
          if(self.D.dict['post_selection']): self.recompose_H()

      def echo_average(self):
          def transform_key(k):
              k_list = k.split('|')
              k_list.insert(0,k_list.pop(3))
              return '|'.join(k_list)
          self.res = {transform_key(k):self.res[k] for k in self.res.keys()}
          circuits = list(set(['|'.join(k.split('|')[1:]) for k in self.res.keys()]))
          echo     = list(set([k.split('|')[0]            for k in self.res.keys()]))
          n_echo   = len(echo)
          res = {c:[0,0] for c in circuits}
          for c in circuits:
              for i in range(n_echo):
                  res[c][0] += self.res['echo_%d|%s' % (i,c)][0]
                  res[c][1] += self.res['echo_%d|%s' % (i,c)][1]**2
              res[c] = (res[c][0]/n_echo,np.sqrt(res[c][1]/n_echo**2))
          self.res = res

      def recompose_H(self):
          circuits  = list(set(['|'.join(k.split('|')[1:]) for k in self.res.keys()]))
          operators = list(set([k.split('|')[0][3:]        for k in self.res.keys()]))
          operators = [x for x in operators if x!='S']
          res = {}
          for c in circuits:
              res['re_H|'+c]=[0,0]
              res['im_H|'+c]=[0,0]
              res['re_S|'+c]=self.res['re_S|'+c]
              res['im_S|'+c]=self.res['im_S|'+c]
          for c in circuits:
              for o in operators:
                  res['re_H|'+c][0] += self.res['re_%s|%s'%(o,c)][0]
                  res['im_H|'+c][0] += self.res['im_%s|%s'%(o,c)][0]
                  res['re_H|'+c][1] += self.res['re_%s|%s'%(o,c)][1]**2
                  res['im_H|'+c][1] += self.res['im_%s|%s'%(o,c)][1]**2
              res['re_H|'+c][1] = np.sqrt(res['re_H|'+c][1])
              res['im_H|'+c][1] = np.sqrt(res['im_H|'+c][1])
          self.res = res

      def finalize(self,instruction):
          self.solve_eigenvalue_problem(instruction)

      def solve_eigenvalue_problem(self,instruction):
          np.random.seed(instruction['seed'])
          from utils import safe_eigh
          n   = self.dict['num_steps']+1
          eps = np.zeros((n,instruction['ntry']))
          det = np.zeros((n,instruction['ntry']))
          nei = 10**8
          for itry in range(instruction['ntry']):
              H = np.zeros((n,n),dtype=complex)
              S = np.zeros((n,n),dtype=complex)
              for i in range(n):
                  for j in range(i,n):
                      mx,sx = self.res['re_S|t_bra_%d|t_ket_%d' % (i,j)]
                      my,sy = self.res['im_S|t_bra_%d|t_ket_%d' % (i,j)]
                      S[i,j] = np.random.normal(mx,sx,1)+1j*np.random.normal(my,sy,1)
                      S[j,i] = np.conj(S[i,j])
                      # -----
                      mx,sx = self.res['re_H|t_bra_%d|t_ket_%d' % (i,j)]
                      my,sy = self.res['im_H|t_bra_%d|t_ket_%d' % (i,j)]
                      H[i,j] = np.random.normal(mx,sx,1)+1j*np.random.normal(my,sy,1)
                      H[j,i] = np.conj(H[i,j])
              S = (S+np.conj(S.T))/2.0
              H = (H+np.conj(H.T))/2.0
              eps_try,_,seig = safe_eigh(H,S,instruction['threshold'])
              if(len(eps_try)<nei): nei = len(eps_try)
              eps[:nei,itry] = eps_try[:nei]
              det[:,itry]    = seig
          final_results = {}
          for k in range(nei):
              final_results['energy_'+str(k)] = np.mean(eps[k,:])+self.H.h0,np.std(eps[k,:])
          for k in range(n):
              final_results['eig(S)_'+str(k)] = np.mean(det[k,:]),np.std(det[k,:])
          self.res = final_results
          self.print_results('final results')
          import pickle
          with open(instruction['results_file'],'wb') as output:
               pickle.dump(final_results,output,pickle.HIGHEST_PROTOCOL)

