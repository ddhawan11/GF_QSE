import numpy as np
from   scipy import linalg as LA
from   utils import replace_sublist

class Decomposer:

      def __init__(self,H,decomposer_dict):
          self.H       = H
          self.dict    = decomposer_dict
          self.n_layer = H.U.shape[0]
          self.n_layer_circuit = decomposer_dict['n_layer_circuit']
          self.log             = ''

      def generate_unitaries(self):
          self.log = ''
          self.u = {}
          self.u['t1<-hf'] = LA.inv(self.H.W)
          self.u['hf<-t1'] = self.H.W
          for i in range(self.n_layer):
              self.u['u%d<-hf'%i] = LA.inv(self.H.U[i,:,:])
              self.u['hf<-u%d'%i] = self.H.U[i,:,:]
              self.u['u%d<-t1'%i] = np.dot(self.u['u%d<-hf'%i],self.u['hf<-t1'])
              self.u['t1<-u%d'%i] = np.dot(self.u['t1<-hf'],self.u['hf<-u%d'%i])
          for i in range(self.n_layer):
              for j in range(i+1,self.n_layer):
                  self.u['u%d<-u%d'%(i,j)] = np.dot(self.u['u%d<-hf'%i],self.u['hf<-u%d'%j])
                  self.u['u%d<-u%d'%(j,i)] = np.dot(self.u['u%d<-hf'%j],self.u['hf<-u%d'%i])
          self.g = {}
          for k in self.u.keys():
              self.g[k] = self.decompose_into_givens(self.u[k],k)

          #------------------------------------------------------------------------------------------#
          # split givens | post_selection | unitaries (- = gate separation, * = measurement)  | case #
          #------------------------------------------------------------------------------------------#
          # N            | N              |          U2 - inv(U2)   U1 - inv(U1)   W - inv(W) |    1 #
          # Y            | N              |          U2 - inv(U2) - U1 - inv(U1) - W - inv(W) |    2 #
          # N            | Y              | inv(V)   U2 * inv(U2)   U1 - inv(U1)   W - inv(W) |    3 #
          # Y            | Y              | inv(V) * U2 - inv(U2) - U1 - inv(U1) - W - inv(W) |    4 #
          #------------------------------------------------------------------------------------------#
          sg = self.dict['split_givens']
          ps = self.dict['post_selection']
          es = self.dict['echo_pulses']>0
          if(not sg): # cases 1 and 3
             self.circuit_instruction = ['t1<-hf','evolve_t1','u0<-t1']
             for i in range(self.n_layer_circuit):
                 self.circuit_instruction.append('evolve_u%d'%i)
                 if(i<self.n_layer_circuit-1): self.circuit_instruction.append('u%d<-u%d'%(i+1,i))
                 else:                         self.circuit_instruction.append('hf<-u%d'%i)
          if(sg):     # cases 2 and 4
             self.circuit_instruction = ['t1<-hf','evolve_t1','hf<-t1']
             for i in range(self.n_layer_circuit):
                 self.circuit_instruction.append('u%d<-hf'%i)
                 self.circuit_instruction.append('evolve_u%d'%i)
                 self.circuit_instruction.append('hf<-u%d'%i)
             self.measure_instruction = []
          #---------------------------------------------------------------------------------
          if(not ps): # cases 1 and 2
             self.measure_instruction = []
          else:       # cases 3 and 4
             self.measure_instruction = ['t1<-hf']
             for i in range(self.n_layer):
                 self.measure_instruction.append('u%d<-hf'%i)
          #---------------------------------------------------------------------------------
          if(es and sg):     # echos on and givens split
             idx_transformation = [u for u in self.circuit_instruction if '<-hf' in u]
             for iu,u in enumerate(idx_transformation): 
                 replace_sublist(self.circuit_instruction,[u],['echo_%d'%iu,u])
             idx_transformation = [u for u in self.circuit_instruction if 'hf<-' in u]
             for iu,u in enumerate(idx_transformation):
                 replace_sublist(self.circuit_instruction,[u],[u,'echo_%d_inv'%iu])
          if(es and not sg): # echos on and givens together
             idx_transformation = [u for u in self.circuit_instruction if '<-' in u]
             for iu,u in enumerate(idx_transformation):
                 replace_sublist(self.circuit_instruction,[u],['echo_%d'%iu,u])
                 if(iu<len(idx_transformation)-1):
                    v=idx_transformation[iu+1]
                    replace_sublist(self.circuit_instruction,[v],['echo_%d_inv'%iu,v])
                 else:
                    replace_sublist(self.circuit_instruction,[u],[u,'echo_%d_inv'%iu])
          #---------------------------------------------------------------------------------
          self.log += 'Circuit instruction: %s\n' % '|'.join(self.circuit_instruction[::-1])
          self.log += 'Measure instruction: %s  ' % str(self.measure_instruction)

      def decompose_into_givens(self,U_original,k):
          self.log += 'Change of basis %s ' % k 
          detU = LA.det(U_original)
          # Hn ... H1 U = 1 (inverse of U)
          U      = U_original.copy()
          nr,nc  = U.shape[0],U.shape[1]
          r,c    = nr-1,0
          G_list = []
          while(c<nc):
             while(r>c):
                [a,b] = U[r-1:r+1,c]
                if(np.abs(b)>self.dict['tolerance']):
                   cx =  a/np.sqrt(a**2+b**2)
                   sx = -b/np.sqrt(a**2+b**2)
                   G  = np.eye(nr)
                   G[r-1,r-1:r+1] = np.array([cx,-sx])
                   G[r  ,r-1:r+1] = np.array([sx, cx])
                   G_list.append([r,np.arctan2(sx,cx)])
                   U  = np.dot(G,U)
                r -= 1
             r = nr-1
             c += 1
          # U = inv(H1) ... inv(Hn) = Gn ... G1
          G_list = [[r,-t] for [r,t] in G_list[::-1]]
          U_givens = np.eye(nr)
          for r,t in G_list:
              G  = np.eye(nr)
              cx,sx          = np.cos(t),np.sin(t)
              G[r-1,r-1:r+1] = np.array([cx,-sx])
              G[r  ,r-1:r+1] = np.array([sx, cx])
              U_givens = np.dot(G,U_givens)
          self.log += ' |U-G| = %.6f ' % np.abs(U_givens-U_original).max()
          self.log += ' det(U) = %.6f, det(G) = %.6f \n' % (detU,LA.det(U_givens))
          return G_list

      def print_details(self):
          return self.log

