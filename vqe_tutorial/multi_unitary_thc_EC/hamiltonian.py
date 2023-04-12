import numpy as np

class Hamiltonian:

      def __init__(self,filename):
          [no,na,nb,h0,h1,h2,U,Z] = np.load(filename,allow_pickle=True)
          self.no = no
          self.ne = (int(na),int(nb))
          self.h0 = h0
          self.h1 = h1
          self.h2 = h2
          self.U  = U
          self.Z  = Z
          self.nt = self.U.shape[0]
          self.prepare_for_decomposition()

      def compute_HF_energy(self):
          E0    = self.h0
          rho_a = np.zeros((self.no,self.no))
          rho_b = np.zeros((self.no,self.no))
          for i in range(self.ne[0]): rho_a[i,i]=1.0
          for i in range(self.ne[1]): rho_b[i,i]=1.0
          E1 = np.einsum('pq,pq',rho_a+rho_b,self.h1)
          E2 = 0.5*np.einsum('pr,qs,prqs',rho_a+rho_b,rho_a+rho_b,self.h2) \
             - 0.5*np.einsum('ps,qr,prqs',rho_a,rho_a,self.h2) \
             - 0.5*np.einsum('ps,qr,prqs',rho_b,rho_b,self.h2)
          return E0,E1,E2

      def print_details(self):
          def print_matrix(t,M,name):
              for i in range(M.shape[0]):
                  for j in range(M.shape[1]):
                      t2.add_row([name,str(i)+','+str(j),'%.6f' % M[i,j]])
          # -----
          from prettytable import PrettyTable
          E0,E1,E2 = self.compute_HF_energy()
          t1 = PrettyTable(['property','value'])
          t1.add_row(['n',       str(self.no)])
          t1.add_row(['n(alpha)',str(self.ne[0])])
          t1.add_row(['n(beta)', str(self.ne[1])])
          t1.add_row(['E0',      str(E0)])
          t1.add_row(['E1(HF)',  str(E1)])
          t1.add_row(['E2(HF)',  str(E2)])
          t1.add_row(['E(HF)',   str(E0+E1+E2)])
          t2 = PrettyTable(['tensor','indices','entry'])
          print_matrix(t2,self.h1,'h1')
          print_matrix(t2,self.t1,'t1')
          for a in range(self.nt):
              print_matrix(t2,self.Z[a,:,:],'z_'+str(a))
          return str(t1)+'\n'+str(t2)

      def prepare_for_decomposition(self):
          from scipy import linalg as LA
          v1    = self.h1.copy() 
          for t in range(self.nt):
              v_tilde = np.zeros(v1.shape)
              z_t    = 0.5*self.Z[t,:,:]
              for p in range(v1.shape[0]):
                  v_tilde[p,p] += z_t[p,p]
                  v_tilde[p,p] -= (np.einsum('ij->i',z_t)[p]+np.einsum('ji->i',z_t)[p])
              u_t  = self.U[t,:,:]
              v1  -= np.einsum('pk,pr,rl->kl',u_t,v_tilde,u_t)
          eta,W = LA.eigh(v1)
          if(LA.det(W)<0):
             W[:,[0,1]] = W[:,[1,0]]
             eta[[0,1]] = eta[[1,0]]
          self.t1  = v1.copy()
          self.eta = eta.copy()
          self.W   = W.copy()
          for t in range(self.nt):
              detUa = LA.det(self.U[t,:,:])
              if(detUa<0):
                 self.U[t,:,[0,1]] = self.U[t,:,[1,0]]
                 self.Z[t,:,[0,1]] = self.Z[t,:,[1,0]]
                 self.Z[t,[0,1],:] = self.Z[t,[1,0],:]

