import numpy as np
from scipy import linalg as LA

np.random.seed(0)

def matrix_sample(M_ave,M_std):
    r,c = M_ave.shape
    M = [[np.random.normal(M_ave[i,j],M_std[i,j]) for i in range(r)] for j in range(c)]
    M = np.array(M)
    M = (M+M.T)/2.0
    return M

def chi_0(v,v_ref):
    return np.sum((v[:,0]-v_ref[:,0])**2/(v[:,1]**2+v_ref[:,1]**2))/v.shape[0]

def chi_1(v,v_ref):
    return np.sum(v[:,1]**2/v_ref[:,1]**2)/v.shape[0]

class regularization:
     # class for the regularized solution of an eigenvalue problem HC=eSC
     def __init__(self,filename_computed,filename_exact=None,
                  nsamples=10000,regularization='tikhonov',parameters=[10.0**(-k) for k in range(1,7)[::-1]]):
         # load computed matrices and exact matrices (only for sanity check)
         self.computed_matrices = np.load(filename_computed,allow_pickle=True).item()
         if(filename_exact is not None):
            self.exact_matrices = np.load(filename_exact,allow_pickle=True).item()
         self.nsamples = nsamples
         self.regularization = regularization
         self.parameters = parameters
         self.num_parameters = len(parameters)
         self.n = self.computed_matrices['H'].shape[0]

     def compute_exact_spectrum(self):
         H,S = self.exact_matrices['H'][:,:,0],self.exact_matrices['S'][:,:,0]
         exact_spectrum = LA.eigh(H,S)[0]
         return exact_spectrum

     def regularize_overlap_matrix(self,S_ave,S_std,delta):
         # regularize overlap matrix as S+delta*I or Sqrt[S**2+delta*I**2]
         if(self.regularization=='shift'):
            S_ave = S_ave + np.eye(self.n)*delta
         if(self.regularization=='tikhonov'):
            s,U = LA.eigh(S_ave)
            s = np.array([np.sqrt(delta**2+si**2) for si in s])
            S_ave = np.einsum('im,m,jm->ij',U,s,U)
         return S_ave,S_std

     def sample_eigenproblem_solution(self,delta=0.0):
         # regularize S, sample a pair (H,S), and return the solution of the eigenproblem
         ct=True
         while(ct):
            H_ave,H_std = self.computed_matrices['H'][:,:,0],self.computed_matrices['H'][:,:,1]
            S_ave,S_std = self.computed_matrices['S'][:,:,0],self.computed_matrices['S'][:,:,1]
            S_ave,S_std = self.regularize_overlap_matrix(S_ave,S_std,delta)
            H_sample = matrix_sample(H_ave,H_std)
            S_sample = matrix_sample(S_ave,S_std)
            ct = np.min(LA.eigh(S_sample)[0])<1e-16
         return LA.eigh(H_sample,S_sample)[0]

     def compute_regularized_spectrum(self,delta=0.0):
         epsilon = np.zeros((self.n,self.nsamples))
         for i in range(self.nsamples):
             epsilon[:,i] = self.sample_eigenproblem_solution(delta)
         spectrum = np.zeros((self.n,2))
         for i in range(self.n):
             spectrum[i,:] = np.mean(epsilon[i,:]),np.std(epsilon[i,:])
         return spectrum

     def regularize(self):
         # compute the spectrum epsilon(delta) as a function of the regularization parameter
         # to measure the discrepancy between regularized and unregularized data, use the chi squared
         # \sum_i |epsilon(i,delta,average)-epsilon(i,delta=0,average)|^2/epsilon(i,delta=0,std)^2
         # to measure the reduction in standard deviations, use the chi squared
         # \sum_i |epsilon(i,delta,std)/epsilon(i,delta=0,std)|^2
         epsilon        = np.zeros((self.n,2,self.num_parameters+1))
         chi            = np.zeros((2,self.num_parameters+1))
         epsilon[:,:,0] = self.compute_regularized_spectrum(delta=1e-16)
         print(epsilon[:,:,0])
         chi[:,0]       = chi_0(epsilon[:,:,0],epsilon[:,:,0]),chi_1(epsilon[:,:,0],epsilon[:,:,0])
         print(chi[:,0])
         for i,delta_i in enumerate(self.parameters):
             epsilon[:,:,i+1] = self.compute_regularized_spectrum(delta=delta_i)
             chi[:,i+1]       = chi_0(epsilon[:,:,i+1],epsilon[:,:,0]),chi_1(epsilon[:,:,i+1],epsilon[:,:,0])
             print(delta_i)
             print(epsilon[:,:,i+1])
             print(chi[:,i+1])

         import matplotlib.pyplot as plt
         fig,pan = plt.subplots(1,2)
         pan[0].plot([0]+self.parameters,chi[0,:],label='chi(0)')
         pan[0].plot([0]+self.parameters,chi[1,:],label='chi(1)')
         pan[0].set_xlabel('delta (a.u.)')
         pan[0].set_ylabel('chi (a.u)')
         pan[0].set_title('effect of regularization')
         pan[0].legend()
         # -----
         e_ref = self.compute_exact_spectrum()
         for i in range(self.n):
             pan[1].errorbar([0]+self.parameters,epsilon[i,0,:]-e_ref[i],yerr=epsilon[i,1,:],label='eigenvalue %d'%i)
         pan[1].set_xlabel('delta (a.u.)')
         pan[1].set_ylabel('eigenvalue (Ha)')
         pan[1].set_title('spectrum')
         plt.show()

R = regularization(filename_computed = 'qse_ip_matrices_qasm.npy',
                   filename_exact = 'qse_ip_matrices.npy',
                   regularization = 'shift',
                   parameters = [0.001*k for k in range(2,31)],
                   nsamples = 1000)
R.regularize()

