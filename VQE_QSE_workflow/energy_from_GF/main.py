import numpy as np
import scipy.linalg as LA
import math
import pickle

from se import *
from gf_nonint import *

def imag_frequencies(iwmax, beta):


    omegas = [(2*n+1)*np.pi/beta for n in range(0,iwmax+1)]
    A      = np.zeros((iwmax))

    for i in range(iwmax):
        A[i]=abs(omegas[1]-omegas[0])

    return omegas, A

def calc_energy_from_LW(omegas,ghf,gf_inf,sigma2,beta,spin_factor,LW_factor):

    nmo       = ghf.shape[1]
    iwmax     = ghf.shape[0]
    energy    = 0.0
    iwmax_max = iwmax+150000
    aux       = np.zeros((iwmax_max))

    for w in range(iwmax):
        aux[w] = aux[w]+np.trace(np.dot(np.real(ghf[w,:,:]),np.real(sigma2[w,:,:])))
        aux[w] = aux[w]-np.trace(np.dot(np.imag(ghf[w,:,:]),np.imag(sigma2[w,:,:])))
        energy = energy+2.0*aux[w]

    e1=energy/beta

    print ('e1', e1)
    
    sigma2_inf        =   np.zeros((nmo,nmo,3))
    sigma2_inf[:,:,0] =  -1*np.imag(sigma2[-1,:,:])*np.imag(omegas[w])
    gf_inf[0,:,:]     =   np.eye(nmo)
    omegas,weights    =   imag_frequencies(iwmax_max, beta)

    for w in range(iwmax,iwmax_max):
        aux[w] = aux[w]-np.trace(np.dot(np.imag(1./omegas[w]*gf_inf[0,:,:]),np.imag(1./omegas[w]*sigma2_inf[0,:,:])))
        energy = energy+2.0*aux[w]
        
    return energy/beta

    
def find_coeff_of_high_freq_tail_greens_function(gf,omegas):

    iwmax = gf.shape[0]
    nmo   = gf.shape[1]

    Y_r    = np.zeros((nmo,nmo))
    Y_i    = np.zeros((nmo,nmo))
    C      = np.zeros((nmo,nmo))
    A      = np.zeros((nmo,nmo))
    D      = np.zeros((nmo,nmo))
    B      = np.zeros((nmo,nmo))
    gf_inf = np.zeros((4,nmo,nmo))

    if (iwmax >= 5000):
        iwmin = iwmax-1000
    else:
        if (iwmax < 1000):
            iwmin = iwmax-200
        else:
            iwmin = iwmax-200

    w_max     = omegas[iwmax-1]
    w_min     = omegas[iwmin]
    multip_w  = 1.0/((w_max**2)*(w_min**2))-1.0/(w_min**4)


    gf_inf[3,:,:]  = (-1.0 * np.real(gf[iwmax-1,:,:])*((w_max**2)/(w_min**2))+np.real(gf[iwmin,:,:]))/multip_w
    gf_inf[1,:,:]  = 1.0 * (np.real(gf[iwmax-1,:,:]) - gf_inf[3,:,:]/(w_max**4))*(w_max**2)



    X_i  = 1./(omegas[iwmin])**3 - 1./(omegas[iwmin] * (omegas[iwmax-1])**2)
    Y_i  = 1j*np.imag(gf[iwmin,:,:]) - 1j*np.imag(gf[iwmax-1,:,:]) * (omegas[iwmax-1]/omegas[iwmin])
    D    = Y_i/X_i

    B    = (1j * np.imag(gf[iwmax-1,:,:]) - D * 1./(omegas[iwmax-1])**3)*omegas[iwmax-1]

    gf_inf[0,:,:] = B
    gf_inf[2,:,:] = D

    return gf_inf
            
def calc_energy(gf,self_energy,inv_T,omegas,E_HF):

    gf_inf  = find_coeff_of_high_freq_tail_greens_function(gf,omegas)
    e2b     = calc_energy_from_LW(omegas,gf,gf_inf,self_energy,inv_T,2.0,0.25)
    print ("e2b",e2b)
    return E_HF+e2b


def main():
    full_Ham = False
    mol_data = np.load('h_dict.npy',allow_pickle=True).item()
    gf_fci   = np.load('qse_green_function_freq.npy',allow_pickle=True)
    nmo      = 4
    rho      = import_density(nmo,'vqe_q_uccsd_1rdm.txt')/2.0

    beta     = 100.0
    nmax     = 2000
    x_mesh   = [(2*n+1)*np.pi/beta for n in range(0,nmax+1)]

    
    gf_nonint   = get_Gnot(mol_data,rho,x_mesh,full_Ham)
    
    self_energy = np.zeros(gf_fci.shape, dtype='complex')

    
    for omega in range(nmax+1):    
        self_energy[omega,:,:] = gf_nonint[omega,:,:] - LA.inv(gf_fci[omega,:,:])
    print(self_energy[0,:,:])

    np.save('qse_self_energy_freq.npy',self_energy)

    self_energy, self_energy_full_corr, self_energy_inf, self_energy_c = \
                      set_up_self_energy_total(self_energy,nmo, x_mesh)


    
    nmo        = mol_data['n']
    S          = np.eye(nmo)
    h1         = mol_data['h1']
    h2         = mol_data['h2']


    F          = h1+(2*np.einsum('ikjl,jl->ik',h2,rho)-np.einsum('iljk,jl->ik',h2,rho))
    E_HF       = 0.5 * np.trace(np.dot(2.0 * rho, h1 + F ) )
    energy_fci = calc_energy(gf_fci,self_energy_full_corr,beta,x_mesh,E_HF)

    print(energy_fci)


if __name__ == "__main__":
    main()
