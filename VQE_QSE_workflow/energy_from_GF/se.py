import numpy as np

def find_coeff_of_high_freq_tail_self_energy(sigma,omegas):

    iwmax=sigma.shape[0]
    nmo=sigma.shape[1]

    Y_r       = np.zeros((nmo,nmo), dtype = "float128")
    Y_i       = np.zeros((nmo,nmo), dtype = "float128")
    C         = np.zeros((nmo,nmo), dtype = "float128")
    A         = np.zeros((nmo,nmo), dtype = "float128")
    D         = np.zeros((nmo,nmo), dtype = "float128")
    B         = np.zeros((nmo,nmo), dtype = "float128")

    sigma_inf = np.zeros((nmo,nmo,3), dtype = "float128")


    if (iwmax >= 5000):
        iwmin = iwmax - 1000
    else:
        if (iwmax < 1000):
            iwmin = iwmax - 200
        else:
            iwmin = iwmax - 200

    Y_r = np.real(sigma[iwmin,:,:]) - np.real(sigma[iwmax-1,:,:])
    X_r = (1./(omegas[iwmin])**2 - 1./(omegas[iwmax-1])**2)

    C   = (1./X_r) * Y_r

    A   = np.real(sigma[iwmax-1,:,:]) - (1./(omegas[iwmax-1])**2) * C
    X_i = 1./(omegas[iwmin])**3 - 1./(omegas[iwmin] * (omegas[iwmax-1])**2)
    Y_i = 1j*np.imag(sigma[iwmin,:,:]) - 1j*np.imag(sigma[iwmax-1,:,:]) * (omegas[iwmax-1]/omegas[iwmin])
    D   = Y_i/X_i
    B   = (1j*np.imag(sigma[iwmax-1,:,:])-D*1./(omegas[iwmax-1])**3)*omegas[iwmax-1]
    sigma_inf[:,:,0] = B.real
    sigma_inf[:,:,1] = C.real
    sigma_inf[:,:,2] = D.real

                
    return A,sigma_inf

def set_up_self_energy_total(self_energy_dmft, nao, omegas):

    iwmax = self_energy_dmft.shape[0]
    nmo   = self_energy_dmft.shape[1]

    self_energy = np.zeros(self_energy_dmft.shape,'complex')

    self_energy = self_energy_dmft


    self_energy_c, sigma_inf = find_coeff_of_high_freq_tail_self_energy(self_energy, omegas)

    self_energy_full_corr    = np.zeros((iwmax,nao,nao),'complex')

    for w in range(iwmax):
        self_energy_full_corr[w,0:nao,0:nao] = self_energy[w,0:nao,0:nao] - self_energy_c[0:nao,0:nao]


    return self_energy,self_energy_full_corr,sigma_inf,self_energy_c

