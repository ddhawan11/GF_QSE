import numpy as np
import h5py as hdf5

##############
def read_gf_T(filename, nmo, iwmax):

    file1 = open(filename, "r")
    GF    = np.zeros((iwmax,nmo,nmo), dtype='complex')
    omega = np.zeros(iwmax)
    w = -1
    for a, line in enumerate(file1.readlines()):
        toks = line.split()
#        w    += 1#int(toks[0])
        p    = int(toks[1])
        q    = int(toks[2])
        if (p==0 and q==0):
            w += 1
        if (len(toks) == 5):           
            omega[w]  = complex(toks[0]).imag
            GF[w,p,q] = float(toks[3])+float(toks[4])*1j
        elif(len(toks) == 4):
            omega[w]  = complex(toks[0]).imag
            GF[w,p,q] = complex(toks[3])

    return omega, GF

##############
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

fig,axs = plt.subplots(1)
#fig.add_gridspec(2, 2)
plt.subplots_adjust(wspace=0.5,hspace=0.5)

with hdf5.File("sim.h5", "r") as gf:
    gf_fci = gf["results/G_ij_omega/data"][...]
    omega =  gf["results/G_ij_omega/mesh/1/points"][...]

green_sv = np.load('qse_gf_sv.npy',allow_pickle=True)
green_qasm = np.load('qse_green_function_freq_jk.npy',allow_pickle=True)
#gf_onsite = np.load('qse_gf_onsite.npy',allow_pickle=True)
#gf_param2 = np.load('qse_gf_param2.npy',allow_pickle=True)
a = 2
beta   = 100.0

nx     = len(omega)
p      = 40
q      = 95
fig.set_size_inches([8,8])
left, bottom, width, height = [0.65, 0.55, 0.25, 0.25]
for i in range(a):
     for j in range(a):
         if ((i==0 and j==0)):# or (i==1 and j==1)):
             x  = omega
             y  = green_sv[0,:,i,j].imag
             print(x.shape, y.shape)
             axs.errorbar(x[p:q],y[p:q],label='Statevector')
             
             y  = green_qasm[3,:,i,j,0].imag
             yerr = green_qasm[3,:,i,j,2]
             print(x.shape, y.shape)             
             axs.errorbar(x[p:q],y[p:q],yerr=yerr[p:q], label='QASM')

axs.set_title('Imag-freq Green function(G$_{00}(\iota\omega)$), Imag part')             
#axs[1].set_title('Imag-freq Green function(G$_{00}(\iota\omega)$), imag part')
#axs[1,0].set_title('imag-freq Green function(G$_{00}$)\n real part')             
#axs[1,1].set_title('imag-freq Green function(G$_{00}$)\n imag part')
# ---
axs.legend(loc=2)
#axs[1].legend(loc=2)
#axs[1,0].legend()
#axs[1,1].legend()

# ---
axs.set_ylabel("Im[$G_{00}$(i$\omega$)]")
#axs[1].set_ylabel("Im[$G_{00}$(i$\omega$)]")
axs.set_xlabel("Frequency")

plt.savefig("H2_G00_imag_jk.png")

