import numpy as np
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
import matplotlib.pyplot as plt
fig,axs = plt.subplots(2)
plt.subplots_adjust(wspace=0.4,hspace=0.4)

nmo   = 2
iwmax = 2000
p = 0
q = 50
omega, GF = read_gf_T("green_function_fci.txt", nmo, iwmax)
for i in range(nmo):
     for j in range(nmo):
         x = omega
         y = GF[:,i,j].real
         axs[0].plot(x[p:q],y[p:q],label='re G(FCI)(%d,%d)'%(i,j))
         y = GF[:,i,j].imag
         axs[1].plot(x[p:q],y[p:q],label='Im G(FCI)(%d,%d)'%(i,j))


green_full = np.load('qse_green_function_freq.npy',allow_pickle=True)
print (green_full.shape)
#green_ea = np.load('qse_greens_functions_ea.npy',allow_pickle=True).item()
print (green_full)
a = 2

beta   = 100.0
nmax   = 2000
x_mesh = [(2*n+1)*np.pi/beta for n in range(0,nmax+1)]
nx     = len(x_mesh)

for i in range(a):
     for j in range(a):
#         if (i==0 and j==1):
             x  = x_mesh
             y  = green_full[:,i,j].real
             axs[0].errorbar(x[p:q],y[p:q],label='re G(QSE)(%d,%d)'%(i,j))
             y  = green_full[:,i,j].imag
             axs[1].errorbar(x[p:q],y[p:q],label='im G(QSE)(%d,%d)'%(i,j))
axs[0].set_title('imaginary-freq Green function, real part')
axs[1].set_title('imaginary-freq Green function, imag part')
# ---
axs[0].legend()
axs[1].legend()
# ---
plt.show()

