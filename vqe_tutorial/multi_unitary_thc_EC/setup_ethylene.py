import numpy as np
import muthc
from   muthc import compute_muthc_delta

g = 10

# dat = Rob's object storing molecular information
dat = np.load('/Users/mario/Documents/GitHub/QCWare/THC/test-systems/ethylene/%d/c2h4.npz' % g)
na  = dat['nalpha']
nb  = dat['nbeta']
h0  = dat['Ecore']
h1  = dat['Hact']
h2  = dat['Iact']
no  = h1.shape[0]
I   = h2

Is   = [0.4*I.copy(),0.3*I.copy(),0.2*I.copy(),0.1*I.copy()]

# thc procedure
U_list,Z_list = muthc.compute_muthc_1(I=I,Is=Is)

dist_2,dist_oo = [],[]
for m,(U,Z) in enumerate(zip(U_list,Z_list)):
    dI_m = compute_muthc_delta(I,U,Z)
    dist_2.append(np.sqrt(np.sum(dI_m)**2))
    dist_oo.append(np.abs(compute_muthc_delta(I,U,Z)).max())
    print("eri approximation %d %.8f %.8f %s " % (m,dist_2[m],dist_oo[m],str(U.shape)))
idx = np.argmin(dist_2)

for iu in range(U_list[idx].shape[0]):
    print("z weight ",np.abs(Z_list[idx][iu,:,:]).max())

np.save('hamiltonian_ethylene-%d' % g,[no,na,nb,h0,h1,h2,U_list[idx][:,:,:],Z_list[idx][:,:,:]],allow_pickle=True)
