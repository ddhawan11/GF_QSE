import numpy as np
import muthc
from   muthc import compute_muthc_delta

# benzene or naphtalene (1 or 2)
n = 1

# dat = Rob's object storing molecular information
dat = np.load('/Users/mario/Documents/GitHub/QCWare/THC/muthc/examples/acenes/npz/%d.npz' % n)
na  = dat['nalpha']
nb  = dat['nbeta']
h0  = dat['Ecore']
h1  = dat['Hact']
h2  = dat['Iact']
no  = h1.shape[0]
I   = h2

dat2 = np.load('/Users/mario/Documents/GitHub/QCWare/THC/muthc/examples/acenes/npz/Is-%d.npz' % n)
Is   = dat2['Is']

# thc procedure
U_list,Z_list = muthc.compute_muthc_1(I=I,Is=Is)

dist_2,dist_oo = [],[]
for m,(U,Z) in enumerate(zip(U_list,Z_list)):
    dI_m = compute_muthc_delta(I,U,Z)
    dist_2.append(np.sum(dI_m)**2)
    dist_oo.append(np.abs(compute_muthc_delta(I,U,Z)).max())
    print("eri approximation %d %.8f %.8f %s " % (m,dist_2[m],dist_oo[m],str(U.shape)))
idx = np.argmin(dist_2)

np.save('hamiltonian_acene-%d' % n,[no,na,nb,h0,h1,h2,U_list[idx][:,:,:],Z_list[idx][:,:,:]],allow_pickle=True)
