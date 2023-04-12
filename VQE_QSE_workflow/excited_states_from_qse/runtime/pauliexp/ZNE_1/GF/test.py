import pickle
import matplotlib.pyplot as plt
import numpy as np

fmesh_file = open("f_mesh.obj", "rb")
fmesh_file_pp = open("f_mesh_E0_rand.obj", "rb")
beta   = 100.0
nmax   = 100
x_mesh = [(2*n+1)*np.pi/beta for n in range(0,nmax+1)]
f_mesh = pickle.load(fmesh_file)
f_mesh_pp = pickle.load(fmesh_file_pp)
green_ip = np.load('qse_gf_ip.npy',allow_pickle=True)
fig, ax = plt.subplots(1)
ax.errorbar(x_mesh, green_ip[:,0,0], label= 'G(FCI,0,0)')
ax.errorbar(x_mesh, f_mesh[:,0,0,0].real, yerr=f_mesh[:,0,0,1].real, label='G(IP,0,0)(With Random E0)')
ax.errorbar(x_mesh, f_mesh_pp[:,0,0,0].real, yerr=f_mesh[:,0,0,1].real, label='G(IP,0,0)')
plt.legend()
plt.show()
