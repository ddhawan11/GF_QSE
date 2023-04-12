import matplotlib.pyplot as plt
import numpy as np


beta   = 100.0
nmax   = 100
x_mesh = [(2*n+1)*np.pi/beta for n in range(0,nmax+1)]

green_full = np.load('qse_green_function_freq_sv.npy',allow_pickle=True)

green_qse = np.load('qse_green_function_freq_qasm.npy',allow_pickle=True)
green_qse_ea1 = np.load('qse_green_function_freq_qasm_ea1.npy',allow_pickle=True)
green_full_ea1 = np.load('qse_green_function_freq_sv_ea1.npy',allow_pickle=True)
green_qse_wt = np.load('qse_green_function_freq_qasm_wt.npy',allow_pickle=True)


plt.plot(x_mesh,np.imag(green_full[:,1,1,0]),label='G(%d,%d)_sv'%(1,1))
plt.plot(x_mesh,np.imag(green_qse[:,1,1,0]),label='G(%d,%d)_qasm'%(1,1))
plt.plot(x_mesh,np.imag(green_qse_ea1[:,1,1,0]),label='G(%d,%d)_qasm_ea1'%(1,1))
plt.plot(x_mesh,np.imag(green_qse_wt[:,1,1,0]),label='G(%d,%d)_qasm_wt'%(1,1))
plt.plot(x_mesh,np.imag(green_full_ea1[:,1,1,0]),label='G(%d,%d)_sv_ea1'%(1,1))
plt.legend()
plt.show()
