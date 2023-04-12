import matplotlib.pyplot as plt
import numpy as np
import jackknife
import post_process

A = [[[1.0,0.2],[2.0,0.1]],[[3.0,0.4],[4.0,0.3]]]
A = np.asarray(A)
x_axis = []
y_axis = []
y_err  = []
new_y  = []
new_y_err = []
shots = [2000, 4000, 8000, 16000]
sample_matrix = np.zeros((30000,2,2))
for x in range(30000):
    sample_matrix[x,:,:] = post_process.sample_matrix(A, is_symmetric=False, is_hermitian=False)
init = 0
for shot in shots:
    sample_matrix1 = np.zeros((shot,2,2))
    for x in range(init, init+shot):
        sample_matrix1[x-init,:,:] = sample_matrix[x,:,:]
    mean_matrix = np.mean(sample_matrix1, axis=0)
    std_matrix  = np.std(sample_matrix1, axis=0)
    m, s, s2 = jackknife.jackknife(sample_matrix1, mean_matrix)
    x_axis.append(1.0/shot**2)
    y_axis.append(mean_matrix[0,0])
    y_err.append(std_matrix[0,0])
    new_y.append(m[0,0])
    new_y_err.append(s[0,0])
    init += shot
    

print(x_axis, y_axis, y_err)
print(new_y, new_y_err)
fig, ax = plt.subplots(1)
ax.errorbar(x_axis, [1,1,1,1], color="blue")
ax.errorbar(x_axis, new_y, yerr=new_y_err, linestyle = 'None', color="blue")
plt.show()
