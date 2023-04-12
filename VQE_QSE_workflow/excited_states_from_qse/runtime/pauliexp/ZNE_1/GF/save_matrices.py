import numpy as np

ip_matrix1 = np.load("QSE_matrixelements_ip.npy", allow_pickle=True).item()

ip_matrix2 = np.load("QSE_matrixelements_ip_1.npy", allow_pickle=True).item()


S_ip = ip_matrix1["S"]
S_ip[0,1] = S_ip[1,0]
S_ip[1,1] = ip_matrix2["S"][1,1]
S_ip[:,:,1] = np.sqrt(np.abs(S_ip[:,:,1]))/100

print(S_ip)

H_ip = ip_matrix1["H"]
H_ip[0,1] = H_ip[1,0]
H_ip[1,1] = ip_matrix2["H"][1,1]
H_ip[:,:,1] = np.sqrt(np.abs(H_ip[:,:,1]))/100

print(H_ip)

ip_transitionmatrix = np.load("QSE_transitionmatrixelements_ip.npy", allow_pickle=True).item()

C = ip_transitionmatrix["C"]

C[:,:,1] = np.sqrt(np.abs(C[:,:,1]))/100

np.save("qse_ip_matrices.npy", {"H": H_ip, "S": S_ip, "C":C}, allow_pickle=True)


ea_matrix1 = np.load("QSE_matrixelements_ea.npy", allow_pickle=True).item()
S_ea = ea_matrix1["S"]
S_ea[:,:,1] = np.sqrt(np.abs(S_ea[:,:,1]))/100

H_ea = ea_matrix1["H"]
H_ea[:,:,1] = np.sqrt(np.abs(H_ea[:,:,1]))/100

ea_transitionmatrix = np.load("QSE_transitionmatrixelements_ea.npy", allow_pickle=True).item()

C = ea_transitionmatrix["C"]
C[:,:,1] = np.sqrt(np.abs(C[:,:,1]))/100
print(C)

np.save("qse_ea_matrices.npy", {"H": H_ea, "S": S_ea, "C":C}, allow_pickle=True)

