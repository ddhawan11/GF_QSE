import numpy as np

def jackknife(sample_matrix, mean_matrix):
    n         = mean_matrix.shape[0]
    m         = mean_matrix.shape[1]


    size_bins = 100
    num_bins  = int(sample_matrix.shape[0]/size_bins)
#    print(size_bins, num_bins)
    U_0       = mean_matrix
    U_sr      = 0
    U_si      = 0
    U         = np.zeros((num_bins,n,m), dtype = U_0.dtype)
    U_dash    = np.zeros((n,m), dtype = U_0.dtype)
    for p in range(num_bins):
        for i in range(n):
            for j in range(m):
                init   = int(p*size_bins)
                end    = int((p+1)*size_bins)
                for k in range(0,init):
                    U[p,i,j] += sample_matrix[k,i,j]
                for k in range(end,sample_matrix.shape[0]):
                    U[p,i,j] += sample_matrix[k,i,j]


#        print("U", U/(size_bins*(num_bins-1)), (size_bins*(num_bins-1)))
        U_dash += U[p,:,:]/(size_bins*(num_bins-1))

    U_dash     /= num_bins
    print("Mean from jackknife",U_dash, U_0)

    U_jackknife =  U_0 - (num_bins-1)*(U_dash-U_0)
    
#    print (U_0, U_dash)
    for p in range(num_bins):
        U_sr += (U[p,:,:].real/(size_bins*(num_bins-1)))**2
        U_si += (U[p,:,:].imag/(size_bins*(num_bins-1)))**2
#    print("U_s", U_sr, U_si)

    U_sr2 = abs(U_sr/num_bins - (U_dash.real)**2)
    U_si2 = abs(U_si/num_bins - (U_dash.imag)**2)
#    print("U_s2", U_sr2, U_si2)
#    print(np.sqrt(U_sr2), np.sqrt(U_si2))
    U_std_real = np.sqrt(num_bins-1) * np.sqrt(U_sr2)

    U_std_imag = np.sqrt(num_bins-1) * np.sqrt(U_si2)

#    print(U_std_real, U_std_imag)

    return U_jackknife, U_std_real, U_std_imag
            

