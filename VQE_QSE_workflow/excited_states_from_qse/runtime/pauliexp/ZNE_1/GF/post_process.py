import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import jackknife
import pickle

def sample_matrix(M,is_hermitian=True, is_symmetric=True):
    r = M.shape[0]
    c = M.shape[1]
    M_sample = np.zeros((r,c), dtype=M.dtype)

    if (is_symmetric):
        for i in range(r):
            for j in range(i+1):
                M_sample[i,j] = np.random.normal(M[i,j,0],M[i,j,1])
                M_sample[j,i] = M_sample[i,j]
    else:
        for i in range(r):
            for j in range(c):
                M_sample[i,j] = np.random.normal(M[i,j,0],M[i,j,1])
    if(is_hermitian):
       M_sample = (M_sample+M_sample.T)/2.0
    
    return M_sample

def sample_energy_differences(E_vec,E_0, E_std):
    n = len(E_vec)
    E_mu = np.zeros(n)
    E_sample = np.zeros(n)
    for i in range(n):
#        rand_E0     = np.random.normal(E_0, E_std)
        E_sample[i] = E_vec[i] - E_0
    return E_sample
#    for i,(mi,si) in enumerate(E_vec):
#        rand_num    = np.random.normal(mi,si)
#        rand_E0     = np.random.normal(E_0, E_std)
#        E_sample[i] = rand_num-rand_E0
#        E_mu[i]        = rand_num
#    return E_sample, E_mu


def safe_eigh(h,s,lindep=1e-15):
    from   scipy     import linalg as LA
    import numpy     as np
    from   functools import reduce
    seig,t = LA.eigh(s)
    if(seig[0]<lindep):
        idx  = seig >= lindep
        t    = t[:,idx]*(1/np.sqrt(seig[idx]))
        heff = reduce(np.dot,(t.T.conj(),h,t))
        w,v  = LA.eigh(heff)
        v    = np.dot(t,v)
    else:
        w,v  = LA.eigh(h, s)
    return w,v

def compute_qse_eigenpairs(vqe_res,qse_res,outfile,shots=1000,lindep=1e-15):
    import numpy as np
    num_elec = 1
    S        = qse_res['S']
    H        = qse_res['H']
    eta,V    = safe_eigh(H[:,:,0],S[:,:,0],lindep=lindep)
    n     = S.shape[0]

    print("S", S)
    print("H", H)
    print("C", qse_res['C'])
    #######################
    eta_samples = np.zeros((shots,n),dtype=eta.dtype)
    V_samples   = np.zeros((shots,n,n),dtype=V.dtype)
    ########################

    eig_val     = np.zeros((n,2))
    eig_vec     = np.zeros((n,n,2))

    
    for i in range(shots):
        S_ne     = True
        H_sample = sample_matrix(H, is_hermitian=True)
        while(S_ne):
            S_sample = sample_matrix(S, is_hermitian=True)
            if (abs(np.trace(S_sample)-num_elec) <= 0.01):
                S_ne = False

            

        eta,V = safe_eigh(H_sample,S_sample,lindep=lindep)
        eta_samples[i,:] = eta
        V_samples[i,:,:]   = V

        eig_val[:,0] += eta
        eig_val[:,1] += eta**2
        eig_vec[:,:,0] += V
        eig_vec[:,:,1] += V**2

    eig_val[:,0] /= shots
    eig_val[:,1] /= shots
    eig_vec[:,:,0] /= shots
    eig_vec[:,:,1] /= shots
    eig_val[:,1] -= eig_val[:,0]**2
    eig_val[:,1] = np.sqrt(np.abs(eig_val[:,1]))
    eig_vec[:,:,1] -= eig_vec[:,:,0]**2
    eig_vec[:,:,1]  = np.sqrt(np.abs(eig_vec[:,:,1]))
    print("eigenvals, average ",eig_val[:,0])
    print("eigenvals, std     ",eig_val[:,1])
    print("eigenvecs, average ",eig_vec[:,:,0])
    print("eigenvecs, std     ",eig_vec[:,:,1])

    import numpy as np
    for i,ei in enumerate(eig_val[:,0]):
        outfile.write('eigenvalue %d = %.8f, %.8f \n' % (i,ei,eig_val[i,1]))
        for j,vji in enumerate(eig_vec[:,i,0]):
            outfile.write('eigenvector %d, component %d = %.8f, %.8f \n' % (i,j,vji,eig_vec[i,j,1]))
        outfile.write('-'*53+'\n')
    
    return (eta_samples, V_samples)

def compute_qse_greens_function(vqe_res,qse_res,x_mesh,kernel,outfile,shots=2048):
    import numpy as np
    # eigenstates from QSE have the form |Psi(mu) > = \sum_I V(I,mu) E(I) | Psi(VQE) > and eigenvalues E(mu)
    
    eta,V  = compute_qse_eigenpairs(vqe_res,qse_res,outfile,shots=shots)

    E0     = vqe_res['results_vqe'][0][0]
    n = len(eta[0])
    m = len(x_mesh)
    G_delta = np.zeros((n,n,shots),dtype='complex')
#    w       = np.zeros(n)
#    for i in range(n):
#        w[i] = 1.0/eta[i,1]**2
#        wt   += w[i]


    for x in range(shots):
        C_sample        = sample_matrix(qse_res['C'], False, False)
        G_delta[:,:,x]  = np.einsum('QI,Im->Qm',C_sample,V[x,:,:])

    ###########random distribution for Ground state energy#############

    E_std = np.std(eta,axis=0, dtype=np.float128)[0]/np.mean(eta, axis=0, dtype=np.float128)[0] * E0

    E0_rand = np.zeros((shots))
#    print(E0, E_std)
    for i in range(shots):
        E0_rand[i] = np.random.normal(E0, E_std)

 
    shot_list = [1000, 3000, 6000]#, 16000]#, 64000, 128000]
    init      = 0
    new_x = []
    new_y_real = []
    new_y_imag = []
    new_err_real = []
    new_err_imag = []
    m_jackknife = []
    s_jackknife_real = []
    s_jackknife_imag = []
    full_sample = np.zeros((shots,m,n,n), dtype='complex')
    for shot in shot_list:
        f_mesh  = np.zeros((m,n,n,3),'complex128')
#        E_std = np.std(eta[init:init+shot],axis=0, dtype=np.float128)[0]/np.mean(eta[init:init+shot], axis=0, dtype=np.float128)[0] * E0
        X_sample = np.zeros((shot,m,n,n), dtype='complex')
        for x in range(init, init+shot):
            # transition matrix elements C(pI) = < Psi(VQE)| a(p) E(I) | Psi(VQE) > 
            # k(x,Emu-E0) = 1/(I*w+E(mu-E0)) or 1/(I*w-(Emu-E0))
            
            m      = len(x_mesh)
            k_mat  = np.zeros((m,n),dtype=np.complex)

            energ_diff = sample_energy_differences(eta[x], E0_rand[x], E_std)

            for ix in range(m):
                for ie in range(n):
                    k_mat[ix,ie] = kernel(x_mesh[ix],energ_diff[ie])

    # G(pq,x) = \sum_mu < Psi(VQE) | a(p) | Psi(mu) > k(x,Emu-E0) < Psi(mu) | a*(q) | Psi(mu) >
    # X(p,mu) = < Psi(VQE) | a(p) | Psi(mu) >
    # [Eq A] --- G(pq,x) = \sum_mu X(p,mu) k(x,Emu-E0) Conjg[X(q,mu)]
    # [Eq B] --- X(p,mu) = < Psi(VQE) | a(p) | Psi(mu) > = \sum_I V(I,mu) < Psi(VQE) | a(p) E(I) | Psi(VQE) > = \sum_I V(I,mu) C(pI)


            G = G_delta[:,:,x]#sample_matrix(G_delta, False, False)

            for i in range(m):
                for p in range(n):
                    for q in range(n):
                        for mu in range(2):
                            X_sample[x-init,i,p,q] += (G[p,mu] * k_mat[i,mu] * np.conj(G)[q,mu])
        #                    X_sample[i,p,q] /= wt
#        X_sample = np.einsum('Pm,xm,Qm->xPQ',G,k_mat,np.conj(G))
#            print("X matrix being used:", X_sample[x-init][0])

            f_mesh[:,:,:,0] += X_sample[x-init]
            f_mesh[:,:,:,1] += X_sample[x-init].real**2
            f_mesh[:,:,:,2] += X_sample[x-init].imag**2
            full_sample[x,:,:,:] = X_sample[x-init,:,:,:]

        f_mesh[:,:,:,0] /= (shot)
        f_mesh[:,:,:,1] /= (shot)
        f_mesh[:,:,:,2] /= (shot)
        f_mesh[:,:,:,1]  = f_mesh[:,:,:,1] - f_mesh[:,:,:,0].real**2
        f_mesh[:,:,:,2]  = f_mesh[:,:,:,2] - f_mesh[:,:,:,0].imag**2
        f_mesh[:,:,:,1]  = np.sqrt(np.abs(f_mesh[:,:,:,1]))
        f_mesh[:,:,:,2]  = np.sqrt(np.abs(f_mesh[:,:,:,2]))


        # mean_jackknife, std_jackknife_real, std_jackknife_imag = jackknife.jackknife(X_sample[:,0,:,:], f_mesh[0,:,:,0])
        # m_jackknife.append(mean_jackknife[0,0])
        # s_jackknife_real.append(std_jackknife_real[0,0])
        # s_jackknife_imag.append(std_jackknife_imag[0,0])

        # print("for " + str(shot) + " shots:")
        # print("Green's Function Mean", f_mesh[0,:,:,0])
        # print("Green's Function Error(Real)", f_mesh[0,:,:,1])
        # print("Green's Function Error(Imag)", f_mesh[0,:,:,2])
        # print("Mean from Jackknife", mean_jackknife)
        # print("Error Jackknife(real)", std_jackknife_real)
        # print("Error Jackknife(imag)", std_jackknife_imag)

        # new_x.append(1.0/(shot)**2)

        # new_y_real.append(f_mesh[0,0,0,0].real)
        # new_y_imag.append(f_mesh[0,0,0,0].imag)
        # new_err_real.append(f_mesh[0,0,0,1])
        # new_err_imag.append(f_mesh[0,0,0,2])

        init += shot


    return full_sample


def statistical_analysis(gf,x_mesh):
    import numpy as np

    shot_list = [1000, 3000, 6000]#, 16000]#, 64000, 128000]
    init      = 0
    new_x = []
    new_y_real = []
    new_y_imag = []
    new_err_real = []
    new_err_imag = []
    m_jackknife = []
    s_jackknife_real = []
    s_jackknife_imag = []
    m = gf.shape[1]
    n = gf.shape[2]
    f_mesh_jk  = np.zeros((len(shot_list),m,n,n,3),'complex128')
    i=0
    green_sv = np.load('qse_gf_sv.npy',allow_pickle=True)
    for shot in shot_list:
        
        print("shot", shot)
        f_mesh  = np.zeros((m,n,n,3),'complex128')
        
        X_sample = np.zeros((shot,m,n,n), dtype='complex')

        X_sample = gf[init:shot+init]  

        f_mesh[:,:,:,0] = np.mean(X_sample, axis=0)
        f_mesh[:,:,:,1] = np.std(np.real(X_sample), axis=0)
        f_mesh[:,:,:,2] = np.std(np.imag(X_sample), axis=0)

        
        print("Number of samples", shot)

#        fig,axs = plt.subplots(nrows=1, ncols=2)
        plt.hist(np.around(np.real(X_sample[:,0,0,0]),6), density=True, bins=20)
        plt.xlabel("Re[G$_{00}$]")
        plt.ylabel("Number of samples")
        plt.savefig("Histogram_"+str(shot)+"_real.pdf", bbox_inches='tight')
        plt.close()
        plt.hist(np.around(np.imag(X_sample[:,0,0,0]),6),density=True, bins=20)
        plt.xlabel("Im[G$_{00}$]")
        plt.ylabel("Number of samples")
        plt.savefig("Histogram_"+str(shot)+"_imag.pdf", bbox_inches='tight')
        plt.close()

        
        for x in range(len(x_mesh)):
            f_mesh_jk[i,x,:,:,0], f_mesh_jk[i,x,:,:,1], f_mesh_jk[i,x,:,:,2] = jackknife.jackknife(X_sample[:,x,:,:], f_mesh[x,:,:,0])
        m_jackknife.append(f_mesh_jk[i,67,0,0,0])
        s_jackknife_real.append(f_mesh_jk[i,67,0,0,1])
        s_jackknife_imag.append(f_mesh_jk[i,67,0,0,2])

        fig, ax = plt.subplots(2)
        fig.set_tight_layout(True)
        ax[0].errorbar(x_mesh[10:20], np.real(f_mesh[10:20,0,0,0]), yerr=f_mesh[10:20,0,0,1])
        ax[1].errorbar(x_mesh[10:20], np.imag(f_mesh[10:20,0,0,0]), yerr=f_mesh[10:20,0,0,2])
        ax[1].set_xlabel("$\iota\omega$")
        ax[0].set_ylabel("$Re[G_{00}(\iota\omega)]$")
        ax[1].set_ylabel("$Im[G_{00}(\iota\omega)]$")

        plt.savefig("GF_"+str(shot)+".pdf")
        plt.close()


        new_x.append(1.0/(shot)**2)

        new_y_real.append(f_mesh[67,0,0,0].real)
        new_y_imag.append(f_mesh[67,0,0,0].imag)
        new_err_real.append(f_mesh[67,0,0,1])
        new_err_imag.append(f_mesh[67,0,0,2])

        print("for " + str(shot) + " shots:")
        print("Green's Function Mean", f_mesh[0,:,:,0], f_mesh[67,:,:,0])
        print("Green's Function Error(Real)", f_mesh[0,:,:,1], f_mesh[67,:,:,0])
        print("Green's Function Error(Imag)", f_mesh[0,:,:,2], f_mesh[67,:,:,0])
        print("Mean from Jackknife", f_mesh_jk[i,0,:,:,0], f_mesh_jk[i,67,:,:,0])
        print("Error Jackknife(real)", f_mesh_jk[i,0,:,:,1], f_mesh_jk[i,67,:,:,0])
        print("Error Jackknife(imag)", f_mesh_jk[i,0,:,:,2], f_mesh_jk[i,67,:,:,0])

        i    +=1
        init += shot

    print("Green's function from Statevector:", green_sv[0,0,:,:], green_sv[0,67,:,:])
    y_sv_real = []
    y_sv_imag = []
    for i in range(len(new_y_real)):
        y_sv_real.append(green_sv[0,67,0,0].real)
        y_sv_imag.append(green_sv[0,67,0,0].imag)

    print(y_sv_real, y_sv_imag)
    fig, ax = plt.subplots(2)
    fig.set_tight_layout(True)
    ax[0].errorbar(new_x, new_y_real, yerr=new_err_real)
 #   ax[0].errorbar(new_x, y_sv_real)
    ax[1].errorbar(new_x, new_y_imag, yerr=new_err_imag)
  #  ax[1].errorbar(new_x, y_sv_imag)
    ax[1].set_xlabel("$1/M^{2}$")
    ax[0].set_ylabel("$Re[G_{00}(\iota\omega)]$")
    ax[1].set_ylabel("$Im[G_{00}(\iota\omega)]$")
    plt.savefig("error_extrapolation.pdf")
    plt.close()

    fig, ax = plt.subplots(2)
    fig.set_tight_layout(True)
    ax[0].errorbar(new_x, new_y_real, yerr=new_err_real)
    ax[0].errorbar(new_x, y_sv_real)
    ax[1].errorbar(new_x, new_y_imag, yerr=new_err_imag)
    ax[1].errorbar(new_x, y_sv_imag)
    ax[1].set_xlabel("$1/M^{2}$")
    ax[0].set_ylabel("$Re[G_{00}(\iota\omega)]$")
    ax[1].set_ylabel("$Im[G_{00}(\iota\omega)]$")
    plt.savefig("error_extrapolation_1.pdf")
    plt.close()
    
    fig, ax = plt.subplots(2)
    fig.set_tight_layout(True)
    ax[0].errorbar(new_x, np.real(m_jackknife), yerr=s_jackknife_real)
#    ax[0].errorbar(new_x, y_sv_real)
    ax[1].errorbar(new_x, np.imag(m_jackknife), yerr=s_jackknife_imag)
#    ax[1].errorbar(new_x, y_sv_imag)
    ax[1].set_xlabel("$1/M^{2}$")
    ax[0].set_ylabel("$Re[G_{00}(\iota\omega)]$")
    ax[1].set_ylabel("$Im[G_{00}(\iota\omega)]$")
    plt.savefig("error_extrapolation_jk.pdf")
    plt.close()

    fig, ax = plt.subplots(2)
    fig.set_tight_layout(True)
    ax[0].errorbar(new_x, np.real(m_jackknife), yerr=s_jackknife_real)
    ax[0].errorbar(new_x, y_sv_real)
    ax[1].errorbar(new_x, np.imag(m_jackknife), yerr=s_jackknife_imag)
    ax[1].errorbar(new_x, y_sv_imag)
    ax[1].set_xlabel("$1/M^{2}$")
    ax[0].set_ylabel("$Re[G_{00}(\iota\omega)]$")
    ax[1].set_ylabel("$Im[G_{00}(\iota\omega)]$")
    plt.savefig("error_extrapolation_jk_1.pdf")
    plt.close()

    return f_mesh, f_mesh_jk


#    return np.einsum('Pm,xm,Qm->xPQ',X,k_mat,np.conj(X)) # [Eq A]


