import numpy as np
from scipy import linalg as LA

np.random.seed(0)

def generate_random_unitary(no):
    M = np.random.random((no,no))
    Q,R = np.linalg.qr(M)
    return Q

def dst(x,a):
    return 1-np.tanh((x-a)/6.0)

def generate_random_HS(no,ne,sigma=1e-2):
    # generate a pair of matrices H,S
    # entries H(i,j) and S(i,j) are statistically independent and have error bars of magnitude at most sigma
    H   = np.zeros((no,no,2))
    S   = np.zeros((no,no,2))
    X   = generate_random_unitary(no)
    Y   = X
    eta = [1.0+0.1*x     for x in range(no)]
    sgm = [dst(x,ne)     for x in range(no)]
    sgm = [x*ne/sum(sgm) for x in sgm]
    res = [x/y for (x,y) in zip(eta,sgm)]
    H[:,:,0] = np.einsum('pm,m,qm->pq',X,eta,X)
    S[:,:,0] = np.einsum('pm,m,qm->pq',Y,sgm,Y)
    H[:,:,1] = np.random.random((no,no))*sigma  # error bars
    S[:,:,1] = np.random.random((no,no))*sigma  # error bars
    H[:,:,1] = (H[:,:,1]+H[:,:,1].T)/2.0
    S[:,:,1] = (S[:,:,1]+S[:,:,1].T)/2.0
    return H,S,res

def get_sample(X):
    no = X.shape[0]
    Xj = np.zeros((no,no))
    for r in range(no):
        for c in range(no):
            Xj[r,c] = np.random.normal(X[r,c,0],X[r,c,1])
    return (Xj+Xj.T)/2.0

def get_samples(H,S,t_mesh,n_samples=10000):
    # we draw many (n_samples) statistical samples of H and S
    nt  = len(t_mesh)
    eps = np.zeros((no,n_samples))
    U   = np.zeros((no,no,n_samples))
    G   = np.zeros((no,no,nt,n_samples))
    for j in range(n_samples):
        Hj         = get_sample(H)   # draw a sample from H
        Sj         = get_sample(S)   # draw a sample from S
        ej,Uj      = LA.eigh(Hj,Sj)  # we diagonalize
        eps[:,j]   = ej              # every sample gives you a set of eigenvalues e(i,j) i=1...6 j=1...10000 
        U[:,:,j]   = Uj              # get sample of the eigenvectors
        ker        = np.exp(-np.einsum('m,t->tm',ej,t_mesh))     # construct exp(-t*e(i)) = k(t,i)
        Trans      = np.einsum('pq,pm->qm',Sj,Uj)                # <Psi_0|a_p|Psi_i> = T(pi)
        G[:,:,:,j] = np.einsum('pm,tm,qm->pqt',Trans,ker,Trans)  # G_{pq}(t) = \sum_i T(pi) T(qi) k(t,i)
    return eps,U,G

def spectrum_analysis(eps,res):
    # very basic idea E(i) = \sum_j e(i,j) / N with variance # S(i) = \sum_j e(i,j)^2 /N - E(i)^2
    import matplotlib.pyplot as plt
    no,n_sample = eps.shape
    for p in range(no):
        print("eigenvalue ",np.mean(eps[p,:])," +/- ",np.std(eps[p,:]))
    # you can compute the covariance between the eigenvalues
    # Cov(i,j) = < ei ej > - < ei > < ej >
    for p in range(no):
        cov_pq = []
        for q in range(no):
            if(q>p): plt.scatter(eps[p,:],eps[q,:],label='pair (%d,%d)'%(p,q))
            cov_pq.append(np.cov(eps[p,:],eps[q,:])[0][1])
        print("covariance "+" ".join(['%.6f' % x for x in cov_pq]))
        #    print("covariance ",p,q,np.cov(eps[p,:],eps[q,:])[0][1])
    for p in range(no):
        for q in range(p+1,no):
            if(q==0):
               plt.scatter([res[p]],[res[q]],marker='x',color='black',label='exact')
            else:
               plt.scatter([res[p]],[res[q]],marker='x',color='black')
    plt.title('covariance between eigenvalues')
    plt.xlabel('eigenvalue i')
    plt.ylabel('eigenvalue j')
    plt.legend(ncol=3)
    plt.show()

def stat(G,p,q):
    nt,ns = G.shape[2:] 
    g_ave = np.zeros(nt)
    g_std = np.zeros(nt)
    for t in range(nt):
        g_ave[t] = np.sum(G[p,q,t,:])/ns         # for every pair p,q and for every time, get mean value and std deviation
        g_std[t] = np.sum(G[p,q,t,:]**2)/ns
    return g_ave,np.sqrt(np.abs(g_std-g_ave**2))

def gf_analysis(G,t_mesh):
    import matplotlib.pyplot as plt
    no = G.shape[0]
    for p in range(no):
        for q in range(p,no):
            m,s = stat(G,p,q)
            plt.errorbar(t_mesh,m,yerr=s,label='pair (%d,%d)'%(p,q))
    plt.title('imaginary-time green function')
    plt.xlabel('imaginary time')
    plt.ylabel('green function')
    plt.legend(ncol=3)
    plt.show()

def gf_correlation_analysis(G,p,q,t_mesh,pearson=True):
    import matplotlib.pyplot as plt
    nt = len(t_mesh)
    ker = np.zeros((nt,nt))
    for ta in range(nt):
        for tb in range(nt):
            ker[ta,tb] = np.cov(G[p,q,ta,:],G[p,q,tb,:])[0][1] # \sum_{j} G(p,q,ta) * G(p,q,tb) / n 
    if(pearson):
       ker = np.einsum('ij,i,j->ij',ker,1.0/np.sqrt(np.diag(ker)),1.0/np.sqrt(np.diag(ker))) # cov(x,y)/std(x)/std(y)
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    res = ax.imshow(ker) #array(norm_conf), cmap=cm.jet, interpolation='nearest')
    cb = fig.colorbar(res)
    ax.set_title('imaginary-time green function covariance kernel k(t,s)')
    ax.set_xlabel('imaginary-time t')
    ax.set_ylabel('imaginary-time s')
    plt.show()

# -------------------------------------------------------

no      = 6                        # 6x6 matrices
ne      = 3                        # 
t_mesh  = np.arange(0,2+1e-6,1e-1) # time mesh
H,S,res = generate_random_HS(no,ne,sigma=2e-2)   # happen on the hardware (or on the simulator) $$$
eps,U,G = get_samples(H,S,t_mesh,n_samples=100)  # post-processing on classical computer
spectrum_analysis(eps,res)                       # post-processing on classical computer
gf_analysis(G,t_mesh)                            # post-processing on classical computer
gf_correlation_analysis(G,0,0,t_mesh,pearson=True)




