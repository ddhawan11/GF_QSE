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

def compute_qse_eigenpairs(vqe_res,qse_res,outfile,lindep=1e-15):
    S     = qse_res['S'][:,:,0]
    H     = qse_res['H'][:,:,0]
    eta,V = safe_eigh(H,S,lindep=lindep)
    import numpy as np
    for i,ei in enumerate(eta):
        outfile.write('eigenvalue %d = %.8f \n' % (i,ei))
        for j,vji in enumerate(V[:,i]):
            outfile.write('eigenvector %d, component %d = %.8f \n' % (i,j,vji))
        outfile.write('-'*53+'\n')
    return eta,V

def compute_qse_greens_function(vqe_res,qse_res,x_mesh,kernel,outfile):
    import numpy as np
    # eigenstates from QSE have the form |Psi(mu) > = \sum_I V(I,mu) E(I) | Psi(VQE) > and eigenvalues E(mu)
    eta,V  = compute_qse_eigenpairs(vqe_res,qse_res,outfile)
    E0     = vqe_res['results_vqe'][0][0]
    # transition matrix elements C(pI) = < Psi(VQE)| a(p) E(I) | Psi(VQE) > 
    C      = qse_res['C'][:,:,0]
    n      = len(eta)
    m      = len(x_mesh)
    # k(x,Emu-E0) = 1/(I*w+E(mu-E0)) or 1/(I*w-(Emu-E0))
    k_mat  = np.zeros((m,n),dtype=np.complex)
    for ix in range(m):
        for ie in range(n):
            k_mat[ix,ie] = kernel(x_mesh[ix],eta[ie]-E0)
    # G(pq,x) = \sum_mu < Psi(VQE) | a(p) | Psi(mu) > k(x,Emu-E0) < Psi(mu) | a*(q) | Psi(mu) >
    # X(p,mu) = < Psi(VQE) | a(p) | Psi(mu) >
    # [Eq A] --- G(pq,x) = \sum_mu X(p,mu) k(x,Emu-E0) Conjg[X(q,mu)]
    # [Eq B] --- X(p,mu) = < Psi(VQE) | a(p) | Psi(mu) > = \sum_I V(I,mu) < Psi(VQE) | a(p) E(I) | Psi(VQE) > = \sum_I V(I,mu) C(pI)
    X = np.einsum('QI,Im->Qm',C,V)                       # [Eq B]
    return np.einsum('Pm,xm,Qm->xPQ',X,k_mat,np.conj(X)) # [Eq A]

