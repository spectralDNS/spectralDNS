#cython: boundscheck=False
#cython: wraparound=False
cimport numpy as np

{0}

def add_pressure_diffusion_NS(np.ndarray[complex_t, ndim=4] du,
                              np.ndarray[complex_t, ndim=4] u_hat,
                              np.ndarray[real_t, ndim=3] ksq,
                              np.ndarray[real_t, ndim=4] kk,
                              np.ndarray[complex_t, ndim=3] p_hat,
                              np.ndarray[real_t, ndim=4] k_over_k2,
                              real_t nu):
    cdef unsigned int i, j, k
    cdef real_t z
    cdef real_t k0, k1, k2
    for i in xrange(ksq.shape[0]):
        for j in xrange(ksq.shape[1]):
            for k in xrange(ksq.shape[2]):
                z = nu*ksq[i,j,k]
                k0 = kk[0,i,j,k]
                k1 = kk[1,i,j,k]
                k2 = kk[2,i,j,k]
                p_hat[i,j,k] = du[0,i,j,k]*k_over_k2[0,i,j,k]+du[1,i,j,k]*k_over_k2[1,i,j,k]+du[2,i,j,k]*k_over_k2[2,i,j,k]
                du[0,i,j,k] = du[0,i,j,k] - (p_hat[i,j,k]*k0+u_hat[0,i,j,k]*z)
                du[1,i,j,k] = du[1,i,j,k] - (p_hat[i,j,k]*k1+u_hat[1,i,j,k]*z)
                du[2,i,j,k] = du[2,i,j,k] - (p_hat[i,j,k]*k2+u_hat[2,i,j,k]*z)
    return du

def add_pressure_diffusion_Bq2D(np.ndarray[complex_t, ndim=3] du, 
                                np.ndarray[complex_t, ndim=2] p_hat, 
                                np.ndarray[complex_t, ndim=3] u_hat,
                                np.ndarray[complex_t, ndim=2] rho_hat,
                                np.ndarray[real_t, ndim=3] k_over_k2, 
                                np.ndarray[real_t, ndim=3] k, 
                                np.ndarray[real_t, ndim=2] ksq, 
                                real_t nu, real_t Ri, real_t Pr):
    cdef unsigned int i, j
    cdef real_t k0, k1, k2
    cdef real_t z
    # Compute pressure (To get actual pressure multiply by 1j)
    for i in xrange(ksq.shape[0]):
        for j in xrange(ksq.shape[1]):
            z = nu*ksq[i,j]
            k0 = k[0,i,j]
            k1 = k[1,i,j]
            k2 = k[2,i,j]
            p_hat[i,j] = du[0,i,j]*k_over_k2[0,i,j]+du[1,i,j]*k_over_k2[1,i,j] - Ri*rho_hat[i,j]*k_over_k2[1,i,j]
            
            du[0,i,j] = du[0,i,j] - (p_hat[i,j]*k0+u_hat[0,i,j]*z)
            du[1,i,j] = du[1,i,j] - (p_hat[i,j]*k1+u_hat[1,i,j]*z+Ri*rho_hat[i,j])
            du[2,i,j] = du[2,i,j] - rho_hat[i,j]*z/Pr
            
    return du

def add_pressure_diffusion_NS2D(np.ndarray[complex_t, ndim=3] du,
                                np.ndarray[complex_t, ndim=2] p_hat,                  
                                np.ndarray[complex_t, ndim=3] u_hat,
                                np.ndarray[real_t, ndim=3] k,
                                np.ndarray[real_t, ndim=2] ksq,
                                np.ndarray[real_t, ndim=3] k_over_k2,
                                real_t nu):
    cdef unsigned int i, j
    cdef real_t z
    cdef real_t k0, k1
    for i in xrange(ksq.shape[0]):
        for j in xrange(ksq.shape[1]):
            z = nu*ksq[i,j]
            k0 = k[0,i,j]
            k1 = k[1,i,j]
            p_hat[i,j] = du[0,i,j]*k_over_k2[0,i,j]+du[1,i,j]*k_over_k2[1,i,j]
            du[0,i,j] = du[0,i,j] - (p_hat[i,j]*k0+u_hat[0,i,j]*z)
            du[1,i,j] = du[1,i,j] - (p_hat[i,j]*k1+u_hat[1,i,j]*z)
    return du
    