#cython: boundscheck=False
#cython: wraparound=False
cimport numpy as np

{0}

def cython_dealias(np.ndarray[complex_t, ndim=4] du,
                  np.ndarray[np.uint8_t, ndim=3] dealias):
    cdef unsigned int i, j, k
    cdef np.uint8_t uu
    for i in xrange(dealias.shape[0]):
        for j in xrange(dealias.shape[1]):
            for k in xrange(dealias.shape[2]):
                uu = dealias[i, j, k]
                du[0, i, j, k].real *= uu
                du[0, i, j, k].imag *= uu
                du[1, i, j, k].real *= uu
                du[1, i, j, k].imag *= uu
                du[2, i, j, k].real *= uu
                du[2, i, j, k].imag *= uu

def cython_add_pressure_diffusion(np.ndarray[complex_t, ndim=4] du,
              np.ndarray[complex_t, ndim=4] u_hat,
              np.ndarray[int, ndim=3] ksq,
              np.ndarray[int, ndim=4] kk,
              np.ndarray[complex_t, ndim=3] p_hat,
              np.ndarray[real_t, ndim=4] k_over_k2,
              real_t nu):
    cdef unsigned int i, j, k
    cdef real_t z
    cdef int_t k0, k1, k2
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

def cython_cross1(np.ndarray[real_t, ndim=4] a,
                  np.ndarray[real_t, ndim=4] b,
                  np.ndarray[real_t, ndim=4] c):
    cdef unsigned int i, j, k
    cdef real_t a0, a1, a2, b0, b1, b2
    for i in xrange(a.shape[1]):
        for j in xrange(a.shape[2]):
            for k in xrange(a.shape[3]):
                a0 = a[0,i,j,k];
                a1 = a[1,i,j,k];
                a2 = a[2,i,j,k];
                b0 = b[0,i,j,k];
                b1 = b[1,i,j,k];
                b2 = b[2,i,j,k];
                c[0,i,j,k] = a1*b2 - a2*b1
                c[1,i,j,k] = a2*b0 - a0*b2
                c[2,i,j,k] = a0*b1 - a1*b0

def cython_cross2(np.ndarray[int, ndim=4] a,
                  np.ndarray[complex_t, ndim=4] b,
                  np.ndarray[complex_t, ndim=4] c):
    cdef unsigned int i, j, k
    cdef int_t a0, a1, a2
    cdef complex_t b0, b1, b2
    for i in xrange(a.shape[1]):
        for j in xrange(a.shape[2]):
            for k in xrange(a.shape[3]):
                a0 = a[0,i,j,k];
                a1 = a[1,i,j,k];
                a2 = a[2,i,j,k];
                b0 = b[0,i,j,k];
                b1 = b[1,i,j,k];
                b2 = b[2,i,j,k];
                c[0,i,j,k].real = -(a1*b2.imag - a2*b1.imag)
                c[0,i,j,k].imag = a1*b2.real - a2*b1.real
                c[1,i,j,k].real = -(a2*b0.imag - a0*b2.imag)
                c[1,i,j,k].imag = a2*b0.real - a0*b2.real
                c[2,i,j,k].real = -(a0*b1.imag - a1*b0.imag)
                c[2,i,j,k].imag = a0*b1.real - a1*b0.real

def transpose_Uc(np.ndarray[complex_t, ndim=3] Uc_hatT,
                 np.ndarray[complex_t, ndim=4] U_mpi, 
                 int num_processes, int Np):
    cdef unsigned int i,j,k,l,kk
    for i in xrange(num_processes): 
        for j in xrange(Uc_hatT.shape[0]):
            for k in xrange(i*Np, (i+1)*Np):
                kk = k-i*Np
                for l in xrange(Uc_hatT.shape[2]):
                    Uc_hatT[j, k, l] = U_mpi[i, j, kk, l]

def transpose_Umpi(np.ndarray[complex_t, ndim=4] U_mpi,
                   np.ndarray[complex_t, ndim=3] Uc_hatT,
                   int num_processes, int Np):
    cdef unsigned int i,j,k,l,kk
    for i in xrange(num_processes): 
        for j in xrange(Uc_hatT.shape[0]):
            for k in xrange(i*Np, (i+1)*Np):
                kk = k-i*Np    
                for l in xrange(Uc_hatT.shape[2]):
                    U_mpi[i,j,kk,l] = Uc_hatT[j,k,l]

def transform_Uc_xz(np.ndarray[complex_t, ndim=3] Uc_hat_x, 
                   np.ndarray[complex_t, ndim=3] Uc_hat_z, 
                   int P1, int N1):
    cdef unsigned int i,j,k,l,i0
    for i in xrange(P1):
        for j in xrange(i*N1, (i+1)*N1):
            i0 = j-i*N1
            for k in xrange(Uc_hat_x.shape[1]):
                for l in xrange(Uc_hat_x.shape[2]):
                    Uc_hat_x[j, k, l] = Uc_hat_z[i0, k, l+i*N1/2]

def transform_Uc_yx(np.ndarray[complex_t, ndim=3] Uc_hat_y, 
                   np.ndarray[complex_t, ndim=3] Uc_hat_xr, 
                   int P2, int N2):
    cdef unsigned int i,j,k,l,i0
    for i in xrange(P2):
        for j in xrange(i*N2, (i+1)*N2):
            i0 = j-i*N2
            for k in xrange(Uc_hat_xr.shape[1]):
                k0 = k+i*N2
                for l in xrange(Uc_hat_xr.shape[2]):
                    Uc_hat_y[i0, k0, l] = Uc_hat_xr[j, k, l]

def transform_Uc_xy(np.ndarray[complex_t, ndim=3] Uc_hat_x, 
                   np.ndarray[complex_t, ndim=3] Uc_hat_y, 
                   int P2, int N2):
    cdef unsigned int i,j,k,l,i0
    for i in xrange(P2):
        for j in xrange(i*N2, (i+1)*N2):
            i0 = j-i*N2
            for k in xrange(Uc_hat_x.shape[1]):
                for l in xrange(Uc_hat_x.shape[2]):
                    Uc_hat_x[j, k, l] = Uc_hat_y[i0, k+i*N2, l]

def transform_Uc_zx(np.ndarray[complex_t, ndim=3] Uc_hat_z, 
                    np.ndarray[complex_t, ndim=3] Uc_hat_xr, 
                    int P1, int N1):
    cdef unsigned int i,j,k,l,i0
    for i in xrange(P1):
        for j in xrange(i*N1, (i+1)*N1):
            i0 = j-i*N1
            for k in xrange(Uc_hat_xr.shape[1]):
                for l in xrange(Uc_hat_xr.shape[2]):
                    Uc_hat_z[i0, k, l+i*N1/2] = Uc_hat_xr[j, k, l]
