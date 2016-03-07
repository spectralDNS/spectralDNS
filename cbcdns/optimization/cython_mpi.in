#cython: boundscheck=False
#cython: wraparound=False
cimport numpy as np

{0}

def transpose_Uc(np.ndarray[complex_t, ndim=3] Uc_hatT,
                 np.ndarray[complex_t, ndim=4] U_mpi,
                 int num_processes):    
    cdef unsigned int i, j, k, l, kk
    cdef unsigned int n0 = Uc_hatT.shape[0]
    cdef unsigned int n1 = U_mpi.shape[2]
    cdef unsigned int n2 = U_mpi.shape[3]    
    for i in xrange(num_processes): 
        for j in xrange(n0):
            for k in xrange(i*n1, (i+1)*n1):
                kk = k-i*n1
                for l in xrange(n2):
                    Uc_hatT[j, k, l] = U_mpi[i, j, kk, l]
    return Uc_hatT

def transpose_Umpi(np.ndarray[complex_t, ndim=4] U_mpi,
                   np.ndarray[complex_t, ndim=3] Uc_hatT,
                   int num_processes):
    cdef unsigned int i,j,k,l,kk
    cdef unsigned int n0 = U_mpi.shape[1]
    cdef unsigned int n1 = U_mpi.shape[2]
    cdef unsigned int n2 = U_mpi.shape[3]    
    for i in xrange(num_processes): 
        for j in xrange(n0):
            for k in xrange(i*n1, (i+1)*n1):
                kk = k-i*n1
                for l in xrange(n2):
                    U_mpi[i,j,kk,l] = Uc_hatT[j,k,l]
    return U_mpi

def transform_Uc_xz(np.ndarray[complex_t, ndim=3] Uc_hat_x, 
                    np.ndarray[complex_t, ndim=3] Uc_hat_z, int P1):
    cdef unsigned int i, j, k, l, i0
    cdef unsigned int n0 = Uc_hat_z.shape[0]
    cdef unsigned int n1 = Uc_hat_z.shape[1]
    cdef unsigned int n2 = Uc_hat_x.shape[2]
    for i in xrange(P1):
        for j in xrange(i*n0, (i+1)*n0):
            i0 = j-i*n0
            for k in xrange(n1):
                for l in xrange(n2):
                    Uc_hat_x[j, k, l] = Uc_hat_z[i0, k, l+i*n2]
    return Uc_hat_x

def transform_Uc_yx(np.ndarray[complex_t, ndim=3] Uc_hat_y, 
                    np.ndarray[complex_t, ndim=3] Uc_hat_xr, int P2):
    cdef unsigned int i, j, k, l, i0
    cdef unsigned int n0 = Uc_hat_y.shape[0]
    cdef unsigned int n1 = Uc_hat_xr.shape[1]
    cdef unsigned int n2 = Uc_hat_xr.shape[2]    
    for i in xrange(P2):
        for j in xrange(i*n0, (i+1)*n0):
            i0 = j-i*n0
            for k in xrange(n1):
                k0 = k+i*n1
                for l in xrange(n2):
                    Uc_hat_y[i0, k0, l] = Uc_hat_xr[j, k, l]
    return Uc_hat_y

def transform_Uc_xy(np.ndarray[complex_t, ndim=3] Uc_hat_x, 
                    np.ndarray[complex_t, ndim=3] Uc_hat_y, int P2):
    cdef unsigned int i, j, k, l, i0
    cdef unsigned int n0 = Uc_hat_y.shape[0]
    cdef unsigned int n1 = Uc_hat_x.shape[1]
    cdef unsigned int n2 = Uc_hat_x.shape[2]
    for i in xrange(P2):
        for j in xrange(i*n0, (i+1)*n0):
            i0 = j-i*n0
            for k in xrange(n1):
                for l in xrange(n2):
                    Uc_hat_x[j, k, l] = Uc_hat_y[i0, k+i*n1, l]
    return Uc_hat_x

def transform_Uc_zx(np.ndarray[complex_t, ndim=3] Uc_hat_z, 
                    np.ndarray[complex_t, ndim=3] Uc_hat_xr, int P1):
    cdef unsigned int i, j, k, l, i0
    cdef unsigned int n0 = Uc_hat_z.shape[0]
    cdef unsigned int n1 = Uc_hat_xr.shape[1]
    cdef unsigned int n2 = Uc_hat_xr.shape[2]
    for i in xrange(P1):
        for j in xrange(i*n0, (i+1)*n0):
            i0 = j-i*n0
            for k in xrange(n1):
                for l in xrange(n2):
                    Uc_hat_z[i0, k, l+i*n2] = Uc_hat_xr[j, k, l]
    return Uc_hat_z

def transpose_x_2D(np.ndarray[complex_t, ndim=3] U_send,
                   np.ndarray[complex_t, ndim=2] Uc_hatT,
                   int num_processes, int Np):
    
    cdef unsigned int i, j, k
    for i in xrange(num_processes): 
        for j in xrange(U_send.shape[1]):
            for k in xrange(U_send.shape[2]):
                U_send[i, j, k] = Uc_hatT[j, i*Np/2+k]
    return U_send

def transpose_y_2D(np.ndarray[complex_t, ndim=2] Uc_hatT, 
                   np.ndarray[complex_t, ndim=2] U_recv, 
                   int num_processes, int Np):
    
    cdef unsigned int i, j, k
    for i in range(num_processes): 
        for j in xrange(Uc_hatT.shape[0]):
            for k in xrange(U_recv.shape[1]):
                Uc_hatT[j, i*Np/2+k] = U_recv[i*Np+j, k]        
    return Uc_hatT

def swap_Nq_2D(np.ndarray[complex_t, ndim=1] fft_y, 
               np.ndarray[complex_t, ndim=2] fu, 
               np.ndarray[complex_t, ndim=1] fft_x, int N):
    
    cdef unsigned int i, j, k
    cdef int Nh = N/2
    fft_x[0] = fu[0, 0].real
    for i in xrange(1, Nh):
        fft_x[i] = 0.5*(fu[i, 0]+fu[N-i, 0].real - 1j*fu[N-i, 0].imag)
        
    fft_x[Nh] = fu[Nh, 0].real
    for i in xrange(Nh+1):
        fu[i, 0] = fft_x[i]
        
    for i in xrange(Nh-1):        
        fu[N/2+1+i, 0].real = fft_x[Nh-1-i].real 
        fu[N/2+1+i, 0].imag = -fft_x[Nh-1-i].imag
    
    fft_y[0] = fu[0, 0].imag
    for i in xrange(1, Nh):
        fft_y[i] = -0.5*1j*(fu[i, 0]-(fu[N-i, 0].real-1j*fu[N-i, 0].imag))
        
    fft_y[Nh] = fu[Nh, 0].imag
    for i in xrange(Nh-1):
        fft_y[N/2+1+i].real = fft_y[Nh-i-1].real
        fft_y[N/2+1+i].imag = -fft_y[Nh-i-1].imag
        
    return fft_y
