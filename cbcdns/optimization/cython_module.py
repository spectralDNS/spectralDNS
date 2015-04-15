#cython: boundscheck=False
#cython: wraparound=False
cimport numpy as np
from numpy.linalg import norm

{0}

ctypedef fused T:
    np.float64_t
    np.float32_t
    np.int64_t
    np.int32_t

def RK4(np.ndarray[complex_t, ndim=4] U_hat, 
        np.ndarray[complex_t, ndim=4] U_hat0, 
        np.ndarray[complex_t, ndim=4] U_hat1, 
        np.ndarray[complex_t, ndim=4] dU,
        np.ndarray[real_t, ndim=1] a, 
        np.ndarray[real_t, ndim=1] b,
        real_t dt,
        ComputeRHS):
    cdef complex_t z
    cdef unsigned int rk, i, j, k, l
    for i in xrange(dU.shape[0]):
        for j in xrange(dU.shape[1]):
            for k in xrange(dU.shape[2]):
                for l in xrange(dU.shape[3]):
                    z = U_hat[i,j,k,l]
                    U_hat1[i,j,k,l] = z 
                    U_hat0[i,j,k,l] = z
        
    for rk in xrange(4):
        dU = ComputeRHS(dU, rk)
        if rk < 3:
            for i in xrange(dU.shape[0]):
                for j in xrange(dU.shape[1]):
                    for k in xrange(dU.shape[2]):
                        for l in xrange(dU.shape[3]):
                            U_hat[i,j,k,l] = U_hat0[i,j,k,l] + b[rk]*dt*dU[i,j,k,l]
            
        for i in xrange(dU.shape[0]):
            for j in xrange(dU.shape[1]):
                for k in xrange(dU.shape[2]):
                    for l in xrange(dU.shape[3]):
                        U_hat1[i,j,k,l] = U_hat1[i,j,k,l] + a[rk]*dt*dU[i,j,k,l]
                        
    for i in xrange(dU.shape[0]):
        for j in xrange(dU.shape[1]):
            for k in xrange(dU.shape[2]):
                for l in xrange(dU.shape[3]):
                    U_hat[i,j,k,l] = U_hat1[i,j,k,l]
                    
    return U_hat

def ForwardEuler(np.ndarray[complex_t, ndim=4] U_hat, 
                 np.ndarray[complex_t, ndim=4] U_hat0, 
                 np.ndarray[complex_t, ndim=4] dU, 
                 real_t dt,
                 ComputeRHS):
    cdef complex_t z
    cdef unsigned int rk, i, j, k, l
    dU = ComputeRHS(dU, 0)
    for i in xrange(dU.shape[0]):
        for j in xrange(dU.shape[1]):
            for k in xrange(dU.shape[2]):
                for l in xrange(dU.shape[3]):
                    U_hat[i,j,k,l] = U_hat[i,j,k,l] + dU[i,j,k,l]*dt 
    return U_hat

def AB2(np.ndarray[complex_t, ndim=4] U_hat, 
        np.ndarray[complex_t, ndim=4] U_hat0, 
        np.ndarray[complex_t, ndim=4] dU,
        real_t dt, int tstep,
        ComputeRHS):
    cdef complex_t z
    cdef real_t p0 = 1.5
    cdef real_t p1 = 0.5
    cdef unsigned int rk, i, j, k, l
    dU = ComputeRHS(dU, 0)
    
    if tstep == 1:
        for i in xrange(dU.shape[0]):
            for j in xrange(dU.shape[1]):
                for k in xrange(dU.shape[2]):
                    for l in xrange(dU.shape[3]):
                        U_hat[i,j,k,l] = U_hat[i,j,k,l] + dU[i,j,k,l]*dt
                        
    else:
        for i in xrange(dU.shape[0]):
            for j in xrange(dU.shape[1]):
                for k in xrange(dU.shape[2]):
                    for l in xrange(dU.shape[3]):
                        U_hat[i,j,k,l] = U_hat[i,j,k,l] + p0*dU[i,j,k,l]*dt - p1*U_hat0[i,j,k,l]   
                    
    for i in xrange(dU.shape[0]):
        for j in xrange(dU.shape[1]):
            for k in xrange(dU.shape[2]):
                for l in xrange(dU.shape[3]):                    
                    U_hat0[i,j,k,l] = dU[i,j,k,l]*dt
    return U_hat

def dealias_rhs_NS(np.ndarray[complex_t, ndim=4] du,
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
    return du

def dealias_rhs_VV(np.ndarray[complex_t, ndim=4] du,
                   np.ndarray[np.uint8_t, ndim=3] dealias):
    du = dealias_rhs_NS(du, dealias)
    return du

def dealias_rhs_MHD(np.ndarray[complex_t, ndim=4] du,
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
                du[3, i, j, k].real *= uu
                du[3, i, j, k].imag *= uu
                du[4, i, j, k].real *= uu
                du[4, i, j, k].imag *= uu
                du[5, i, j, k].real *= uu
                du[5, i, j, k].imag *= uu

    return du

def add_pressure_diffusion_NS(np.ndarray[complex_t, ndim=4] du,
                              np.ndarray[complex_t, ndim=4] u_hat,
                              np.ndarray[int_t, ndim=3] ksq,
                              np.ndarray[int_t, ndim=4] kk,
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
    return du

def cross1(np.ndarray[real_t, ndim=4] c,
           np.ndarray[real_t, ndim=4] a,
           np.ndarray[real_t, ndim=4] b):
    cdef unsigned int i, j, k
    cdef real_t a0, a1, a2, b0, b1, b2
    for i in xrange(a.shape[1]):
        for j in xrange(a.shape[2]):
            for k in xrange(a.shape[3]):
                a0 = a[0,i,j,k]
                a1 = a[1,i,j,k]
                a2 = a[2,i,j,k]
                b0 = b[0,i,j,k]
                b1 = b[1,i,j,k]
                b2 = b[2,i,j,k]
                c[0,i,j,k] = a1*b2 - a2*b1
                c[1,i,j,k] = a2*b0 - a0*b2
                c[2,i,j,k] = a0*b1 - a1*b0
    return c

def cross2(np.ndarray[complex_t, ndim=4] c,
           np.ndarray[T, ndim=4] a,
           np.ndarray[complex_t, ndim=4] b):
    cdef unsigned int i, j, k
    cdef T a0, a1, a2
    cdef complex_t b0, b1, b2
    for i in xrange(a.shape[1]):
        for j in xrange(a.shape[2]):
            for k in xrange(a.shape[3]):
                a0 = a[0,i,j,k]
                a1 = a[1,i,j,k]
                a2 = a[2,i,j,k]
                b0 = b[0,i,j,k]
                b1 = b[1,i,j,k]
                b2 = b[2,i,j,k]
                c[0,i,j,k].real = -(a1*b2.imag - a2*b1.imag)
                c[0,i,j,k].imag = a1*b2.real - a2*b1.real
                c[1,i,j,k].real = -(a2*b0.imag - a0*b2.imag)
                c[1,i,j,k].imag = a2*b0.real - a0*b2.real
                c[2,i,j,k].real = -(a0*b1.imag - a1*b0.imag)
                c[2,i,j,k].imag = a0*b1.real - a1*b0.real
    return c

def transpose_Uc(np.ndarray[complex_t, ndim=3] Uc_hatT,
                 np.ndarray[complex_t, ndim=4] U_mpi,
                 int num_processes, int Np, int Nf):    
    cdef unsigned int i, j, k, l, kk
    for i in xrange(num_processes): 
        for j in xrange(Np):
            for k in xrange(i*Np, (i+1)*Np):
                kk = k-i*Np
                for l in xrange(Nf):
                    Uc_hatT[j, k, l] = U_mpi[i, j, kk, l]
    return Uc_hatT

def transpose_Umpi(np.ndarray[complex_t, ndim=4] U_mpi,
                   np.ndarray[complex_t, ndim=3] Uc_hatT,
                   int num_processes, int Np, int Nf):
    cdef unsigned int i,j,k,l,kk
    for i in xrange(num_processes): 
        for j in xrange(Np):
            for k in xrange(i*Np, (i+1)*Np):
                kk = k-i*Np  
                for l in xrange(Nf):
                    U_mpi[i,j,kk,l] = Uc_hatT[j,k,l]
    return U_mpi

def transform_Uc_xz(np.ndarray[complex_t, ndim=3] Uc_hat_x, 
                    np.ndarray[complex_t, ndim=3] Uc_hat_z,
                    int P1, int N1, int N2):
    cdef unsigned int i, j, k, l, i0
    for i in xrange(P1):
        for j in xrange(i*N1, (i+1)*N1):
            i0 = j-i*N1
            for k in xrange(N2):
                for l in xrange(N1/2):
                    Uc_hat_x[j, k, l] = Uc_hat_z[i0, k, l+i*N1/2]
    return Uc_hat_x

def transform_Uc_yx(np.ndarray[complex_t, ndim=3] Uc_hat_y, 
                    np.ndarray[complex_t, ndim=3] Uc_hat_xr,
                    int P2, int N1, int N2):
    cdef unsigned int i, j, k, l, i0
    for i in xrange(P2):
        for j in xrange(i*N2, (i+1)*N2):
            i0 = j-i*N2
            for k in xrange(N2):
                k0 = k+i*N2
                for l in xrange(N1/2):
                    Uc_hat_y[i0, k0, l] = Uc_hat_xr[j, k, l]
    return Uc_hat_y

def transform_Uc_xy(np.ndarray[complex_t, ndim=3] Uc_hat_x, 
                    np.ndarray[complex_t, ndim=3] Uc_hat_y,
                    int P2, int N1, int N2):
    cdef unsigned int i, j, k, l, i0
    for i in xrange(P2):
        for j in xrange(i*N2, (i+1)*N2):
            i0 = j-i*N2
            for k in xrange(N2):
                for l in xrange(N1/2):
                    Uc_hat_x[j, k, l] = Uc_hat_y[i0, k+i*N2, l]
    return Uc_hat_x

def transform_Uc_zx(np.ndarray[complex_t, ndim=3] Uc_hat_z, 
                    np.ndarray[complex_t, ndim=3] Uc_hat_xr,
                    int P1, int N1, int N2):
    cdef unsigned int i, j, k, l, i0
    for i in xrange(P1):
        for j in xrange(i*N1, (i+1)*N1):
            i0 = j-i*N1
            for k in xrange(N2):
                for l in xrange(N1/2):
                    Uc_hat_z[i0, k, l+i*N1/2] = Uc_hat_xr[j, k, l]
    return Uc_hat_z

### 2D routines

def add_pressure_diffusion_Bq2D(np.ndarray[complex_t, ndim=3] du, 
                                np.ndarray[complex_t, ndim=2] p_hat, 
                                np.ndarray[complex_t, ndim=3] u_hat,
                                np.ndarray[complex_t, ndim=2] rho_hat,
                                np.ndarray[real_t, ndim=3] k_over_k2, 
                                np.ndarray[int_t, ndim=3] k, 
                                np.ndarray[int_t, ndim=2] ksq, 
                                real_t nu, real_t Ri, real_t Pr):
    cdef unsigned int i, j
    cdef int_t k0, k1, k2
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

def RK4_2D(np.ndarray[complex_t, ndim=3] U_hat, 
        np.ndarray[complex_t, ndim=3] U_hat0, 
        np.ndarray[complex_t, ndim=3] U_hat1, 
        np.ndarray[complex_t, ndim=3] dU,
        np.ndarray[real_t, ndim=1] a, 
        np.ndarray[real_t, ndim=1] b,
        real_t dt,
        ComputeRHS):
    cdef complex_t z
    cdef unsigned int rk, i, j, k
    for i in xrange(dU.shape[0]):
        for j in xrange(dU.shape[1]):
            for k in xrange(dU.shape[2]):
                z = U_hat[i,j,k]
                U_hat1[i,j,k] = z 
                U_hat0[i,j,k] = z
        
    for rk in xrange(4):
        dU = ComputeRHS(dU, rk)
        if rk < 3:
            for i in xrange(dU.shape[0]):
                for j in xrange(dU.shape[1]):
                    for k in xrange(dU.shape[2]):
                        U_hat[i,j,k] = U_hat0[i,j,k] + b[rk]*dt*dU[i,j,k]
            
        for i in xrange(dU.shape[0]):
            for j in xrange(dU.shape[1]):
                for k in xrange(dU.shape[2]):
                    U_hat1[i,j,k] = U_hat1[i,j,k] + a[rk]*dt*dU[i,j,k]
                        
    for i in xrange(dU.shape[0]):
        for j in xrange(dU.shape[1]):
            for k in xrange(dU.shape[2]):
                U_hat[i,j,k] = U_hat1[i,j,k]
                    
    return U_hat

def ForwardEuler_2D(np.ndarray[complex_t, ndim=3] U_hat, 
                    np.ndarray[complex_t, ndim=3] U_hat0, 
                    np.ndarray[complex_t, ndim=3] dU, 
                    real_t dt,
                    ComputeRHS):
    cdef complex_t z
    cdef unsigned int rk, i, j, k
    dU = ComputeRHS(dU, 0)
    for i in xrange(dU.shape[0]):
        for j in xrange(dU.shape[1]):
            for k in xrange(dU.shape[2]):
                U_hat[i,j,k] = U_hat[i,j,k] + dU[i,j,k]*dt 
    return U_hat

def AB2_2D(np.ndarray[complex_t, ndim=3] U_hat, 
           np.ndarray[complex_t, ndim=3] U_hat0, 
           np.ndarray[complex_t, ndim=3] dU,
           real_t dt, int tstep,
           ComputeRHS):
    cdef complex_t z
    cdef real_t p0 = 1.5
    cdef real_t p1 = 0.5
    cdef unsigned int rk, i, j, k
    dU = ComputeRHS(dU, 0)
    
    if tstep == 1:
        for i in xrange(dU.shape[0]):
            for j in xrange(dU.shape[1]):
                for k in xrange(dU.shape[2]):
                    U_hat[i,j,k] = U_hat[i,j,k] + dU[i,j,k]*dt
                        
    else:
        for i in xrange(dU.shape[0]):
            for j in xrange(dU.shape[1]):
                for k in xrange(dU.shape[2]):
                    U_hat[i,j,k] = U_hat[i,j,k] + p0*dU[i,j,k]*dt - p1*U_hat0[i,j,k]   
                    
    for i in xrange(dU.shape[0]):
        for j in xrange(dU.shape[1]):
            for k in xrange(dU.shape[2]):
                U_hat0[i,j,k] = dU[i,j,k]*dt
    return U_hat

def dealias_rhs_NS2D(np.ndarray[complex_t, ndim=3] du,
                     np.ndarray[np.uint8_t, ndim=2] dealias):
    cdef unsigned int i, j
    cdef np.uint8_t uu
    for i in xrange(dealias.shape[0]):
        for j in xrange(dealias.shape[1]):
            uu = dealias[i, j]
            du[0, i, j].real *= uu
            du[0, i, j].imag *= uu
            du[1, i, j].real *= uu
            du[1, i, j].imag *= uu
    return du

def dealias_rhs_Bq2D(np.ndarray[complex_t, ndim=3] du,
                     np.ndarray[np.uint8_t, ndim=2] dealias):
    cdef unsigned int i, j
    cdef np.uint8_t uu
    for i in xrange(dealias.shape[0]):
        for j in xrange(dealias.shape[1]):
            uu = dealias[i, j]
            du[0, i, j].real *= uu
            du[0, i, j].imag *= uu
            du[1, i, j].real *= uu
            du[1, i, j].imag *= uu
            du[2, i, j].real *= uu
            du[2, i, j].imag *= uu 
    return du

def cross1_2D(np.ndarray[real_t, ndim=2] c,
              np.ndarray[real_t, ndim=3] a,
              np.ndarray[real_t, ndim=3] b):
    cdef unsigned int i, j
    cdef real_t a0, a1, b0, b1
    for i in xrange(a.shape[1]):
        for j in xrange(a.shape[2]):
            a0 = a[0,i,j]
            a1 = a[1,i,j]
            b0 = b[0,i,j]
            b1 = b[1,i,j]
            c[i,j] = a0*b1 - a1*b0
    return c

def cross2_2D(np.ndarray[complex_t, ndim=2] c,
              np.ndarray[T, ndim=3] a,
              np.ndarray[complex_t, ndim=3] b):
    cdef unsigned int i, j
    cdef T a0, a1
    cdef complex_t b0, b1
    for i in xrange(a.shape[1]):
        for j in xrange(a.shape[2]):
            a0 = a[0,i,j]
            a1 = a[1,i,j]
            b0 = b[0,i,j]
            b1 = b[1,i,j]
            c[i,j].real = -(a0*b1.imag - a1*b0.imag)
            c[i,j].imag = a0*b1.real - a1*b0.real
    return c

def add_pressure_diffusion_NS2D(np.ndarray[complex_t, ndim=3] du,
                                np.ndarray[complex_t, ndim=2] p_hat,                  
                                np.ndarray[complex_t, ndim=3] u_hat,
                                np.ndarray[int_t, ndim=3] k,
                                np.ndarray[int_t, ndim=2] ksq,
                                np.ndarray[real_t, ndim=3] k_over_k2,
                                real_t nu):
    cdef unsigned int i, j
    cdef real_t z
    cdef int_t k0, k1
    for i in xrange(ksq.shape[0]):
        for j in xrange(ksq.shape[1]):
            z = nu*ksq[i,j]
            k0 = k[0,i,j]
            k1 = k[1,i,j]
            p_hat[i,j] = du[0,i,j]*k_over_k2[0,i,j]+du[1,i,j]*k_over_k2[1,i,j]
            du[0,i,j] = du[0,i,j] - (p_hat[i,j]*k0+u_hat[0,i,j]*z)
            du[1,i,j] = du[1,i,j] - (p_hat[i,j]*k1+u_hat[1,i,j]*z)
    return du

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
