from numba import jit, int64, int32, uint8, autojit

from numba import {0} as float
from numba import {1} as complex

#@jit(complex[:,:,:,:](complex[:,:,:,:], complex[:,:,:,:], complex[:,:,:,:],
                      #complex[:,:,:,:], float[:], float[:]), 
                      #nopython=True, locals=globals())
#def RK4(U_hat, U_hat0, U_hat1, dU, a, b):
    #for i in xrange(dU.shape[0]):
        #for j in xrange(dU.shape[1]):
            #for k in xrange(dU.shape[2]):
                #for l in xrange(dU.shape[3]):
                    #z = U_hat[i,j,k,l]
                    #U_hat1[i,j,k,l] = z 
                    #U_hat0[i,j,k,l] = z
                    
    #for rk in xrange(4):
        #dU = ComputeRHS(dU, rk)
        #if rk < 3:
            #for i in xrange(dU.shape[0]):
                #for j in xrange(dU.shape[1]):
                    #for k in xrange(dU.shape[2]):
                        #for l in xrange(dU.shape[3]):
                            #U_hat[i,j,k,l] = U_hat0[i,j,k,l] + b[rk]*dU[i,j,k,l]
            
        #for i in xrange(dU.shape[0]):
            #for j in xrange(dU.shape[1]):
                #for k in xrange(dU.shape[2]):
                    #for l in xrange(dU.shape[3]):
                        #U_hat1[i,j,k,l] = U_hat1[i,j,k,l] + a[rk]*dU[i,j,k,l]
                        
    #for i in xrange(dU.shape[0]):
        #for j in xrange(dU.shape[1]):
            #for k in xrange(dU.shape[2]):
                #for l in xrange(dU.shape[3]):
                    #U_hat[i,j,k,l] = U_hat1[i,j,k,l]
                    
    #return U_hat

#@jit(complex[:,:,:,:](complex[:,:,:,:], complex[:,:,:,:], complex[:,:,:,:], float))
#def ForwardEuler(U_hat, U_hat0, dU, dt):
    #dU = ComputeRHS(dU, 0)
    #for i in xrange(dU.shape[0]):
        #for j in xrange(dU.shape[1]):
            #for k in xrange(dU.shape[2]):
                #for l in xrange(dU.shape[3]):
                    #U_hat[i,j,k,l] = U_hat[i,j,k,l] + dU[i,j,k,l]*dt  

#@jit(complex[:,:,:,:](complex[:,:,:,:], complex[:,:,:,:], complex[:,:,:,:], float, int64))
#def AB2(U_hat, U_hat0, dU, dt, tstep):
    #p0 = 1.5
    #p1 = 0.5
    #dU = ComputeRHS(dU, 0)
    
    #if tstep == 1:
        #for i in xrange(dU.shape[0]):
            #for j in xrange(dU.shape[1]):
                #for k in xrange(dU.shape[2]):
                    #for l in xrange(dU.shape[3]):
                        #U_hat[i,j,k,l] = U_hat[i,j,k,l] + dU[i,j,k,l]*dt
                        
    #else:
        #for i in xrange(dU.shape[0]):
            #for j in xrange(dU.shape[1]):
                #for k in xrange(dU.shape[2]):
                    #for l in xrange(dU.shape[3]):
                        #U_hat[i,j,k,l] = U_hat[i,j,k,l] + p0*dU[i,j,k,l]*dt - p1*U_hat0[i,j,k,l]   
                    
    #for i in xrange(dU.shape[0]):
        #for j in xrange(dU.shape[1]):
            #for k in xrange(dU.shape[2]):
                #for l in xrange(dU.shape[3]):                    
                    #U_hat0[i,j,k,l] = dU[i,j,k,l]*dt

@jit(float[:,:,:,:](float[:,:,:,:], float[:,:,:,:], float[:,:,:,:]), nopython=True)
def cross1(c, a, b):
    """Regular c = a x b"""
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

@jit(complex[:,:,:,:](complex[:,:,:,:], int64[:,:,:,:], complex[:,:,:,:]), nopython=True)    
def cross2a(c, a, b):
    """ c = 1j*(a x b)"""
    for i in xrange(a.shape[1]):
        for j in xrange(a.shape[2]):
            for k in xrange(a.shape[3]):
                a0 = a[0,i,j,k]
                a1 = a[1,i,j,k]
                a2 = a[2,i,j,k]
                b0 = b[0,i,j,k]
                b1 = b[1,i,j,k]
                b2 = b[2,i,j,k]
                c[0,i,j,k] = -(a1*b2.imag - a2*b1.imag) +1j*(a1*b2.real - a2*b1.real)
                c[1,i,j,k] = -(a2*b0.imag - a0*b2.imag) +1j*(a2*b0.real - a0*b2.real)
                c[2,i,j,k] = -(a0*b1.imag - a1*b0.imag) +1j*(a0*b1.real - a1*b0.real)
    return c

@jit(complex[:,:,:,:](complex[:,:,:,:], float[:,:,:,:], complex[:,:,:,:]), nopython=True)    
def cross2b(c, a, b):
    """ c = 1j*(a x b)"""
    for i in xrange(a.shape[1]):
        for j in xrange(a.shape[2]):
            for k in xrange(a.shape[3]):
                a0 = a[0,i,j,k]
                a1 = a[1,i,j,k]
                a2 = a[2,i,j,k]
                b0 = b[0,i,j,k]
                b1 = b[1,i,j,k]
                b2 = b[2,i,j,k]
                c[0,i,j,k] = -(a1*b2.imag - a2*b1.imag) +1j*(a1*b2.real - a2*b1.real)
                c[1,i,j,k] = -(a2*b0.imag - a0*b2.imag) +1j*(a2*b0.real - a0*b2.real)
                c[2,i,j,k] = -(a0*b1.imag - a1*b0.imag) +1j*(a0*b1.real - a1*b0.real)
    return c

def cross2(c, a, b):
    if a.dtype == int64.key:
        c = cross2a(c, a, b)
    elif a.dtype == float.key:
        c = cross2b(c, a, b)
    return c

@jit(complex[:,:,:,:](complex[:,:,:,:], uint8[:,:,:]))
def dealias_rhs(du, dealias):
    for i in xrange(dealias.shape[0]):
        for j in xrange(dealias.shape[1]):
            for k in xrange(dealias.shape[2]):
                uu = dealias[i, j, k]
                du[0, i, j, k] = du[0, i, j, k]*uu
                du[1, i, j, k] = du[1, i, j, k]*uu
                du[2, i, j, k] = du[2, i, j, k]*uu
    return du

  
@jit(complex[:,:,:,:](complex[:,:,:,:], complex[:,:,:,:], 
                      float[:,:,:], float[:,:,:,:], complex[:,:,:],
                      float[:,:,:,:], float), nopython=True)
def add_pressure_diffusion(du, u_hat, ksq, kk, p_hat, k_over_k2, nu):    
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

@jit(complex[:,:,:](complex[:,:,:], complex[:,:,:,:], int64, int64, int64), nopython=True)
def transpose_Uc(Uc_hatT, U_mpi, num_processes, Np, Nf):
    for i in xrange(num_processes): 
        for j in xrange(Np):
            for k in xrange(i*Np, (i+1)*Np):
                kk = k-i*Np
                for l in xrange(Nf):
                    Uc_hatT[j, k, l] = U_mpi[i, j, kk, l]
    return Uc_hatT

@jit(complex[:,:,:,:](complex[:,:,:,:], complex[:,:,:], int64, int64, int64), nopython=True)
def transpose_Umpi(U_mpi, Uc_hatT, num_processes, Np, Nf):
    for i in xrange(num_processes): 
        for j in xrange(Np):
            for k in xrange(i*Np, (i+1)*Np):
                kk = k-i*Np    
                for l in xrange(Nf):
                    U_mpi[i,j,kk,l] = Uc_hatT[j,k,l]
    return U_mpi

@jit(complex[:,:,:](complex[:,:,:], complex[:,:,:], int64, int64, int64), nopython=True)
def transform_Uc_xz(Uc_hat_x, Uc_hat_z, P1, N1, N2):
    for i in xrange(P1):
        for j in xrange(i*N1, (i+1)*N1):
            i0 = j-i*N1
            for k in xrange(N2):
                for l in xrange(N1/2):
                    Uc_hat_x[j, k, l] = Uc_hat_z[i0, k, l+i*N1/2]
    return Uc_hat_x

@jit(complex[:,:,:](complex[:,:,:], complex[:,:,:], int64, int64, int64), nopython=True)
def transform_Uc_yx(Uc_hat_y, Uc_hat_xr, P2, N1, N2):
    for i in xrange(P2):
        for j in xrange(i*N2, (i+1)*N2):
            i0 = j-i*N2
            for k in xrange(N2):
                k0 = k+i*N2
                for l in xrange(N1/2):
                    Uc_hat_y[i0, k0, l] = Uc_hat_xr[j, k, l]
    return Uc_hat_y

@jit(complex[:,:,:](complex[:,:,:], complex[:,:,:], int64, int64, int64), nopython=True)
def transform_Uc_xy(Uc_hat_x, Uc_hat_y, P2, N1, N2):
    for i in xrange(P2):
        for j in xrange(i*N2, (i+1)*N2):
            i0 = j-i*N2
            for k in xrange(N2):
                for l in xrange(N1/2):
                    Uc_hat_x[j, k, l] = Uc_hat_y[i0, k+i*N2, l]
    return Uc_hat_x

@jit(complex[:,:,:](complex[:,:,:], complex[:,:,:], int64, int64, int64), nopython=True)
def transform_Uc_zx(Uc_hat_z, Uc_hat_xr, P1, N1, N2):
    for i in xrange(P1):
        for j in xrange(i*N1, (i+1)*N1):
            i0 = j-i*N1
            for k in xrange(N2):
                for l in xrange(N1/2):
                    Uc_hat_z[i0, k, l+i*N1/2] = Uc_hat_xr[j, k, l]
    return Uc_hat_z
