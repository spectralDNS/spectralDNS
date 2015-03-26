from numba import jit, int64, int32, uint8

from numba import {0} as float
from numba import {1} as complex

@jit(float[:,:,:,:](float[:,:,:,:], float[:,:,:,:], float[:,:,:,:]), nopython=True)
def cross1(a, b, c):
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

@jit(complex[:,:,:,:](int64[:,:,:,:], complex[:,:,:,:], complex[:,:,:,:]), nopython=True)    
def cross2(a, b, c):
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

@jit(complex[:,:,:,:](complex[:,:,:,:], uint8[:,:,:]))
def dealias(du, dealias):
    for i in xrange(dealias.shape[0]):
        for j in xrange(dealias.shape[1]):
            for k in xrange(dealias.shape[2]):
                uu = dealias[i, j, k]
                du[0, i, j, k] = du[0, i, j, k]*uu
                du[1, i, j, k] = du[1, i, j, k]*uu
                du[2, i, j, k] = du[2, i, j, k]*uu
    return du

  
@jit(complex[:,:,:,:](complex[:,:,:,:], complex[:,:,:,:], 
                      int64[:,:,:], int64[:,:,:,:], complex[:,:,:],
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
