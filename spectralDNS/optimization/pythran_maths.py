import numpy as np


#pythran export cross1(float64[:,:,:,:], float64[:,:,:,:], float64[:,:,:,:])
#pythran export cross1(float32[:,:,:,:], float32[:,:,:,:], float32[:,:,:,:])
def cross1(c, a, b):
    c[0] = a[0] * b[2] - a[2] * b[1]
    c[1] = a[2] * b[0] - a[0] * b[2]
    c[2] = a[0] * b[1] - a[1] * b[0]
    return c

#pythran export cross2(complex128[:,:,:,:], float64[:,:,:,:], complex128[:,:,:,:])
#pythran export cross2(complex128[:,:,:,:], int64[:,:,:,:], complex128[:,:,:,:])
#pythran export cross2(complex64[:,:,:,:], float32[:,:,:,:], complex64[:,:,:,:])
#pythran export cross2(complex64[:,:,:,:], int32[:,:,:,:], complex64[:,:,:,:])
def cross2(c, a, b):
    cross1(c, a, b)
    c *= 1j
    return c

#pythran export add_pressure_diffusion_NS(complex128[:,:,:,:], complex128[:,:,:,:], float, float[:,:,:], float[:,:,:,:], complex128[:,:,:], float[:,:,:,:])
#pythran export add_pressure_diffusion_NS(complex64[:,:,:,:], complex64[:,:,:,:], float, float[:,:,:], float[:,:,:,:], complex64[:,:,:], float[:,:,:,:])
def add_pressure_diffusion_NS(du, u_hat, nu, ksq, kk, p_hat, k_over_k2):
    for i in xrange(ksq.shape[0]):
        k0 = kk[0,i,0,0]
        for j in xrange(ksq.shape[1]):
            k1 = kk[1,0,j,0]
            for k in xrange(ksq.shape[2]):
                k2 = kk[2,0,0,k]
                z = nu*ksq[i,j,k]
                p_hat[i,j,k] = np.sum(du[:3,i,j,k]*k_over_k2[:3,i,j,k])
                du[0,i,j,k] -= (p_hat[i,j,k]*k0+u_hat[0,i,j,k]*z)
                du[1,i,j,k] -= (p_hat[i,j,k]*k1+u_hat[1,i,j,k]*z)
                du[2,i,j,k] -= (p_hat[i,j,k]*k2+u_hat[2,i,j,k]*z)
    return du
