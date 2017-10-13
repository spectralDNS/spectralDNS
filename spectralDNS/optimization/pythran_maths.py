import numpy as np


#pythran export cross1_(float64[:,:,:,:], float64[:,:,:], float64[:,:,:], float64[:,:,:], float64[:,:,:,:])
#pythran export cross1_(float32[:,:,:,:], float32[:,:,:], float32[:,:,:], float32[:,:,:], float32[:,:,:,:])
def cross1_(c, a0, a1, a2, b):
    for i in range(b.shape[1]):
        for j in range(b.shape[2]):
            for k in range(b.shape[3]):
                a00 = a0[i,j,k]
                a11 = a1[i,j,k]
                a22 = a2[i,j,k]
                b0 = b[0,i,j,k]
                b1 = b[1,i,j,k]
                b2 = b[2,i,j,k]
                c[0,i,j,k] = a11*b2 - a22*b1
                c[1,i,j,k] = a22*b0 - a00*b2
                c[2,i,j,k] = a00*b1 - a11*b0

    return c

#pythran export cross2_(complex128[:,:,:,:], float64[:,:,:], float64[:,:,:], float64[:,:,:], complex128[:,:,:,:])
#pythran export cross2_(complex128[:,:,:,:], int64[:,:,:], int64[:,:,:], int64[:,:,:], complex128[:,:,:,:])
#pythran export cross2_(complex64[:,:,:,:], float32[:,:,:], float32[:,:,:], float32[:,:,:], complex64[:,:,:,:])
#pythran export cross2_(complex64[:,:,:,:], int32[:,:,:], int32[:,:,:], int32[:,:,:], complex64[:,:,:,:])
def cross2_(c, a0, a1, a2, b):
    for i in range(b.shape[1]):
        for j in range(b.shape[2]):
            for k in range(b.shape[3]):
                a00 = a0[i,0,0]
                a11 = a1[0,j,0]
                a22 = a2[0,0,k]
                b0 = b[0,i,j,k]
                b1 = b[1,i,j,k]
                b2 = b[2,i,j,k]
                c[0,i,j,k] = -(a11*b2.imag - a22*b1.imag) +1j*(a11*b2.real - a22*b1.real)
                c[1,i,j,k] = -(a22*b0.imag - a00*b2.imag) +1j*(a22*b0.real - a00*b2.real)
                c[2,i,j,k] = -(a00*b1.imag - a11*b0.imag) +1j*(a00*b1.real - a11*b0.real)

    return c

#pythran export add_pressure_diffusion_NS(complex128[:,:,:,:], complex128[:,:,:,:], float64, float64[:,:,:], float64[:,:,:], float64[:,:,:], float64[:,:,:], complex128[:,:,:], float64[:,:,:,:])
#pythran export add_pressure_diffusion_NS(complex64[:,:,:,:], complex64[:,:,:,:], float32, float32[:,:,:], float32[:,:,:], float32[:,:,:], float32[:,:,:], complex64[:,:,:], float32[:,:,:,:])
def add_pressure_diffusion_NS(du, u_hat, nu, ksq, k0, k1, k2, p_hat, k_over_k2):
    for i in range(ksq.shape[0]):
        k00 = k0[i,0,0]
        for j in range(ksq.shape[1]):
            k11 = k1[0,j,0]
            for k in range(ksq.shape[2]):
                k22 = k2[0,0,k]
                z = nu*ksq[i,j,k]
                p_hat[i,j,k] = du[0,i,j,k]*k_over_k2[0,i,j,k] + du[1,i,j,k]*k_over_k2[1,i,j,k] + du[2,i,j,k]*k_over_k2[2,i,j,k]
                du[0,i,j,k] -= (p_hat[i,j,k]*k00 + u_hat[0,i,j,k]*z)
                du[1,i,j,k] -= (p_hat[i,j,k]*k11 + u_hat[1,i,j,k]*z)
                du[2,i,j,k] -= (p_hat[i,j,k]*k22 + u_hat[2,i,j,k]*z)
    return du
