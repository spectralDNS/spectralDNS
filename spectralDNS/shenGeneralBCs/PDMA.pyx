# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 14:48:17 2015

@author: Diako Darian
"""
import numpy as np
cimport cython
cimport numpy as np
#cython: boundscheck=False
#cython: wraparound=False
#from libcpp.vector cimport vector

ctypedef fused T:
    np.float64_t
    np.complex128_t


def PDMA(np.ndarray[np.float64_t, ndim=1] a, np.ndarray[np.float64_t, ndim=1] b,
         np.ndarray[np.float64_t, ndim=1] c, np.ndarray[np.float64_t, ndim=1] d,
         np.ndarray[np.float64_t, ndim=1] e, np.ndarray[T, ndim=3] r, np.ndarray[T, ndim=3] x):

    cdef:
        unsigned int ii, jj
        
    for ii in range(r.shape[1]):
        for jj in range(r.shape[2]):
            PDMA_1D(a,b,c,d,e, r[:, ii, jj].real, x[:, ii, jj].real)
            PDMA_1D(a,b,c,d,e, r[:, ii, jj].imag, x[:, ii, jj].imag)
    return x

def PDMA_1D(np.ndarray[np.float64_t, ndim=1] a, np.ndarray[np.float64_t, ndim=1] b,
            np.ndarray[np.float64_t, ndim=1] c, np.ndarray[np.float64_t, ndim=1] d,
            np.ndarray[np.float64_t, ndim=1] e, np.ndarray[np.float64_t, ndim=1] r, np.ndarray[np.float64_t, ndim=1] x):
    
    cdef:
        int i
        int N = r.shape[0]-2 
        np.ndarray[np.float64_t, ndim=1] rho
        np.ndarray[np.float64_t, ndim=1] alpha
        np.ndarray[np.float64_t, ndim=1] beta


    rho = np.empty(N)
    alpha = np.zeros(N-1)
    beta = np.zeros(N-2)

    alpha[0] = d[0]/c[0]
    beta[0] = e[0]/c[0]
    
    alpha[1] = (d[1]-b[1]*beta[0])/(c[1]-b[1]*alpha[0])
    beta[1] = e[1]/(c[1]-b[1]*alpha[0])
    
    
    rho[0] = r[0]/c[0]
    rho[1] = (r[1] - b[1]*rho[0])/(c[1]-b[1]*alpha[0])

    for i in xrange(2,N):
        rho[i] = (r[i] - a[i]*rho[i-2] - rho[i-1]*(b[i] - a[i]*alpha[i-2])) / (c[i] - a[i]*beta[i-2] - alpha[i-1]*(b[i] - a[i]*alpha[i-2]))
        if i <= (N-2):
            alpha[i] = (d[i] - beta[i-1]*(b[i] - a[i]*alpha[i-2])) / (c[i] - a[i]*beta[i-2] - alpha[i-1]*(b[i] - a[i]*alpha[i-2]))
        if i <= (N-3):
            beta[i] = e[i]/(c[i] - a[i]*beta[i-2] - alpha[i-1]*(b[i] - a[i]*alpha[i-2]))

    for i in xrange(N-1,-1,-1):
        x[i] = rho[i]
        if i<=(N-2):
            x[i] -= alpha[i]*x[i+1]
        if i<=(N-3):
            x[i] -= beta[i]*x[i+2]    
    return x

def PDMA_3D_complex(np.ndarray[np.float64_t, ndim=1] d, 
                    np.ndarray[np.float64_t, ndim=1] f, 
                    np.ndarray[np.float64_t, ndim=1] e, 
                    np.ndarray[np.complex128_t, ndim=3] b):
    cdef:
        unsigned int N = d.shape[0]
        int k, i , j
    
    x = np.zeros((N, b.shape[1], b.shape[2]), dtype=np.complex)   

    alpha = np.zeros(N)
    gamma = np.zeros(N-1)
    delta = np.zeros(N-2)
    c     = np.zeros((N, b.shape[1], b.shape[2]), dtype=np.complex)
    z     = np.zeros((N, b.shape[1], b.shape[2]), dtype=np.complex)
    
    # Factor A=LDL'
    alpha[0] = d[0]
    gamma[0] = f[0]/alpha[0]
    delta[0] = e[0]/alpha[0]
    
    alpha[1] = d[1]-f[0]*gamma[0]
    gamma[1] = (f[1]-e[0]*gamma[0])/alpha[1]
    delta[1] = e[1]/alpha[1]
    
    for k in range(2, N-2):
        alpha[k]=d[k]-e[k-2]*delta[k-2]-alpha[k-1]*gamma[k-1]**2
        gamma[k]=(f[k]-e[k-1]*gamma[k-1])/alpha[k]
        delta[k]=e[k]/alpha[k]
    
    alpha[-2]=d[N-1]-e[N-3]*delta[N-3]-alpha[N-2]*gamma[N-2]**2
    gamma[-1]=(f[-1]-e[-1]*gamma[-2])/alpha[-2]
    alpha[-1]=d[-1]-e[-1]*delta[-1]-alpha[-2]*gamma[-1]**2
    
    # Update Lx=b, Dc=z

    z[0]=b[0]
    z[1]=b[1]-gamma[0]*z[0]

    for k in range(2,N):
        z[k]=b[k]-gamma[k-1]*z[k-1]-delta[k-2]*z[k-2]
    
    for i in range(b.shape[1]):
        for j in range(b.shape[2]):
            c[:,i,j] = z[:,i,j]/alpha[:] 

    # Backsubstitution L'x=c
    for i in range(b.shape[1]):
        for j in range(b.shape[2]):
            x[-1,i,j] = c[-1,i,j]
            x[-2,i,j] = c[-2,i,j]-gamma[-1]*x[-1,i,j]
            for k in range(N-3, -1, -1):
                x[k,i,j] = c[k,i,j]-gamma[k]*x[k+1,i,j]-delta[k]*x[k+2,i,j]
    return x
