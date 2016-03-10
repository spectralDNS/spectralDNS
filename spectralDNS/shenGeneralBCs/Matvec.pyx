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

def B_matvec(np.ndarray[np.float64_t, ndim=1] K, np.ndarray[np.float64_t, ndim=2] B,
             np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b):

    cdef:
        unsigned int ii, jj
        
    for ii in range(b.shape[1]):
        for jj in range(b.shape[2]):
            B_matvec_1D(K,B, v[:, ii, jj].real, b[:, ii, jj].real)
            B_matvec_1D(K,B, v[:, ii, jj].imag, b[:, ii, jj].imag)
    return b

def B_matvec_1D(np.ndarray[np.float64_t, ndim=1] K, np.ndarray[np.float64_t, ndim=2] c,
                np.ndarray[np.float64_t, ndim=1] v, np.ndarray[np.float64_t, ndim=1] b):

    cdef:
        int i
        int N = K.shape[0]-2

    b[0] = v[0]*c[0,0] + v[1]*c[0,1] + v[2]*c[0,2]
    b[1] = v[0]*c[1,0] + v[1]*c[1,1] + v[2]*c[1,2] + v[3]*c[1,3]

    b[(N-2)] = v[(N-4)]*c[N-2,N-4] + v[(N-3)]*c[N-2,N-3] + v[(N-2)]*c[N-2,N-2] + v[N-1]*c[N-2,N-1] 
    b[(N-1)] = v[(N-3)]*c[N-1,N-3] + v[(N-2)]*c[N-1,N-2] + v[N-1]*c[N-1,N-1]
            
    for i in xrange(2, N-2):
        b[i] = v[(i-2)]*c[i,(i-2)] + v[(i-1)]*c[i,i-1] + v[i]*c[i,i] + v[(i+1)]*c[i,i+1] + v[(i+2)]*c[i,i+2] 

    return b



def C_matvec(np.ndarray[np.float64_t, ndim=1] K, np.ndarray[np.float64_t, ndim=2] C,
             np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b):

    cdef:
        unsigned int ii, jj
        
    for ii in range(b.shape[1]):
        for jj in range(b.shape[2]):
            C_matvec_1D(K,C, v[:, ii, jj].real, b[:, ii, jj].real)
            C_matvec_1D(K,C, v[:, ii, jj].imag, b[:, ii, jj].imag)
    return b

def C_matvec_1D(np.ndarray[np.float64_t, ndim=1] K, np.ndarray[np.float64_t, ndim=2] C,
                np.ndarray[np.float64_t, ndim=1] v, np.ndarray[np.float64_t, ndim=1] b):

    cdef:
        int k, j
        int N = K.shape[0]-2
    
    for k in xrange(N):
        for j in xrange(N):
            if j>(k-2):
                b[k] += v[j]*C[k,j]

    return b

def C_matvecNeumann(np.ndarray[np.float64_t, ndim=1] K, np.ndarray[np.float64_t, ndim=2] C,
             np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b):

    cdef:
        unsigned int ii, jj
        
    for ii in range(b.shape[1]):
        for jj in range(b.shape[2]):
            C_matvec_1D_Neumann(K,C, v[:, ii, jj].real, b[:, ii, jj].real)
            C_matvec_1D_Neumann(K,C, v[:, ii, jj].imag, b[:, ii, jj].imag)
    return b

def C_matvec_1D_Neumann(np.ndarray[np.float64_t, ndim=1] K, np.ndarray[np.float64_t, ndim=2] C,
                np.ndarray[np.float64_t, ndim=1] v, np.ndarray[np.float64_t, ndim=1] b):

    cdef:
        int k, j
        int N = K.shape[0]-2
    
    for k in xrange(1,N):
        if k%2 != 0:
            b[0] += C[0,k]*v[k]
        for j in xrange((k-1), N, 2):
            b[k] += v[j]*C[k,j]
      
    return b


def A_matvec(np.ndarray[np.float64_t, ndim=1] K, np.ndarray[np.float64_t, ndim=2] A,
             np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b):

    cdef:
        unsigned int ii, jj
        
    for ii in range(b.shape[1]):
        for jj in range(b.shape[2]):
            A_matvec_1D(K,A, v[:, ii, jj].real, b[:, ii, jj].real)
            A_matvec_1D(K,A, v[:, ii, jj].imag, b[:, ii, jj].imag)
    return b

def A_matvec_1D(np.ndarray[np.float64_t, ndim=1] K, np.ndarray[np.float64_t, ndim=2] A,
                np.ndarray[np.float64_t, ndim=1] v, np.ndarray[np.float64_t, ndim=1] b):

    cdef:
        int k, j
        int N = K.shape[0]-2

    for k in xrange(N):
        for j in xrange(N):
            if j>(k-1):
                b[k] += v[j]*A[k,j]       

    return b


def D_matvec(np.ndarray[np.float64_t, ndim=1] cj, 
             np.ndarray[np.float64_t, ndim=1] a_j, np.ndarray[np.float64_t, ndim=1] b_j,
             np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b):

    cdef:
        unsigned int ii, jj
        
    for ii in range(b.shape[1]):
        for jj in range(b.shape[2]):
            D_matvec_1D(cj,a_j,b_j, v[:, ii, jj].real, b[:, ii, jj].real)
            D_matvec_1D(cj,a_j,b_j, v[:, ii, jj].imag, b[:, ii, jj].imag)
    return b

def D_matvec_1D(np.ndarray[np.float64_t, ndim=1] cj,
                np.ndarray[np.float64_t, ndim=1] a_j, np.ndarray[np.float64_t, ndim=1] b_j,
                np.ndarray[np.float64_t, ndim=1] v, np.ndarray[np.float64_t, ndim=1] b):

    cdef:
        int i
        int N = b.shape[0]-2

    for i in xrange(N-2):
        b[i] = (np.pi/2.0)*(v[i]*cj[i] + v[i+1]*a_j[i] + v[i+2]*b_j[i])       
    b[N-2] = (np.pi/2.0)*(v[N-2]*cj[N-2] + v[N-1]*a_j[N-2])
    b[N-1] = (np.pi/2.0)*v[N-1]*cj[N-1]
    return b



def Helmholtz_AB_matvec(np.ndarray[np.float64_t, ndim=1] K, np.ndarray[np.float64_t, ndim=2] A, 
                        np.ndarray[np.float64_t, ndim=2] B,
                        np.ndarray[np.float64_t, ndim=2] alpha,
                        np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b): 

    cdef:
        unsigned int ii, jj
        
    for ii in range(b.shape[1]):
        for jj in range(b.shape[2]):
            Helmholtz_AB_1D(K, A, B, alpha[ii, jj], v[:, ii, jj].real, b[:, ii, jj].real)
            Helmholtz_AB_1D(K, A, B, alpha[ii, jj], v[:, ii, jj].imag, b[:, ii, jj].imag)

    return b

def Helmholtz_AB_1D(np.ndarray[np.float64_t, ndim=1] K, 
                    np.ndarray[np.float64_t, ndim=2] Amat, np.ndarray[np.float64_t, ndim=2] Bmat,
                    np.float64_t alpha, np.ndarray[np.float64_t, ndim=1] v, np.ndarray[np.float64_t, ndim=1] b):

    cdef:
        int i, j
        int N = K.shape[0]-2 

    for i in xrange(N):
        for j in xrange(N):
            if j>=i:
                b[i] += -Amat[i,j]*v[j]
                if (j-2)<=i:
                    b[i] += alpha*Bmat[i,j]*v[j]
            if (j+2)==i or (j+1)==i:
                b[i] += alpha*Bmat[i,j]*v[j]

    return b

def Helmholtz_AB_vectorNeumann(np.ndarray[np.float64_t, ndim=1] K, np.ndarray[np.float64_t, ndim=2] A, 
                        np.ndarray[np.float64_t, ndim=2] B,
                        np.ndarray[np.float64_t, ndim=2] alpha,
                        np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b): 

    cdef:
        unsigned int ii, jj
        np.ndarray[np.float64_t, ndim=2] C
        
    for ii in range(b.shape[1]):
        for jj in range(b.shape[2]):
            C = A -alpha[ii, jj]*B
            Helmholtz_AB_1D_Neumannv2(K, C, v[:, ii, jj].real, b[:, ii, jj].real)
            Helmholtz_AB_1D_Neumannv2(K, C, v[:, ii, jj].imag, b[:, ii, jj].imag)
            #Helmholtz_AB_1D_Neumann(K, A, B, alpha[ii, jj], v[:, ii, jj].real, b[:, ii, jj].real)
            #Helmholtz_AB_1D_Neumann(K, A, B, alpha[ii, jj], v[:, ii, jj].imag, b[:, ii, jj].imag)

    return b

def Helmholtz_AB_1D_Neumann(np.ndarray[np.float64_t, ndim=1] K, 
                    np.ndarray[np.float64_t, ndim=2] A, np.ndarray[np.float64_t, ndim=2] B,
                    np.float64_t alpha, np.ndarray[np.float64_t, ndim=1] v, np.ndarray[np.float64_t, ndim=1] b):

    cdef:
        int j, k
        int N = K.shape[0]-2 
        
    for k in xrange(N):
        if k>=2:
            b[k] -= alpha*B[k,k-2]*v[k-2]
        for j in xrange(k,N,2):
            b[k] += A[k,j]*v[j]
            if j <= (k+2):
                b[k] -= alpha*B[k,j]*v[j]
                
    return b

def Helmholtz_AB_1D_Neumannv2(np.ndarray[np.float64_t, ndim=1] K, 
                    np.ndarray[np.float64_t, ndim=2] A, np.ndarray[np.float64_t, ndim=1] v, np.ndarray[np.float64_t, ndim=1] b):

    cdef:
        int j, k
        int N = K.shape[0]-2 
        
    for k in xrange(N):
        for j in xrange(N):
            if j >= (k-2):
                b[k] += A[k,j]*v[j]
                
    return b
    
def Helmholtz_CB_matvec(np.ndarray[np.float64_t, ndim=1] K,np.ndarray[np.float64_t, ndim=2] C, 
                        np.ndarray[np.float64_t, ndim=2] B,
                        np.ndarray[np.float64_t, ndim=2] m, np.ndarray[np.float64_t, ndim=2] n,
                        np.ndarray[np.complex128_t, ndim=3] u_hat,
                        np.ndarray[np.complex128_t, ndim=3] v_hat,
                        np.ndarray[np.complex128_t, ndim=3] w_hat,
                        np.ndarray[np.complex128_t, ndim=3] b):                          

    cdef:
        unsigned int ii, jj
        
    for ii in range(b.shape[1]):
        for jj in range(b.shape[2]):
            Helmholtz_CB_1D_matvec(K, C, B, m[ii, jj],n[ii, jj],u_hat[:, ii, jj].real, -v_hat[:, ii, jj].imag, -w_hat[:, ii, jj].imag, b[:, ii, jj].real)
            Helmholtz_CB_1D_matvec(K, C, B, m[ii, jj],n[ii, jj],u_hat[:, ii, jj].imag,v_hat[:, ii, jj].real, w_hat[:, ii, jj].real, b[:, ii, jj].imag)

    return b

def Helmholtz_CB_1D_matvec(np.ndarray[np.float64_t, ndim=1] K, np.ndarray[np.float64_t, ndim=2] C, 
                        np.ndarray[np.float64_t, ndim=2] B,
                        np.float64_t m, np.float64_t n,
                        np.ndarray[np.float64_t, ndim=1] u_hat,
                        np.ndarray[np.float64_t, ndim=1] v_hat,
                        np.ndarray[np.float64_t, ndim=1] w_hat,
                        np.ndarray[np.float64_t, ndim=1] b):    
                            
    cdef:
        int k, j
        int N = K.shape[0]-2 
    
    for k in xrange(N):
        for j in xrange(N):
            if (j+1)>=k:
                b[k] += C[k,j]*u_hat[j]
                if (j-2)<=k:
                    b[k] += m*B[k,j]*v_hat[j] + n*B[k,j]*w_hat[j]
            elif (j+2)==k:
                b[k] += m*B[k,j]*v_hat[j] + n*B[k,j]*w_hat[j]

    return b


