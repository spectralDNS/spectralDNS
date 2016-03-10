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

def B_mat(np.ndarray[np.float64_t, ndim=1] K, np.ndarray[np.float64_t, ndim=1] ck,
          np.ndarray[np.float64_t, ndim=1] a_j, np.ndarray[np.float64_t, ndim=1] b_j,
          np.ndarray[np.float64_t, ndim=1] a_k, np.ndarray[np.float64_t, ndim=1] b_k,
          np.ndarray[np.float64_t, ndim=2] c):

    cdef:
        int j, k
        int N = K.shape[0]-2
        double pi = np.pi
  
    for k in xrange(N):
        for j in xrange(N):
            if k == j:
                c[k,j] = (pi/2.)*(ck[j] + a_j[j]*a_k[k] + b_j[j]*b_k[k])
            if k == j-1:    
                c[k,j] = (pi/2.)*(a_k[k] + a_j[j]*b_k[k])
            if k == j+1:    
                c[k,j] = (pi/2.)*(a_j[j] + b_j[j]*a_k[k])
            if k==(j-2): 
                c[k,j] = (pi/2.)*b_k[k]
            if k==(j+2): 
                c[k,j] = (pi/2.)*b_j[j]
    return c

def C_mat(np.ndarray[np.float64_t, ndim=1] K,
          np.ndarray[np.float64_t, ndim=1] a_j, np.ndarray[np.float64_t, ndim=1] b_j,
          np.ndarray[np.float64_t, ndim=1] a_k, np.ndarray[np.float64_t, ndim=1] b_k,
          np.ndarray[T, ndim=2] c):

    cdef:
        int j, k
        int N = K.shape[0]-2
        
    for k in xrange(N):
        for j in xrange(N):
            if k == j:
                c[k,j] = np.pi*(a_j[j]*(K[j] + 1) + b_j[j]*a_k[j]*(K[j] + 2))
            if k == j-1:    
                c[k,j] = np.pi*(K[j] + a_j[j]*a_k[k]*(K[j]+1) + b_j[j]*(K[j]+2) + b_j[j]*b_k[k]*(K[j]+2))
            if k == j+1:    
                c[k,j] = np.pi*b_j[j]*(K[j]+2) 
            if j>(k+1):
                if (j-k)%2==0:
                    c[k,j] = np.pi*(a_k[k]*K[j] + a_j[j]*(K[j]+1) + a_j[j]*b_k[k]*(K[j]+1) + b_j[j]*a_k[k]*(K[j]+2))
                else:
                    c[k,j] = np.pi*(K[j] + b_k[k]*K[j] + a_j[j]*a_k[k]*(K[j]+1) + b_j[j]*(K[j]+2) + b_j[j]*b_k[k]*(K[j]+2))
                             
    return c


def A_mat(np.ndarray[np.float64_t, ndim=1] K, 
          np.ndarray[np.float64_t, ndim=1] a_j, np.ndarray[np.float64_t, ndim=1] b_j,
          np.ndarray[np.float64_t, ndim=1] a_k, np.ndarray[np.float64_t, ndim=1] b_k,
          np.ndarray[T, ndim=2] c):

    cdef:
        int j, k
        int N = K.shape[0]-2  
        double pi = np.pi
    for k in xrange(N):
        for j in xrange(N):
            if k == j:
                #c[k,j] = 2*pi*(K[j] + 1)*(K[j] + 2)*b_j[j]
                c[k,j] = 2*pi*(j + 1)*(j + 2)*b_j[j]
            if j>k:
                if (j-k)%2==0:
                    #c[k,j] = (np.pi/2)*( K[j]*(K[j]**2 - K[k]**2) + b_j[j]*(K[j]+2)*((K[j]+2)**2 - K[k]**2) + a_k[k]* (K[j]*(K[j]**2 - (K[k]+1)**2) + b_j[j]*(K[j]+2)*((K[j]+2)**2 - (K[k]+1)**2))+b_k[k]* (K[j]*(K[j]**2 - (K[k]+2)**2) + b_j[j]*(K[j]+2)*((K[j]+2)**2 - (K[k]+2)**2)))
                    c[k,j] = (pi/2.)*( j*(j**2 - k**2) + b_j[j]*(j+2)*((j+2)**2 - k**2) + a_k[k]* (j*(j**2 - (k+1)**2) + b_j[j]*(j+2)*((j+2)**2 - (k+1)**2))+b_k[k]*(j*(j**2 - (k+2)**2) + b_j[j]*(j+2)*((j+2)**2 - (k+2)**2)))

                else:
                    #c[k,j] = (pi/2.0)*(a_j[j]*(K[j]+1)*((K[j]+1)**2 - K[k]**2) +a_k[k]*a_j[j]*(K[j]+1)*((K[j]+1)**2 - (K[k]+1)**2) + b_k[k]*a_j[j]*(K[j]+1)*((K[j]+1)**2 - (K[k]+2)**2))
                    c[k,j] = (pi/2.0)*(a_j[j]*(j+1)*((j+1)**2 - k**2) +a_k[k]*a_j[j]*(j+1)*((j+1)**2 - (k+1)**2) + b_k[k]*a_j[j]*(j+1)*((j+1)**2 - (k+2)**2))
    return c
             
                
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


def TDMA(np.ndarray[np.float64_t, ndim=1] a, np.ndarray[np.float64_t, ndim=1] b,
         np.ndarray[np.float64_t, ndim=1] c, np.ndarray[T, ndim=3] d):

    cdef:
        unsigned int ii, jj
        
    for ii in range(d.shape[1]):
        for jj in range(d.shape[2]):
            TDMA1D(a,b,c, d[:, ii, jj].real)
            TDMA1D(a,b,c, d[:, ii, jj].imag)
    return d

def TDMA1D(np.ndarray[np.float64_t, ndim=1] a, np.ndarray[np.float64_t, ndim=1] b,
            np.ndarray[np.float64_t, ndim=1] c, np.ndarray[np.float64_t, ndim=1] d):
    
    cdef:
        int i
        int N = d.shape[0]-2


    b[0] /= a[0]
    b[1] /= a[1]

    d[0] /= a[0]
    d[1] /= a[1]

    for i in xrange(2,N):
        d[i] = ((d[i]/c[i-2])-d[i-2]) / ((a[i]/c[i-2])-b[i-2])
        if i <(N-2):
            b[i] = (b[i]/c[i-2])/((a[i]/c[i-2])-b[i-2])

    for i in xrange(N-3,-1,-1):
        d[i] = d[i] - b[i]*d[i+2]

    return d    


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
    
             