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

def Helmholtz_AB_Solver(np.ndarray[np.float64_t, ndim=1] K,
                        np.ndarray[np.float64_t, ndim=2] alpha, np.int64_t Neu, np.ndarray[T, ndim=3] v,
                        np.ndarray[np.float64_t, ndim=2] A, np.ndarray[np.float64_t, ndim=2] B,
                        np.ndarray[T, ndim=3] b): 


    cdef:
        unsigned int ii, jj
        
    for ii in range(b.shape[1]):
        for jj in range(b.shape[2]):
            Helmholtz_AB_1D_Solver(K, alpha[ii, jj], Neu, v[:, ii, jj].real, A, B, b[:, ii, jj].real)
            Helmholtz_AB_1D_Solver(K, alpha[ii, jj], Neu, v[:, ii, jj].imag, A, B, b[:, ii, jj].imag)

    return b

def Helmholtz_AB_1D_Solver(np.ndarray[np.float64_t, ndim=1] K, np.float64_t alpha,
                           np.int64_t Neu, np.ndarray[np.float64_t, ndim=1] v, 
                           np.ndarray[np.float64_t, ndim=2] A, np.ndarray[np.float64_t, ndim=2] B, 
                           np.ndarray[np.float64_t, ndim=1] b):

    cdef:
        int i, j, k, index1, index2
        np.float64_t s, maxU
        int N = K.shape[0]-2
        np.ndarray[np.int64_t, ndim=1] pivot
        np.ndarray[np.float64_t, ndim=1] r
        np.ndarray[np.float64_t, ndim=1] y
        np.ndarray[np.float64_t, ndim=2] U
        np.ndarray[np.float64_t, ndim=2] L
    
    U = np.zeros((N,N))
    L = np.zeros((N,N))
    r = np.zeros(N)
    y = np.zeros(N)
    pivot = np.empty(N, dtype = int)
    
    # Make L and the pivot matrix  
    for i in xrange(N):
        pivot[i] = i
        for j in xrange(N): 
            U[i,j] = -A[i,j] + alpha*B[i,j]
            if i == j:
                L[i,j] = 1.0

    # LU-decomposition
    for j in xrange(Neu, N-1):
        if j == N-2:
            maxU = np.abs(U[j,j])
            if np.abs(U[j+1,j])>= maxU:
                maxU = np.abs(U[j+1,j])
                pivot[[j,j+1]] = pivot[[j+1,j]]
                U[[pivot[j], pivot[j+1]],j:N] = U[[pivot[j+1], pivot[j]],j:N]
                L[[pivot[j], pivot[j+1]],:] = L[[pivot[j+1], pivot[j]],:]            

            L[j+1,j] = U[j+1,j]/U[j,j]
            U[j+1,j] = U[j+1,j] - L[j+1,j]*U[j,j]
            U[j+1,j+1] = U[j+1,j+1] - L[j+1,j]*U[j,j+1]           
        else:  
            index1 = j
            index2 = j        
            maxU = np.abs(U[j,j])
            for k in xrange(j+1,j+3):
                if np.abs(U[k,j])>= maxU:
                    maxU = np.abs(U[k,j])
                    index2 = k

            if index1 != index2:        
                U[[index1, index2],j:N] = U[[index2, index1],j:N]
                L[[index1, index2],:] = L[[index2, index1],:]
                pivot[[index1,index2]] = pivot[[index2,index1]]   

            for k in xrange(j+1, j+3):
                L[k,j] = U[k,j]/U[j,j]
                for i in xrange(j, N): 
                    U[k,i] = U[k,i] - L[k,j]*U[j,i] 
    # Compute Pv
    for i in range(N):
        r[i] = v[pivot[i]]

    # L - Forward substitution
    y[0] = r[0]
    y[1] = r[1] - L[1,0]*y[0]

    for i in xrange(2,N):
        s = 0.0
        for k in xrange((i-2), i):
            s += L[i,k]*y[k]
        y[i] = r[i] - s
    # U - Backward substitution
    b[N-1] = y[N-1]/U[N-1,N-1]
    for i in xrange(N-2,Neu-1,-1):
        s = 0.0
        for j in xrange(i+1,N):
            s += U[i,j]*b[j]
        b[i] = (y[i]-s) / U[i,i]    


    return b
