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

ctypedef fused T:
    np.float64_t
    np.complex128_t


def B_matvec(np.ndarray[np.float64_t, ndim=1] uud, np.ndarray[np.float64_t, ndim=1] ud,
             np.ndarray[np.float64_t, ndim=1] lld, np.ndarray[np.float64_t, ndim=1] ld,
             np.ndarray[np.float64_t, ndim=1] dd,
             np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b):

    cdef:
        int i, j, k
        int N = v.shape[0]-2
    for j in xrange(b.shape[1]):
        for k in xrange(b.shape[2]):
            b[0, j, k] = v[0,j,k]*dd[0] + v[1,j,k]*ud[0] + v[2,j,k]*uud[0]
            b[1, j, k] = v[0,j,k]*ld[0] + v[1,j,k]*dd[1] + v[2, j, k]*ud[1] + v[3,j,k]*uud[1]
            
            b[(N-2),j,k] = v[(N-3),j,k]*lld[-2] + v[(N-2),j,k]*ld[-2] + v[(N-1),j,k]*dd[-2] + v[N, j, k]*ud[-1] 
            b[(N-1),j,k]     = v[(N-2),j,k]*lld[-1] + v[(N-1),j,k]*ld[-1] + v[N,j,k]*dd[-1]
            
    for i in xrange(2, N-2):
        for j in xrange(b.shape[1]):
            for k in xrange(b.shape[2]):
                b[i, j, k] = v[(i-2),i,j]*lld[i-2] + v[(i-1),i,j]*ld[i-1] + v[i,j,k]*dd[i] + v[(i+1),j,k]*ud[i] + v[(i+2),j,k]*uud[i]                    
   


def C_mat(np.ndarray[np.float64_t, ndim=1] K, np.ndarray[np.float64_t, ndim=1] ck,
          np.ndarray[np.float64_t, ndim=1] a_j, np.ndarray[np.float64_t, ndim=1] b_j,
          np.ndarray[np.float64_t, ndim=1] a_k, np.ndarray[np.float64_t, ndim=1] b_k,
          np.ndarray[T, ndim=2] c):

    cdef:
        int j, k
        int N   = K.shape[0]-2
        
    for k in xrange(N):
        for j in xrange(N):
            if k == j:
                c[k,j] = pi*(a_j[j]*(K[j] + 1) + b_j[j]*a_k[j]*(K[j] + 2))
            if k == j-1;    
                c[k,j] = pi*(K[j] + a_j[j]*a_k[k]*(K[j]+1) + b_j[j]*(K[j]+2) + b_j[j]*b_k[k]*(K[j]+2))
            if k == j+1:    
                c[k,j] = pi*b_j[j]*(K[j]+2) 
            if j>(k+1):
                if j%2==0:
                    c[k,j] = pi*(a_k[k]*K[j] + a_j[j]*(K[j]+1) + a_j[j]*b_k[k]*(K[j]+1) + b_j[j]*a_k[k]*(K[j]+2))
                else:
                    c[k,j] = pi*(K[j] + b_k[k]*K[j] + a_j[j]*a_k[k]*(K[j]+1) + b_j[j]*(K[j]+2) + b_j[j]*b_k[k]*(K[j]+2))
                             

def C_matvec(np.ndarray[np.float64_t, ndim=1] K, np.ndarray[np.float64_t, ndim=1] ck,
             np.ndarray[np.float64_t, ndim=1] a_j, np.ndarray[np.float64_t, ndim=1] b_j,
             np.ndarray[np.float64_t, ndim=1] a_k, np.ndarray[np.float64_t, ndim=1] b_k,
             np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b):

    cdef:
        int i, j, k, l
        int N   = v.shape[0]-2
        np.ndarray c = np.empty((N,N))

    for k in xrange(N):
        for j in xrange(N):
            if k == j:
                c[k,j] = pi*(a_j[i]*(K[i] + 1) + b_j[i]*a_k[i]*(K[i] + 2))
            if k == j-1;    
                c[k,j] = pi*(K[j] + a_j[j]*a_k[k]*(K[j]+1) + b_j[j]*(K[j]+2) + b_j[j]*b_k[k]*(K[j]+2))
            if k == j+1:    
                c[k,j] = pi*b_j[j]*(K[j]+2) 
            if j>(k+1):
                if j%2==0:
                    c[k,j] = pi*(a_k[k]*K[j] + a_j[j]*(K[j]+1) + a_j[j]*b_k[k]*(K[j]+1) + b_j[j]*a_k[k]*(K[j]+2))
                else:
                    c[k,j] = pi*(K[j] + b_k[k]*K[j] + a_j[j]*a_k[k]*(K[j]+1) + b_j[j]*(K[j]+2) + b_j[j]*b_k[k]*(K[j]+2))
    
    for k in xrange(N):
        for j in xrange(N):
            if j>(k-2):
		for i in xrange(b.shape[1]):
		    for l in xrange(b.shape[2]):
			b[k,i,l] += c[k,j]*v[j,i,l]               


def A_mat(np.ndarray[np.float64_t, ndim=1] K, np.ndarray[np.float64_t, ndim=1] ck,
          np.ndarray[np.float64_t, ndim=1] a_j, np.ndarray[np.float64_t, ndim=1] b_j,
          np.ndarray[np.float64_t, ndim=1] a_k, np.ndarray[np.float64_t, ndim=1] b_k,
          np.ndarray[T, ndim=2] c):

    cdef:
        int j, k
        int N = K.shape[0]-2
        
    for k in xrange(N):
        for j in xrange(N):
            if k == j:
                c[k,j] = 2*pi*(K[j] + 1)*(K[j] + 2)*b_j[j]
            if j>k:
                if j%2==0:
                    c[k,j] = (pi/2)*( K[j]*(K[j]**2 - K[k]**2)      + b_j[j]*(K[j]+2)*((K[j]+2)**2 - K[k]**2) + 
                              a_k[k]* (K[j]*(K[j]**2 - (K[k]+1)**2) + b_j[j]*(K[j]+2)*((K[j]+2)**2 - (K[k]+1)**2))+
                              b_k[k]* (K[j]*(K[j]**2 - (K[k]+2)**2) + b_j[j]*(K[j]+2)*((K[j]+2)**2 - (K[k]+2)**2)))
                else:
                    c[k,j] = (pi/2)*(a_j[j]*(K[j]+1)*((K[j]+1)**2 - K[k]**2) +
                              a_k[k]*a_j[j]*(K[j]+1)*((K[j]+1)**2 - (K[k]+1)**2) + 
                              b_k[k]*a_j[j]*(K[j]+1)*((K[j]+1)**2 - (K[k]+2)**2))
                             

def A_matvec(np.ndarray[np.float64_t, ndim=1] K, np.ndarray[np.float64_t, ndim=1] ck,
             np.ndarray[np.float64_t, ndim=1] a_j, np.ndarray[np.float64_t, ndim=1] b_j,
             np.ndarray[np.float64_t, ndim=1] a_k, np.ndarray[np.float64_t, ndim=1] b_k,
             np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b):

    cdef:
        int i, j, k, l
        int N   = v.shape[0]-2
        np.ndarray c = np.empty((N,N))

    for k in xrange(N):
        for j in xrange(N):
            if k == j:
                c[k,j] = 2*pi*(K[j] + 1)*(K[j] + 2)*b_j[j]
            if j>k:
                if j%2==0:
                    c[k,j] = (pi/2)*( K[j]*(K[j]**2 - K[k]**2)      + b_j[j]*(K[j]+2)*((K[j]+2)**2 - K[k]**2) + 
                              a_k[k]* (K[j]*(K[j]**2 - (K[k]+1)**2) + b_j[j]*(K[j]+2)*((K[j]+2)**2 - (K[k]+1)**2))+
                              b_k[k]* (K[j]*(K[j]**2 - (K[k]+2)**2) + b_j[j]*(K[j]+2)*((K[j]+2)**2 - (K[k]+2)**2)))
                else:
                    c[k,j] = (pi/2)*(a_j[j]*(K[j]+1)*((K[j]+1)**2 - K[k]**2) +
                              a_k[k]*a_j[j]*(K[j]+1)*((K[j]+1)**2 - (K[k]+1)**2) + 
                              b_k[k]*a_j[j]*(K[j]+1)*((K[j]+1)**2 - (K[k]+2)**2))
    
    for k in xrange(N):
        for j in xrange(N):
            if j>(k-1):
		for i in xrange(b.shape[1]):
		    for l in xrange(b.shape[2]):
			b[k,i,l] += c[k,j]*v[j,i,l]            