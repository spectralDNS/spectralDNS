# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 10:46:28 2015

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
            
            b[(N-1),j,k] = v[(N-3),j,k]*lld[-2] + v[(N-2),j,k]*ld[-2] + v[(N-1),j,k]*dd[-2] + v[N, j, k]*ud[-1] 
            b[N,j,k]     = v[(N-2),j,k]*lld[-1] + v[(N-1),j,k]*ld[-1] + v[N,j,k]*dd[-1]
            
    for i in xrange(2, N-1):
        for j in xrange(b.shape[1]):
            for k in xrange(b.shape[2]):
                b[i, j, k] = v[(i-2),i,j]*lld[i-2] + v[(i-1),i,j]*ld[i-1] + v[i,j,k]*dd[i] + v[(i+1),j,k]*ud[i] + v[(i+2),j,k]*uud[i]                    
   

                  
def C_matvec(np.ndarray[np.float64_t, ndim=1] ld, np.ndarray[np.float64_t, ndim=1] ud,
             uud, np.ndarray[np.float64_t, ndim=1] dd,
             np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b):

    cdef:
        int i, j, k
        int N = v.shape[0]-2 
              
    for j in xrange(b.shape[1]):
        for k in xrange(b.shape[2]):
            b[0, j, k] = v[0,j,k]*dd[0] + v[1,j,k]*ud[0] + sum(v[2:,j,k]*uud[:][0])
            b[(N-1),j,k] = v[(N-2),j,k]*ld[-2] + v[(N-1),j,k]*dd[-2] + v[N, j, k]*ud[-1] 
            b[N,j,k]     = v[(N-1),j,k]*ld[-1] + v[N,j,k]*dd[-1]

    for i in xrange(1, N-1):
        for j in xrange(b.shape[1]):
            for k in xrange(b.shape[2]):
                b[i:(N-1), j, k] = v[(i-1),j,k]*ld[i-1] + v[i,j,k]*dd[i] + v[(i+1), j, k]*ud[i] + sum(v[(i+2):,j,k]*uud[:-i][i])            
                   
 
def A_matvec(np.ndarray[np.float64_t, ndim=1] ud, np.ndarray[np.float64_t, ndim=1] dd,
             np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b):

    cdef:
        int i, j, k
        int N = v.shape[0]-2
    
    for j in xrange(b.shape[1]):
        for k in xrange(b.shape[2]):
            b[0, j, k] = v[0,j,k]*dd[0] + sum(v[1:,j,k]*ud[:][0])
            b[N,j,k]   = v[N,j,k]*dd[-1]

    for i in xrange(1, N-1):
        for j in xrange(b.shape[1]):
            for k in xrange(b.shape[2]):
                b[i:N, j, k] = v[i,j,k]*dd[i] + sum(v[(i+1):,j,k]*ud[:-i][i])            
                           
