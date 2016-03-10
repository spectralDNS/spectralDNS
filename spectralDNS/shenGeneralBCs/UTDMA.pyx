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

  
def UTDMA(np.ndarray[np.float64_t, ndim=1] a_k, np.ndarray[np.float64_t, ndim=1] b_k,
          np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b):

    cdef:
        unsigned int ii, jj
        
    for ii in range(b.shape[1]):
        for jj in range(b.shape[2]):
            UTDMA_1D(a_k,b_k, v[:, ii, jj].real, b[:, ii, jj].real)
            UTDMA_1D(a_k,b_k, v[:, ii, jj].imag, b[:, ii, jj].imag)
    return b

def UTDMA_1D(np.ndarray[np.float64_t, ndim=1] a_k, np.ndarray[np.float64_t, ndim=1] b_k,
             np.ndarray[np.float64_t, ndim=1] v, np.ndarray[np.float64_t, ndim=1] b):
    
    cdef:
        int i
        int N = v.shape[0]-2
        double pi = np.pi

    b[N-1] = v[N-1]*(2.0/pi)
    b[N-2] = v[N-2]*(2.0/pi) - a_k[N-2]*b[N-1]
    for i in xrange(N-3,-1,-1):
        b[i] = v[i]*(2.0/pi) - b[i+1]*a_k[i] - b[i+2]*b_k[i] 
    b[0] /= 2.0

    return b

def UTDMA_Neumann(np.ndarray[np.float64_t, ndim=1] b_k,
                  np.ndarray[T, ndim=3] v, np.ndarray[T, ndim=3] b):

    cdef:
        unsigned int ii, jj
        
    for ii in range(b.shape[1]):
        for jj in range(b.shape[2]):
            UTDMA_1D_Neumann(b_k, v[:, ii, jj].real, b[:, ii, jj].real)
            UTDMA_1D_Neumann(b_k, v[:, ii, jj].imag, b[:, ii, jj].imag)
    return b

def UTDMA_1D_Neumann(np.ndarray[np.float64_t, ndim=1] b_k,
                     np.ndarray[np.float64_t, ndim=1] v, np.ndarray[np.float64_t, ndim=1] b):
    
    cdef:
        int i
        int N = v.shape[0]-2
        double pi = np.pi

    b[N-1] = v[N-1]*(2.0/pi)
    b[N-2] = v[N-2]*(2.0/pi) 
    for i in xrange(N-3,-1,-1):
        b[i] = v[i]*(2.0/pi) - b[i+2]*b_k[i] 
    b[0] /= 2.0

    return b    