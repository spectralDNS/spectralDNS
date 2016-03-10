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

def TDMA(np.ndarray[np.float64_t, ndim=1] a, np.ndarray[np.float64_t, ndim=1] b,
         np.ndarray[np.float64_t, ndim=1] c, np.ndarray[T, ndim=3] d):

    cdef:
        unsigned int ii, jj
        
    for ii in range(d.shape[1]):
        for jj in range(d.shape[2]):
            TDMA_1D(a,b,c, d[:, ii, jj].real)
            TDMA_1D(a,b,c, d[:, ii, jj].imag)
    return d

def TDMA_1D(np.ndarray[np.float64_t, ndim=1] a, np.ndarray[np.float64_t, ndim=1] b,
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

def TDMA_Neumann(np.ndarray[np.float64_t, ndim=1] a, np.ndarray[np.float64_t, ndim=1] b,
                 np.ndarray[np.float64_t, ndim=1] c, np.ndarray[T, ndim=3] d):

    cdef:
        unsigned int ii, jj
        
    for ii in range(d.shape[1]):
        for jj in range(d.shape[2]):
            TDMA_1D_Neumann(a,b,c, d[:, ii, jj].real)
            TDMA_1D_Neumann(a,b,c, d[:, ii, jj].imag)
    return d

def TDMA_1D_Neumann(np.ndarray[np.float64_t, ndim=1] a, np.ndarray[np.float64_t, ndim=1] b,
                    np.ndarray[np.float64_t, ndim=1] c, np.ndarray[np.float64_t, ndim=1] d):
    
    cdef:
        int i
        int N = d.shape[0]-2


    b[1] /= a[1]
    b[2] /= a[2]

    d[0] /= a[0]
    d[1] /= a[1]
    d[2] /= a[2]

    for i in xrange(3,N):
        d[i] = ((d[i]/c[i-2])-d[i-2]) / ((a[i]/c[i-2])-b[i-2])
        if i <(N-2):
            b[i] = (b[i]/c[i-2])/((a[i]/c[i-2])-b[i-2])

    for i in xrange(N-3,0,-1):
        d[i] = d[i] - b[i]*d[i+2]

    return d
